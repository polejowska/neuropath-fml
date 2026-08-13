#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import glob as _glob
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import webdataset as wds
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install webdataset first: pip install webdataset") from exc


IMG_EXTS = ("jpg", "jpeg", "png", "webp")
_D4_NONID = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
)  # dihedral-8 minus identity (every view is a real move)


# ============================================================================
# Dynamic import of the user's extraction script (identical embedding path)
# ============================================================================
def load_extractor(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"--extractor-path not found: {p}\nPoint it at the extraction .py that "
            f"defines load_foundation / extract_features."
        )
    spec = importlib.util.spec_from_file_location("bratspath_extractor", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    for name in ("load_foundation", "extract_features"):
        if not hasattr(mod, name):
            raise AttributeError(f"{p} does not define {name!r}")
    return mod


# ============================================================================
# IO utils (behaviour matches the identity extraction script)
# ============================================================================
def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    np.savez(tmp, **arrays)
    tmp_npz = Path(str(tmp) if str(tmp).endswith(".npz") else str(tmp) + ".npz")
    tmp_npz.replace(path)


def atomic_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def discover_shards(pattern: str) -> List[Path]:
    shards = sorted(Path().glob(pattern))
    if not shards:
        shards = [Path(p) for p in sorted(_glob.glob(pattern))]
    return shards


def _check_feats_finite(feats_np: np.ndarray, context: str) -> np.ndarray:
    """Strict: any NaN/Inf embedding aborts. This is a numerical fault, not a data quirk."""
    finite = np.isfinite(feats_np.reshape(feats_np.shape[0], -1)).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        raise RuntimeError(
            f"ABORT [{context}]: {n_bad}/{len(feats_np)} embeddings NaN/Inf. "
            f"If on MPS this is likely float16 AMP overflow -- re-run --no-amp."
        )
    return feats_np


def patch_rng(seed: int, key: str) -> np.random.Generator:
    """Stable per-patch RNG: reproducible across runs/restarts (Python hash() is salted)."""
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng([int(seed), int.from_bytes(h, "little")])


# ============================================================================
# Macenko stain deconvolution / transfer (structure-preserving, colour-only)
# ============================================================================
def _rgb_to_od(arr: np.ndarray, Io: float = 255.0) -> np.ndarray:
    a = arr.reshape(-1, 3).astype(np.float64)
    return -np.log((a + 1.0) / Io)


def _orient_he(HE: np.ndarray) -> np.ndarray:
    """Make stain vectors sign- and order-consistent so profiles can be aggregated.

    - Flip each column so its dominant component is positive (OD vectors live in the
      positive orthant).
    - Order columns so col0 = haematoxylin-like (larger red-channel OD), col1 = eosin.
    """
    HE = HE.astype(np.float64).copy()
    for j in range(HE.shape[1]):
        v = HE[:, j]
        if v[int(np.argmax(np.abs(v)))] < 0:
            HE[:, j] = -v
    if HE[0, 0] < HE[0, 1]:
        HE = HE[:, ::-1]
    return np.ascontiguousarray(HE)


def macenko_decompose(
    arr: np.ndarray,
    Io: float = 255.0,
    beta: float = 0.15,
    alpha: float = 1.0,
    min_tissue: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (HE [3x2], maxC [2], C [2xN]) for an HxWx3 uint8 RGB image. HE is oriented."""
    od = _rgb_to_od(arr, Io)  # (N, 3)
    mask = ~np.any(od < beta, axis=1)
    odhat = od[mask]
    if odhat.shape[0] < min_tissue:
        raise ValueError("insufficient tissue for stain estimation")

    cov = np.cov(odhat.T)
    _, V = np.linalg.eigh(cov)  # ascending eigenvalues
    Vtop = V[:, 1:3]  # top-2 eigenvectors (3x2)

    proj = odhat @ Vtop
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100.0 - alpha)
    v1 = Vtop @ np.array([np.cos(min_phi), np.sin(min_phi)])
    v2 = Vtop @ np.array([np.cos(max_phi), np.sin(max_phi)])
    HE = np.stack([v1, v2], axis=1) if v1[0] > v2[0] else np.stack([v2, v1], axis=1)
    HE = _orient_he(HE)

    C, *_ = np.linalg.lstsq(HE, od.T, rcond=None)  # (2, N)
    maxC = np.percentile(C, 99, axis=1)
    return HE.astype(np.float64), maxC.astype(np.float64), C.astype(np.float64)


def stain_is_valid(
    HE: np.ndarray, maxC: np.ndarray, min_angle_deg: float, min_maxc: float
) -> bool:
    """Reject degenerate stains: near-collinear H/E vectors or vanishing concentration."""
    if not (np.all(np.isfinite(HE)) and np.all(np.isfinite(maxC))):
        return False
    if np.any(maxC < min_maxc):
        return False
    cosang = abs(float(np.dot(HE[:, 0], HE[:, 1])))
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return ang >= min_angle_deg


def macenko_recompose_hed(
    C: np.ndarray,
    maxc_src: np.ndarray,
    he_tgt: np.ndarray,
    maxc_tgt: np.ndarray,
    shape: Tuple[int, int, int],
    rng: np.random.Generator,
    hed_scale: float = 0.0,
    hed_bias: float = 0.0,
    Io: float = 255.0,
) -> np.ndarray:
    """Recompose source concentrations onto a target stain, with per-view HED jitter.

    Colour mapping only: spatial structure (where H vs E sits) is the source's, the
    appearance (which colours, how saturated) is the target's, and HED jitter adds a
    small unique per-view perturbation so no two views share a colour.
    """
    ratio = maxc_tgt / np.maximum(maxc_src, 1e-6)  # (2,)
    C2 = C * ratio[:, None]
    if hed_scale > 0.0 or hed_bias > 0.0:
        gain = 1.0 + hed_scale * rng.uniform(-1.0, 1.0, size=2)
        bias = hed_bias * rng.uniform(-1.0, 1.0, size=2) * maxc_tgt
        C2 = C2 * gain[:, None] + bias[:, None]
    C2 = np.clip(C2, 0.0, None)
    od = he_tgt @ C2  # (3, N)
    I = Io * np.exp(-od)
    return np.clip(I, 0.0, 255.0).T.reshape(shape).astype(np.uint8)


def simple_color_jitter(
    arr: np.ndarray, rng: np.random.Generator, strength: float = 0.10
) -> np.ndarray:
    """RGB brightness/contrast/per-channel-gain jitter -- fallback when Macenko fails."""
    img = arr.astype(np.float32) / 255.0
    bright = 1.0 + strength * rng.uniform(-1.0, 1.0)
    contrast = 1.0 + strength * rng.uniform(-1.0, 1.0)
    gain = 1.0 + 0.5 * strength * rng.uniform(-1.0, 1.0, size=3)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * contrast + mean
    img = img * bright * gain
    return np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)


def dihedral(arr: np.ndarray, k: int) -> np.ndarray:
    """k in 1..7: horizontal flip (if k&4) then rotate 0/90/180/270 (k&3). Never identity."""
    if k & 4:
        arr = arr[:, ::-1]
    r = k & 3
    if r:
        arr = np.rot90(arr, r)
    return np.ascontiguousarray(arr)


def schedule_geometry(naug: int, rng: np.random.Generator) -> List[int]:
    """`naug` geometry ops, sampled WITHOUT replacement from the 7 non-identity moves;
    reshuffle and cycle if naug > 7 (still maximally spread)."""
    out: List[int] = []
    while len(out) < naug:
        perm = list(_D4_NONID)
        rng.shuffle(perm)
        out.extend(perm)
    return out[:naug]


# ============================================================================
# Unlabelled per-slide stain library  ->  continuous stain manifold
# ============================================================================
def slide_of(path: str) -> str:
    """'.../Anonymised_Image_080_patch_13328_8673.jpg' -> 'Anonymised_Image_080'."""
    stem = Path(path).stem
    m = re.match(r"^(.*)_patch_.*$", stem)
    return m.group(1) if m else stem


def build_stain_library(
    glob_pat: str,
    cache_path: Path,
    *,
    per_slide: int,
    min_patches: int,
    min_valid: int,
    max_slides: int,
    min_angle_deg: float,
    min_maxc: float,
    min_tissue: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (HE [D,3,2], maxC [D,2], slide_ids [D]) -- one robust profile per slide."""
    cfg = dict(
        glob=glob_pat,
        per_slide=per_slide,
        min_patches=min_patches,
        min_valid=min_valid,
        max_slides=max_slides,
        min_angle_deg=min_angle_deg,
        min_maxc=min_maxc,
        min_tissue=min_tissue,
        seed=seed,
    )
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        if json.loads(str(d["cfg"])) == cfg:
            print(f"[stain-lib] loaded {len(d['slide'])} cached slide profiles")
            return (
                d["he"].astype(np.float64),
                d["maxc"].astype(np.float64),
                d["slide"].astype(str),
            )
        print("[stain-lib] cache config changed -> recomputing")

    files = sorted(_glob.glob(glob_pat))
    if not files:
        raise RuntimeError(f"No unlabelled patches match {glob_pat!r}")
    by_slide: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        by_slide[slide_of(f)].append(f)

    slides = sorted(by_slide)
    rng = np.random.default_rng(seed)
    if max_slides > 0 and len(slides) > max_slides:
        slides = list(
            rng.choice(np.asarray(slides, dtype=object), size=max_slides, replace=False)
        )

    he_list, maxc_list, sid_list = [], [], []
    skipped = 0
    for sid in tqdm(slides, desc="stain library"):
        paths = by_slide[sid]
        if len(paths) < min_patches:
            skipped += 1
            continue
        pick = (
            paths
            if len(paths) <= per_slide
            else list(
                rng.choice(
                    np.asarray(paths, dtype=object), size=per_slide, replace=False
                )
            )
        )
        HEs, MCs = [], []
        for p in pick:
            try:
                arr = np.asarray(Image.open(p).convert("RGB"))
                HE, mc, _ = macenko_decompose(arr, min_tissue=min_tissue)
            except Exception:
                continue
            if stain_is_valid(HE, mc, min_angle_deg, min_maxc):
                HEs.append(HE)
                MCs.append(mc)
        if len(HEs) < min_valid:
            skipped += 1
            continue
        HE_med = _orient_he(np.median(np.stack(HEs, axis=0), axis=0))
        HE_med = HE_med / (np.linalg.norm(HE_med, axis=0, keepdims=True) + 1e-8)
        he_list.append(HE_med)
        maxc_list.append(np.median(np.stack(MCs, axis=0), axis=0))
        sid_list.append(sid)

    if not he_list:
        raise RuntimeError(
            "No usable slide stain profiles. Lower --stain-lib-min-patches / "
            "--stain-lib-min-valid or check --unlabelled-glob."
        )

    he_arr = np.stack(he_list).astype(np.float64)  # (D, 3, 2)
    maxc_arr = np.stack(maxc_list).astype(np.float64)  # (D, 2)
    sid_arr = np.asarray(sid_list, dtype=object)
    atomic_npz(
        cache_path,
        he=he_arr.astype(np.float32),
        maxc=maxc_arr.astype(np.float32),
        slide=sid_arr,
        cfg=np.asarray(json.dumps(cfg, sort_keys=True), dtype=object),
    )
    print(
        f"[stain-lib] built {len(he_list)} slide profiles "
        f"({skipped} slides skipped for too little / too poor tissue)"
    )
    return he_arr, maxc_arr, sid_arr.astype(str)


class StainManifold:
    """Sampleable stain space over the per-slide profiles.

    modes:
      discrete    -- pick one real slide profile
      interpolate -- barycentric blend of 2-3 real profiles (stays on-manifold; default)
      pca         -- Gaussian in PCA space, clipped to the empirical gamut (max novelty)
    """

    def __init__(self, HE: np.ndarray, maxc: np.ndarray, mode: str, pca_k: int):
        self.HE = HE.astype(np.float64)
        self.maxc = maxc.astype(np.float64)
        self.logmaxc = np.log(np.maximum(self.maxc, 1e-6))
        self.mode = mode
        self.D = len(HE)
        feat = np.concatenate(
            [self.HE.reshape(self.D, 6), self.logmaxc], axis=1
        )  # (D,8)
        self.mu = feat.mean(axis=0)
        Xc = feat - self.mu
        if self.D >= 2:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            k = int(min(pca_k, Vt.shape[0]))
            self.comp = Vt[:k]  # (k, 8)
            proj = Xc @ self.comp.T  # (D, k)
            self.pstd = proj.std(axis=0) + 1e-8
            self.pmin, self.pmax = proj.min(axis=0), proj.max(axis=0)
        else:
            self.comp = np.zeros((0, 8))
            self.pstd = self.pmin = self.pmax = np.zeros((0,))

    def _decode(self, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        HE = feat[:6].reshape(3, 2)
        HE = HE / (np.linalg.norm(HE, axis=0, keepdims=True) + 1e-8)
        return _orient_he(HE), np.exp(feat[6:])

    def sample(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == "discrete" or self.D == 1:
            i = int(rng.integers(0, self.D))
            return self.HE[i].copy(), self.maxc[i].copy()
        if self.mode == "interpolate":
            m = min(self.D, 2 if rng.random() < 0.7 else 3)
            idx = rng.choice(self.D, size=m, replace=False)
            w = rng.dirichlet(np.ones(m))
            HE = (self.HE[idx] * w[:, None, None]).sum(axis=0)
            HE = HE / (np.linalg.norm(HE, axis=0, keepdims=True) + 1e-8)
            lmc = (self.logmaxc[idx] * w[:, None]).sum(axis=0)
            return _orient_he(HE), np.exp(lmc)
        # pca
        z = np.clip(rng.normal(0.0, self.pstd), self.pmin, self.pmax)
        return self._decode(self.mu + z @ self.comp)


# ============================================================================
# LOCAL plan: counts + targets + source keys at (patient, slide, class)
# ============================================================================
def iter_embedding_files(root: Path):
    for pdir in sorted(root.glob("patient-*")):
        for f in sorted(pdir.glob("slide-*.npz")):
            yield pdir.name, f.stem, f


def local_target(counts: Dict[int, int], alpha: float, policy: str, cap: int) -> int:
    vals = np.asarray(list(counts.values()), dtype=np.float64)
    ref = float(vals.max()) if policy == "max" else float(np.median(vals))
    return int(min(cap, np.ceil(alpha * ref)))


def n_aug_for_cell(
    cell_count: int, target: int, min_aug: int, max_aug: int, rng: np.random.Generator
) -> int:
    """Copies per source patch to lift a cell of size `cell_count` toward `target`."""
    if cell_count <= 0:
        return min_aug
    per = max(0.0, (target - cell_count) / cell_count)
    base = int(np.floor(per))
    n = base + (1 if rng.random() < (per - base) else 0)
    return int(np.clip(n, min_aug, max_aug))


def build_local_plan(
    ident_root: Path, *, alpha: float, policy: str, cap: int, floor: int
):
    """Return (target_info, cell_counts, slide_targets).

    target_info   : key -> (patient_dir, slide_stem, y)  for scarce-source patches
    cell_counts   : (patient_dir, slide_stem, y) -> count
    slide_targets : (patient_dir, slide_stem) -> local target
    """
    target_info: Dict[str, Tuple[str, str, int]] = {}
    cell_counts: Dict[Tuple[str, str, int], int] = {}
    slide_targets: Dict[Tuple[str, str], int] = {}

    for pdir, sstem, f in tqdm(
        list(iter_embedding_files(ident_root)), desc="local plan"
    ):
        d = np.load(f, allow_pickle=True)
        names = d["names"].astype(str)
        y = d["y"].astype(np.int64)
        counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
        tgt = local_target(counts, alpha, policy, cap)
        slide_targets[(pdir, sstem)] = tgt
        for lab, n in counts.items():
            cell_counts[(pdir, sstem, lab)] = n
        for name, lab in zip(names, y):
            lab = int(lab)
            c = counts[lab]
            if floor <= c < tgt:  # minority in THIS slide (floor guards 1-off noise)
                target_info[str(name)] = (pdir, sstem, lab)

    return target_info, cell_counts, slide_targets


# ============================================================================
# Shard streaming
# ============================================================================
def _sample_key(sample: Dict[str, Any]) -> str:
    k = sample.get("__key__", "")
    return k.decode("utf-8") if isinstance(k, bytes) else str(k)


def _sample_image(sample: Dict[str, Any]) -> Optional[Image.Image]:
    for ext in IMG_EXTS:
        if ext in sample:
            raw = sample[ext]
            if isinstance(raw, (bytes, bytearray)):
                return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


def raw_shard_iter(shard: Path):
    ds = wds.WebDataset([str(shard)], shardshuffle=False, empty_check=False)
    for sample in ds:
        yield _sample_key(sample), sample


# ============================================================================
# Main augmentation + embedding pass
# ============================================================================
def shard_done_marker(parts_root: Path, shard: Path) -> Path:
    return parts_root / f"shard-{shard.stem}" / "_SHARD_DONE.json"


def process_shard(
    *,
    shard: Path,
    target_info,
    cell_counts,
    slide_targets,
    manifold: StainManifold,
    model,
    transform,
    extract_features,
    base_foundation: str,
    device: torch.device,
    parts_root: Path,
    batch_size: int,
    dtype: str,
    amp: bool,
    l2_normalize: bool,
    hed_scale: float,
    hed_bias: float,
    near_frac: float,
    near_hed_scale: float,
    stain_min_angle: float,
    stain_min_maxc: float,
    stain_min_tissue: int,
    fallback_jitter: float,
    min_aug: int,
    max_aug: int,
    seed: int,
):
    done = shard_done_marker(parts_root, shard)
    if done.exists():
        return json.loads(done.read_text())

    np_dtype = np.float16 if dtype == "float16" else np.float32
    amp_enabled = bool(amp and device.type in {"cuda", "mps"})
    amp_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.bfloat16
    context = f"aug/{shard.stem}"

    buffers: Dict[Tuple[str, str], Dict[str, list]] = defaultdict(
        lambda: {"X": [], "y": [], "names": []}
    )
    batch: List[Tuple[str, str, str, int, torch.Tensor]] = []
    n_sources = n_aug = n_fallback = 0

    def flush_batch():
        nonlocal batch
        if not batch:
            return
        tensors = torch.stack([b[4] for b in batch], dim=0).to(
            device, non_blocking=True
        )
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                feats = extract_features(
                    model, tensors, base_foundation, l2_normalize=l2_normalize
                )
        feats_np = feats.detach().cpu().float().numpy()
        feats_np = _check_feats_finite(feats_np, context).astype(np_dtype, copy=False)
        for i, (akey, pdir, sstem, lab, _) in enumerate(batch):
            b = buffers[(pdir, sstem)]
            b["X"].append(feats_np[i][None, :])
            b["y"].append(np.asarray([lab], dtype=np.int16))
            b["names"].append(akey)
        batch = []

    for key, sample in raw_shard_iter(shard):
        info = target_info.get(key)
        if info is None:
            continue  # not a scarce-cell source (filtering, not a fallback)
        pdir, sstem, lab = info
        img = _sample_image(sample)
        if img is None:
            raise RuntimeError(f"target {key!r}: no decodable image in {shard.name}.")
        arr = np.asarray(img)

        rng = patch_rng(seed, key)
        cell = (pdir, sstem, lab)
        naug = n_aug_for_cell(
            cell_counts.get(cell, 0),
            slide_targets.get((pdir, sstem), 0),
            min_aug,
            max_aug,
            rng,
        )
        geoms = schedule_geometry(naug, rng)
        n_sources += 1

        # Decompose the source ONCE; fall back to RGB jitter if it is not real tissue.
        try:
            he_src, maxc_src, C = macenko_decompose(arr, min_tissue=stain_min_tissue)
            src_ok = stain_is_valid(he_src, maxc_src, stain_min_angle, stain_min_maxc)
        except ValueError:
            src_ok = False

        for j in range(naug):
            k_geom = geoms[j]
            if not src_ok:
                aug = simple_color_jitter(dihedral(arr, k_geom), rng, fallback_jitter)
                n_fallback += 1
            else:
                if rng.random() < near_frac:
                    # "near": source's own slide stain + stronger HED jitter (within-slide variation)
                    he_t, maxc_t, hs = he_src, maxc_src, near_hed_scale
                else:
                    # "far": a different slide's stain from the manifold (cross-domain)
                    he_t, maxc_t = manifold.sample(rng)
                    hs = hed_scale
                stained = macenko_recompose_hed(
                    C,
                    maxc_src,
                    he_t,
                    maxc_t,
                    arr.shape,
                    rng,
                    hed_scale=hs,
                    hed_bias=hed_bias,
                )
                aug = dihedral(stained, k_geom)
            batch.append(
                (f"{key}#aug{j}", pdir, sstem, lab, transform(Image.fromarray(aug)))
            )
            n_aug += 1
            if len(batch) >= batch_size:
                flush_batch()

    flush_batch()

    # Write per-(patient,slide) part files for this shard.
    shard_parts_dir = parts_root / f"shard-{shard.stem}"
    tmp_dir = parts_root / f".shard-{shard.stem}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_patches = 0
    for (pdir, sstem), b in buffers.items():
        if not b["names"]:
            continue
        X = np.concatenate(b["X"], axis=0)
        y = np.concatenate(b["y"], axis=0)
        names = np.asarray(b["names"], dtype=object)
        atomic_npz(
            tmp_dir / pdir / f"{sstem}.part.npz",
            X=X,
            y=y,
            names=names,
            patient=np.asarray(pdir, dtype=object),
            slide=np.asarray(sstem, dtype=object),
        )
        n_files += 1
        n_patches += int(X.shape[0])

    meta = {
        "shard": shard.name,
        "n_source_patches": n_sources,
        "n_augmented": n_aug,
        "n_fallback_views": n_fallback,
        "n_files": n_files,
        "n_patches": n_patches,
    }
    atomic_json(tmp_dir / "_SHARD_DONE.json", meta)
    if shard_parts_dir.exists():
        shutil.rmtree(shard_parts_dir)
    tmp_dir.replace(shard_parts_dir)

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


# ============================================================================
# Consolidate + embedding-space diversity diagnostic
# ============================================================================
def _intra_source_cos_distance(X: np.ndarray, names: np.ndarray) -> Tuple[float, int]:
    """Mean pairwise cosine DISTANCE among views of the same source patch.

    Tells you whether augmentation diversity survives the foundation model or collapses.
    Assumes L2-normalised X (mean pairwise cos = (||sum||^2 - n)/(n(n-1)))."""
    src = np.asarray([n.split("#aug")[0] for n in names.astype(str)])
    Xf = X.astype(np.float64)
    total, groups = 0.0, 0
    for s in np.unique(src):
        idx = np.where(src == s)[0]
        n = idx.size
        if n < 2:
            continue
        g = Xf[idx]
        g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
        ssum = g.sum(axis=0)
        mean_cos = (float(ssum @ ssum) - n) / (n * (n - 1))
        total += 1.0 - mean_cos
        groups += 1
    return (total, groups)


def consolidate(
    parts_root: Path,
    final_root: Path,
    foundation_out: str,
    split: str,
    config: Dict[str, Any],
    diagnose: bool,
    restart_final: bool = False,
):
    manifest = final_root / "manifest.json"
    if manifest.exists() and not restart_final:
        m = json.loads(manifest.read_text())
        print(
            f"[skip] consolidated {foundation_out}/{split}: {m.get('n_patches', '?')} patches"
        )
        return m
    if restart_final and final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)

    groups: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for shard_dir in sorted(parts_root.glob("shard-*")):
        if not (shard_dir / "_SHARD_DONE.json").exists():
            continue
        for part in shard_dir.glob("patient-*/slide-*.part.npz"):
            groups[(part.parent.name, part.stem.replace(".part", ""))].append(part)
    if not groups:
        raise RuntimeError(f"No completed part files under {parts_root}")

    n_patches = n_files = 0
    diag_total, diag_groups = 0.0, 0
    for (pdir, sstem), paths in tqdm(sorted(groups.items()), desc="consolidate"):
        Xs, ys, names = [], [], []
        for p in sorted(paths):
            d = np.load(p, allow_pickle=True)
            Xs.append(d["X"])
            ys.append(d["y"])
            names.extend(d["names"].tolist())
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        names_arr = np.asarray(names, dtype=object)

        # Dedup by name (keep last), stable sort by name.
        order = np.argsort(names_arr.astype(str), kind="mergesort")
        X, y, names_arr = X[order], y[order], names_arr[order]
        if len(names_arr) > 1:
            _, ulr = np.unique(names_arr[::-1].astype(str), return_index=True)
            keep = np.sort(len(names_arr) - 1 - ulr)
            X, y, names_arr = X[keep], y[keep], names_arr[keep]

        if diagnose:
            t, g = _intra_source_cos_distance(X, names_arr)
            diag_total += t
            diag_groups += g

        atomic_npz(
            final_root / pdir / f"{sstem}.npz",
            X=X,
            y=y,
            names=names_arr,
            patient=np.asarray(pdir, dtype=object),
            slide=np.asarray(sstem, dtype=object),
        )
        n_patches += int(X.shape[0])
        n_files += 1

    meta = {
        "split": split,
        "foundation": foundation_out,
        "n_patient_slide_files": n_files,
        "n_patches": n_patches,
        "augmentation": "local_macenko_unlabelled_stain_manifold + hed_jitter + dihedral7",
        **config,
    }
    if diagnose:
        meta["mean_intra_source_cos_distance"] = (
            (diag_total / diag_groups) if diag_groups else None
        )
        meta["diag_source_groups"] = diag_groups
        if diag_groups:
            print(
                f"[diag] mean intra-source cosine distance in embedding space: "
                f"{meta['mean_intra_source_cos_distance']:.4f} "
                f"(higher = more diversity survived; over {diag_groups} sources)"
            )
    atomic_json(manifest, meta)
    print(
        f"[consolidated] {foundation_out}/{split}: {n_patches:,} augmented patches "
        f"in {n_files} files"
    )
    return meta


# ============================================================================
# CLI
# ============================================================================
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument(
        "--foundation",
        default="virchow2",
        help="Identity artifact name to read AND base model to embed with.",
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--train-glob", default="data/train/shard-*.tar")
    ap.add_argument(
        "--extractor-path",
        default="extract_embeddings.py",
        help="Path to your extraction .py (load_foundation / extract_features).",
    )
    ap.add_argument(
        "--out-suffix",
        default="stainaug_local",
        help="Output artifact = <foundation>_<out-suffix> (separate folder).",
    )

    # Unlabelled stain source
    ap.add_argument(
        "--unlabelled-glob", default="data-unlabelled/unlabelled_patches/*/*.jpg"
    )
    ap.add_argument("--stain-lib-per-slide", type=int, default=40)
    ap.add_argument("--stain-lib-min-patches", type=int, default=8)
    ap.add_argument("--stain-lib-min-valid", type=int, default=5)
    ap.add_argument("--stain-lib-max-slides", type=int, default=0, help="0 = all.")

    # LOCAL balancing (per patient, slide, class)
    ap.add_argument(
        "--target-alpha",
        type=float,
        default=0.5,
        help="Lift each minority class toward alpha * (slide reference count).",
    )
    ap.add_argument("--target-policy", choices=["max", "median"], default="max")
    ap.add_argument(
        "--target-cap",
        type=int,
        default=1500,
        help="Hard cap on the local per-cell target.",
    )
    ap.add_argument(
        "--target-floor",
        type=int,
        default=1,
        help="Only augment cells with count >= floor (guards 1-off noise).",
    )
    ap.add_argument("--min-aug-per-patch", type=int, default=1)
    ap.add_argument("--max-aug-per-patch", type=int, default=16)

    # Stain sampling
    ap.add_argument(
        "--stain-sampler",
        choices=["interpolate", "pca", "discrete"],
        default="interpolate",
    )
    ap.add_argument("--pca-k", type=int, default=6)
    ap.add_argument(
        "--hed-scale",
        type=float,
        default=0.08,
        help="Per-view HED multiplicative jitter (far/manifold views).",
    )
    ap.add_argument("--hed-bias", type=float, default=0.02)
    ap.add_argument(
        "--near-frac",
        type=float,
        default=0.3,
        help="Fraction of views kept near the source's own slide stain.",
    )
    ap.add_argument("--near-hed-scale", type=float, default=0.12)
    ap.add_argument(
        "--fallback-jitter",
        type=float,
        default=0.10,
        help="RGB jitter strength when Macenko fails on the source.",
    )

    # Stain validity gates
    ap.add_argument("--stain-min-angle-deg", type=float, default=10.0)
    ap.add_argument("--stain-min-maxc", type=float, default=0.05)
    ap.add_argument("--stain-min-tissue", type=int, default=50)

    # Embedding / runtime (mirror the extraction script)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-l2", action="store_true")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--max-shards", type=int, default=0, help="0 = all (for testing).")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--restart-final", action="store_true")
    ap.add_argument("--no-consolidate", action="store_true")
    ap.add_argument(
        "--no-diag",
        action="store_true",
        help="Skip the embedding-space diversity diagnostic.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build+print the local plan and exit (no shard reads, no model).",
    )
    args = ap.parse_args()

    device = auto_device() if args.device == "auto" else torch.device(args.device)
    print(f"device: {device}  python: {platform.python_version()}")

    artifacts = Path(args.artifacts)
    ident_root = (
        artifacts / "embeddings_by_patient_slide" / args.foundation / args.split
    )
    if not ident_root.exists():
        raise FileNotFoundError(f"Identity embeddings not found: {ident_root}")

    art = f"{args.foundation}_{args.out_suffix}"
    parts_root = artifacts / "embedding_parts" / art / args.split
    final_root = artifacts / "embeddings_by_patient_slide" / art / args.split
    if art == args.foundation + "_stainaug":
        raise SystemExit(
            "Refusing to overwrite the old 'stainaug' folder; change --out-suffix."
        )
    if args.restart:
        for r in (parts_root, final_root):
            if r.exists():
                shutil.rmtree(r)
    parts_root.mkdir(parents=True, exist_ok=True)

    # ---- LOCAL plan from existing identity embeddings ----
    target_info, cell_counts, slide_targets = build_local_plan(
        ident_root,
        alpha=args.target_alpha,
        policy=args.target_policy,
        cap=args.target_cap,
        floor=args.target_floor,
    )

    scarce_cells = sorted(
        {(p, s, y) for (p, s, y) in ((i[0], i[1], i[2]) for i in target_info.values())}
    )
    rng_est = np.random.default_rng(args.seed)
    est_aug = sum(
        n_aug_for_cell(
            cell_counts.get((p, s, y), 0),
            slide_targets.get((p, s), 0),
            args.min_aug_per_patch,
            args.max_aug_per_patch,
            rng_est,
        )
        for k, (p, s, y) in target_info.items()
    )
    zero_cells = sum(1 for (p, s), _ in slide_targets.items())  # informational only

    print(f"\nLocal scarce cells (patient,slide,class): {len(scarce_cells)}")
    print(f"Source patches to augment:  {len(target_info):,}")
    print(f"Estimated augmented patches: ~{est_aug:,}")
    print(f"Slides with a local target:  {len(slide_targets):,}")

    if args.dry_run:
        plan = {
            "granularity": "patient_slide_class",
            "target_policy": args.target_policy,
            "target_alpha": args.target_alpha,
            "target_cap": args.target_cap,
            "n_scarce_cells": len(scarce_cells),
            "n_source_patches": len(target_info),
            "est_augmented_patches": int(est_aug),
            "scarce_cells": [
                {
                    "patient": p,
                    "slide": s,
                    "class": y,
                    "count": cell_counts[(p, s, y)],
                    "slide_target": slide_targets[(p, s)],
                }
                for (p, s, y) in scarce_cells
            ],
        }
        atomic_json(parts_root / "_aug_plan.json", plan)
        print(f"[dry-run] wrote plan to {parts_root / '_aug_plan.json'}")
        return

    # ---- Unlabelled per-slide stain library -> manifold ----
    he, maxc, sids = build_stain_library(
        args.unlabelled_glob,
        parts_root / "_stain_library.npz",
        per_slide=args.stain_lib_per_slide,
        min_patches=args.stain_lib_min_patches,
        min_valid=args.stain_lib_min_valid,
        max_slides=args.stain_lib_max_slides,
        min_angle_deg=args.stain_min_angle_deg,
        min_maxc=args.stain_min_maxc,
        min_tissue=args.stain_min_tissue,
        seed=args.seed,
    )
    manifold = StainManifold(he, maxc, args.stain_sampler, args.pca_k)
    print(
        f"[stain] manifold over {manifold.D} slide profiles (mode={args.stain_sampler})"
    )

    # ---- Load model via the user's extraction script (identical embedding path) ----
    ext = load_extractor(args.extractor_path)
    model, single_transform, crop_transform = ext.load_foundation(
        args.foundation, device
    )
    transform = crop_transform  # virchow2 base = resize-only -> crop_transform
    extract_features = ext.extract_features

    shards = discover_shards(args.train_glob)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    print(f"shards: {len(shards)}")

    # ---- Main pass ----
    t0 = time.time()
    for i, shard in enumerate(shards):
        meta = process_shard(
            shard=shard,
            target_info=target_info,
            cell_counts=cell_counts,
            slide_targets=slide_targets,
            manifold=manifold,
            model=model,
            transform=transform,
            extract_features=extract_features,
            base_foundation=args.foundation,
            device=device,
            parts_root=parts_root,
            batch_size=args.batch_size,
            dtype=args.dtype,
            amp=not args.no_amp,
            l2_normalize=not args.no_l2,
            hed_scale=args.hed_scale,
            hed_bias=args.hed_bias,
            near_frac=args.near_frac,
            near_hed_scale=args.near_hed_scale,
            stain_min_angle=args.stain_min_angle_deg,
            stain_min_maxc=args.stain_min_maxc,
            stain_min_tissue=args.stain_min_tissue,
            fallback_jitter=args.fallback_jitter,
            min_aug=args.min_aug_per_patch,
            max_aug=args.max_aug_per_patch,
            seed=args.seed,
        )
        print(
            f"[shard {i + 1}/{len(shards)}] {shard.name}: "
            f"{meta['n_source_patches']} sources -> {meta['n_augmented']} aug "
            f"({meta['n_fallback_views']} fallback, {meta['n_patches']} written)"
        )

    print(f"augmentation pass done in {time.time() - t0:.1f}s")

    if not args.no_consolidate:
        config = {
            "granularity": "patient_slide_class",
            "target_alpha": args.target_alpha,
            "target_policy": args.target_policy,
            "target_cap": args.target_cap,
            "target_floor": args.target_floor,
            "min_aug_per_patch": args.min_aug_per_patch,
            "max_aug_per_patch": args.max_aug_per_patch,
            "stain_sampler": args.stain_sampler,
            "n_stain_profiles": int(manifold.D),
            "hed_scale": args.hed_scale,
            "hed_bias": args.hed_bias,
            "near_frac": args.near_frac,
            "near_hed_scale": args.near_hed_scale,
            "l2_normalize": not args.no_l2,
            "dtype": args.dtype,
            "base_foundation": args.foundation,
            "seed": args.seed,
        }
        consolidate(
            parts_root,
            final_root,
            art,
            args.split,
            config,
            diagnose=not args.no_diag,
            restart_final=args.restart_final,
        )


if __name__ == "__main__":
    main()

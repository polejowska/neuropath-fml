#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib.util
import io
import json
import os
import platform
import shutil
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import webdataset as wds
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install webdataset first: pip install webdataset") from exc


IMG_EXTS = ("jpg", "jpeg", "png", "webp")


# ----------------------------------------------------------------------------
# Dynamic import of the user's extraction script (for identical embedding path)
# ----------------------------------------------------------------------------
def load_extractor(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"--extractor-path not found: {p}\n"
            f"Point it at the extraction .py you pasted (the one defining "
            f"load_foundation / extract_features)."
        )
    spec = importlib.util.spec_from_file_location("bratspath_extractor", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    for name in ("load_foundation", "extract_features"):
        if not hasattr(mod, name):
            raise AttributeError(f"{p} does not define {name!r}")
    return mod


# ----------------------------------------------------------------------------
# Small local IO utils (behaviour matches the extraction script)
# ----------------------------------------------------------------------------
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
    import glob

    shards = sorted(Path().glob(pattern))
    if not shards:
        shards = [Path(p) for p in sorted(glob.glob(pattern))]
    return shards


def _check_feats_finite(feats_np: np.ndarray, context: str) -> np.ndarray:
    """Strict: any NaN/Inf embedding aborts the run. No zeroing, no partial keep."""
    finite = np.isfinite(feats_np.reshape(feats_np.shape[0], -1)).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        raise RuntimeError(
            f"ABORT [{context}]: {n_bad}/{len(feats_np)} embeddings NaN/Inf. "
            f"No fallback (no zeroing). If on MPS this is likely float16 AMP overflow "
            f"-- re-run with --no-amp --batch-size 128."
        )
    return feats_np


# ----------------------------------------------------------------------------
# Macenko stain deconvolution / transfer  (structure-preserving, colour-only)
# ----------------------------------------------------------------------------
def _rgb_to_od(arr: np.ndarray, Io: float = 255.0) -> np.ndarray:
    a = arr.reshape(-1, 3).astype(np.float64)
    return -np.log((a + 1.0) / Io)


def macenko_decompose(
    arr: np.ndarray, Io: float = 255.0, beta: float = 0.15, alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (HE [3x2], maxC [2], C [2xN]) for an HxWx3 uint8 RGB image."""
    od = _rgb_to_od(arr, Io)  # (N, 3)
    mask = ~np.any(od < beta, axis=1)
    odhat = od[mask]
    if odhat.shape[0] < 50:
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
    # Order so column 0 = haematoxylin-like (larger R-channel OD), 1 = eosin-like.
    HE = np.stack([v1, v2], axis=1) if v1[0] > v2[0] else np.stack([v2, v1], axis=1)

    C, *_ = np.linalg.lstsq(HE, od.T, rcond=None)  # (2, N)
    maxC = np.percentile(C, 99, axis=1)
    return HE.astype(np.float64), maxC.astype(np.float64), C.astype(np.float64)


def macenko_recompose(
    C: np.ndarray,
    maxc_src: np.ndarray,
    he_tgt: np.ndarray,
    maxc_tgt: np.ndarray,
    shape: Tuple[int, int, int],
    Io: float = 255.0,
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Recompose an image from source concentrations onto a target stain."""
    ratio = maxc_tgt / np.maximum(maxc_src, 1e-6)
    if jitter > 0.0 and rng is not None:
        ratio = ratio * rng.uniform(1.0 - jitter, 1.0 + jitter, size=2)
    C2 = C * ratio[:, None]
    od = he_tgt @ C2  # (3, N)
    I = Io * np.exp(-od)
    I = np.clip(I, 0.0, 255.0).T.reshape(shape)
    return I.astype(np.uint8)


def dihedral(arr: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7: optional horizontal flip then rotation by 0/90/180/270."""
    if k & 4:
        arr = arr[:, ::-1]
    r = k & 3
    if r:
        arr = np.rot90(arr, r)
    return np.ascontiguousarray(arr)


# ----------------------------------------------------------------------------
# Walk existing identity embeddings: counts, target keys, donor candidates
# ----------------------------------------------------------------------------
def iter_embedding_files(root: Path):
    for pdir in sorted(root.glob("patient-*")):
        for f in sorted(pdir.glob("slide-*.npz")):
            yield pdir.name, f.stem, f


def build_plan(
    ident_root: Path,
    scarce_threshold: int,
    donor_min_cell: int,
    donors_per_cell: int,
    num_donors: int,
    seed: int,
):
    """Return (target_info, cell_counts, donor_keys, donor_meta).

    target_info : dict key -> (patient_dir, slide_stem, y_int)   (scarce sources only)
    cell_counts : dict (patient_dir, y) -> count
    donor_keys  : set of keys whose images we need for the donor pool
    donor_meta  : dict key -> (patient_dir, y)
    """
    rng = np.random.default_rng(seed)

    # Pass 1: per-(patient, class) counts (read y only).
    cell_counts: Dict[Tuple[str, int], int] = defaultdict(int)
    for pdir, sstem, f in tqdm(
        list(iter_embedding_files(ident_root)), desc="counting cells"
    ):
        y = np.load(f)["y"].astype(np.int64)
        for lab, n in zip(*np.unique(y, return_counts=True)):
            cell_counts[(pdir, int(lab))] += int(n)

    scarce_cells = {c for c, n in cell_counts.items() if 0 < n < scarce_threshold}
    donor_cells = {c for c, n in cell_counts.items() if n >= donor_min_cell}

    # Pass 2: collect target keys (scarce) and reservoir-sample donor candidates.
    target_info: Dict[str, Tuple[str, str, int]] = {}
    donor_reservoir: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    donor_meta: Dict[str, Tuple[str, int]] = {}

    for pdir, sstem, f in tqdm(
        list(iter_embedding_files(ident_root)), desc="selecting patches"
    ):
        d = np.load(f, allow_pickle=True)
        names = d["names"].astype(str)
        y = d["y"].astype(np.int64)
        for k, lab in zip(names, y):
            lab = int(lab)
            cell = (pdir, lab)
            if cell in scarce_cells:
                target_info[str(k)] = (pdir, sstem, lab)
            if cell in donor_cells:
                res = donor_reservoir[cell]
                if len(res) < donors_per_cell:
                    res.append(str(k))
                    donor_meta[str(k)] = (pdir, lab)

    # Build the donor pool with patient/class diversity, capped at num_donors.
    all_cands = []
    for cell, keys in donor_reservoir.items():
        for k in keys:
            all_cands.append(k)
    rng.shuffle(all_cands)
    donor_keys = set(all_cands[:num_donors])
    donor_meta = {k: donor_meta[k] for k in donor_keys}

    return target_info, dict(cell_counts), donor_keys, donor_meta


def n_aug_for_cell(
    cell_count: int, target: int, min_aug: int, max_aug: int, rng: np.random.Generator
) -> int:
    """Copies per source patch to lift a cell of size cell_count toward `target`."""
    if cell_count <= 0:
        return min_aug
    per = max(0.0, (target - cell_count) / cell_count)
    base = int(np.floor(per))
    frac = per - base
    n = base + (1 if rng.random() < frac else 0)
    return int(np.clip(n, min_aug, max_aug))


# ----------------------------------------------------------------------------
# Shard streaming: collect only the images we actually need
# ----------------------------------------------------------------------------
def _sample_key(sample: Dict[str, Any]) -> str:
    k = sample.get("__key__", "")
    if isinstance(k, bytes):
        k = k.decode("utf-8")
    return str(k)


def _sample_image(sample: Dict[str, Any]) -> Optional[Image.Image]:
    for ext in IMG_EXTS:
        if ext in sample:
            raw = sample[ext]
            if isinstance(raw, (bytes, bytearray)):
                return Image.open(io.BytesIO(raw)).convert("RGB")
    return None


def raw_shard_iter(shard: Path):
    """Yield (key, sample_dict) WITHOUT decoding images (decode lazily on demand)."""
    ds = wds.WebDataset([str(shard)], shardshuffle=False, empty_check=False)
    for sample in ds:
        yield _sample_key(sample), sample


def collect_donor_params(
    shards: List[Path],
    donor_keys: set,
    donor_meta: Dict[str, Tuple[str, int]],
    cache_path: Path,
):
    """Stream shards (early-stop) to compute Macenko params for each donor image."""
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        cached_keys = set(d["key"].astype(str).tolist())
        if cached_keys == set(donor_keys):
            print(f"[donors] loaded {len(cached_keys)} cached donor stains")
            return (
                d["he"].astype(np.float64),
                d["maxc"].astype(np.float64),
                d["patient"].astype(str),
                d["cls"].astype(np.int64),
            )
        # Requested donor key set differs from cache -> recompute below.

    need = set(donor_keys)
    he_list, maxc_list, pat_list, cls_list, key_list = [], [], [], [], []
    pbar = tqdm(total=len(need), desc="donor stains")
    for shard in shards:
        if not need:
            break
        for key, sample in raw_shard_iter(shard):
            if key not in need:
                continue
            need.discard(key)
            pbar.update(1)
            img = _sample_image(sample)
            if img is None:
                raise RuntimeError(
                    f"donor {key!r}: no decodable image in shard {shard.name}."
                )
            try:
                he, maxc, _ = macenko_decompose(np.asarray(img))
            except ValueError as e:
                raise RuntimeError(
                    f"donor {key!r}: stain estimation failed ({e}). No fallback -- "
                    f"raise --donor-min-cell so donors come from denser tissue, or "
                    f"drop this key."
                ) from e
            pdir, lab = donor_meta[key]
            he_list.append(he)
            maxc_list.append(maxc)
            pat_list.append(pdir)
            cls_list.append(int(lab))
            key_list.append(key)
            if not need:
                break
    pbar.close()

    if not he_list:
        raise RuntimeError("No donor stains could be estimated. Check shards / keys.")

    he_arr = np.stack(he_list).astype(np.float32)  # (D, 3, 2)
    maxc_arr = np.stack(maxc_list).astype(np.float32)  # (D, 2)
    pat_arr = np.asarray(pat_list, dtype=object)
    cls_arr = np.asarray(cls_list, dtype=np.int64)
    key_arr = np.asarray(key_list, dtype=object)
    atomic_npz(
        cache_path, he=he_arr, maxc=maxc_arr, patient=pat_arr, cls=cls_arr, key=key_arr
    )
    print(
        f"[donors] estimated {len(he_list)} donor stains "
        f"({len(set(pat_list))} patients, {len(set(cls_list))} classes)"
    )
    return (
        he_arr.astype(np.float64),
        maxc_arr.astype(np.float64),
        pat_arr.astype(str),
        cls_arr,
    )


class DonorPool:
    def __init__(self, he, maxc, patient, cls):
        self.he = he  # (D, 3, 2)
        self.maxc = maxc  # (D, 2)
        self.patient = np.asarray(patient)
        self.cls = np.asarray(cls)
        self.n = len(he)

    def pick(self, src_patient: str, src_cls: int, rng: np.random.Generator) -> int:
        idx = np.where((self.patient != src_patient) & (self.cls != src_cls))[0]
        if idx.size == 0:
            raise RuntimeError(
                f"No cross-patient-cross-class donor for source patient={src_patient!r} "
                f"class={src_cls}. No fallback (won't relax the constraint). Donor pool "
                f"is too small or not diverse -- increase --num-donors / --donors-per-cell "
                f"or lower --donor-min-cell."
            )
        return int(rng.choice(idx))


# ----------------------------------------------------------------------------
# Main augmentation + embedding pass
# ----------------------------------------------------------------------------
def shard_done_marker(parts_root: Path, shard: Path) -> Path:
    return parts_root / f"shard-{shard.stem}" / "_SHARD_DONE.json"


def process_shard(
    *,
    shard: Path,
    target_info: Dict[str, Tuple[str, str, int]],
    cell_counts: Dict[Tuple[str, int], int],
    donors: DonorPool,
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
    stain_jitter: float,
    target_per_cell: int,
    min_aug: int,
    max_aug: int,
    seed: int,
    shard_idx: int,
):
    done = shard_done_marker(parts_root, shard)
    if done.exists():
        return json.loads(done.read_text())

    rng = np.random.default_rng([seed, shard_idx])
    np_dtype = np.float16 if dtype == "float16" else np.float32
    amp_enabled = bool(amp and device.type in {"cuda", "mps"})
    amp_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.bfloat16
    context = f"aug/{shard.stem}"

    # Per-(patient,slide) accumulation for this shard.
    buffers: Dict[Tuple[str, str], Dict[str, list]] = defaultdict(
        lambda: {"X": [], "y": [], "names": []}
    )
    batch: List[Tuple[str, str, str, int, torch.Tensor]] = []
    n_sources = n_aug = 0

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
            continue  # not a scarce-cell source; filtering, not a fallback
        pdir, sstem, lab = info

        img = _sample_image(sample)
        if img is None:
            raise RuntimeError(
                f"target {key!r}: no decodable image in shard {shard.name}."
            )
        arr = np.asarray(img)
        try:
            he_src, maxc_src, C = macenko_decompose(arr)
        except ValueError as e:
            raise RuntimeError(
                f"target {key!r}: stain estimation failed ({e}). No fallback -- "
                f"exclude this patch upstream if it is not real tissue."
            ) from e

        n_sources += 1
        cell = (pdir, lab)
        naug = n_aug_for_cell(
            cell_counts.get(cell, 0), target_per_cell, min_aug, max_aug, rng
        )

        for j in range(naug):
            k_dih = int(rng.integers(0, 8))
            d_idx = donors.pick(pdir, lab, rng)
            stained = macenko_recompose(
                C,
                maxc_src,
                donors.he[d_idx],
                donors.maxc[d_idx],
                arr.shape,
                jitter=stain_jitter,
                rng=rng,
            )
            aug = dihedral(stained, k_dih)
            tensor = transform(Image.fromarray(aug))
            batch.append((f"{key}#aug{j}", pdir, sstem, lab, tensor))
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
        out = tmp_dir / pdir / f"{sstem}.part.npz"
        atomic_npz(
            out,
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


def consolidate(
    parts_root: Path,
    final_root: Path,
    foundation: str,
    split: str,
    config: Dict[str, Any],
    restart_final: bool = False,
):
    manifest = final_root / "manifest.json"
    if manifest.exists() and not restart_final:
        m = json.loads(manifest.read_text())
        print(
            f"[skip] consolidated {foundation}_stainaug/{split}: {m.get('n_patches', '?')} patches"
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

        out = final_root / pdir / f"{sstem}.npz"
        atomic_npz(
            out,
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
        "foundation": f"{foundation}_stainaug",
        "n_patient_slide_files": n_files,
        "n_patches": n_patches,
        "augmentation": "macenko_cross_patient_cross_class_stain_transfer + dihedral8",
        **config,
    }
    atomic_json(manifest, meta)
    print(
        f"[consolidated] {foundation}_stainaug/{split}: {n_patches:,} augmented "
        f"patches in {n_files} files"
    )
    return meta


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
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
        help="Path to your extraction .py (defines load_foundation / extract_features).",
    )

    # Scarcity / augmentation volume
    ap.add_argument(
        "--scarce-threshold",
        type=int,
        default=1000,
        help="Augment cells with 0 < count < threshold.",
    )
    ap.add_argument(
        "--target-per-cell",
        type=int,
        default=1000,
        help="Aim to lift each scarce cell toward this many patches.",
    )
    ap.add_argument("--min-aug-per-patch", type=int, default=1)
    ap.add_argument("--max-aug-per-patch", type=int, default=16)

    # Donor pool
    ap.add_argument("--num-donors", type=int, default=256)
    ap.add_argument("--donors-per-cell", type=int, default=2)
    ap.add_argument(
        "--donor-min-cell",
        type=int,
        default=200,
        help="Only draw donor stains from cells with >= this many patches.",
    )

    # Stain
    ap.add_argument(
        "--stain-jitter",
        type=float,
        default=0.05,
        help="Per-copy multiplicative jitter on target concentrations (0 disables).",
    )

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
        "--dry-run",
        action="store_true",
        help="Print the augmentation plan and exit (no shard reads).",
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

    art = f"{args.foundation}_stainaug"
    parts_root = artifacts / "embedding_parts" / art / args.split
    final_root = artifacts / "embeddings_by_patient_slide" / art / args.split
    if args.restart:
        for r in (parts_root, final_root):
            if r.exists():
                shutil.rmtree(r)
    parts_root.mkdir(parents=True, exist_ok=True)

    # ---- Build the plan from existing embeddings ----
    target_info, cell_counts, donor_keys, donor_meta = build_plan(
        ident_root,
        args.scarce_threshold,
        args.donor_min_cell,
        args.donors_per_cell,
        args.num_donors,
        args.seed,
    )
    scarce_cells = sorted(
        {(p, y) for (p, y) in ((info[0], info[2]) for info in target_info.values())}
    )
    est_aug = 0
    rng_est = np.random.default_rng(args.seed)
    for k, (pdir, sstem, lab) in target_info.items():
        est_aug += n_aug_for_cell(
            cell_counts.get((pdir, lab), 0),
            args.target_per_cell,
            args.min_aug_per_patch,
            args.max_aug_per_patch,
            rng_est,
        )

    print(f"\nScarce cells (<{args.scarce_threshold}): {len(scarce_cells)}")
    print(f"Source patches to augment:  {len(target_info):,}")
    print(f"Estimated augmented patches: ~{est_aug:,}")
    print(f"Donor pool size requested:  {len(donor_keys)}")

    if args.dry_run:
        plan = {
            "scarce_threshold": args.scarce_threshold,
            "n_scarce_cells": len(scarce_cells),
            "n_source_patches": len(target_info),
            "est_augmented_patches": int(est_aug),
            "scarce_cells": [
                {"patient": p, "class": y, "count": cell_counts[(p, y)]}
                for (p, y) in scarce_cells
            ],
        }
        atomic_json(parts_root / "_aug_plan.json", plan)
        print(f"[dry-run] wrote plan to {parts_root / '_aug_plan.json'}")
        return

    # ---- Load model via the user's extraction script (identical embedding path) ----
    ext = load_extractor(args.extractor_path)
    model, single_transform, crop_transform = ext.load_foundation(
        args.foundation, device
    )
    # virchow2 base = resize-only -> the extraction script uses crop_transform.
    transform = crop_transform
    extract_features = ext.extract_features

    # ---- Donor stains ----
    shards = discover_shards(args.train_glob)
    if args.max_shards > 0:
        shards = shards[: args.max_shards]
    print(f"shards: {len(shards)}")
    he, maxc, dpat, dcls = collect_donor_params(
        shards, donor_keys, donor_meta, parts_root / "_donors.npz"
    )
    donors = DonorPool(he, maxc, dpat, dcls)

    # ---- Main pass ----
    t0 = time.time()
    for shard_idx, shard in enumerate(shards):
        meta = process_shard(
            shard=shard,
            target_info=target_info,
            cell_counts=cell_counts,
            donors=donors,
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
            stain_jitter=args.stain_jitter,
            target_per_cell=args.target_per_cell,
            min_aug=args.min_aug_per_patch,
            max_aug=args.max_aug_per_patch,
            seed=args.seed,
            shard_idx=shard_idx,
        )
        print(
            f"[shard {shard_idx + 1}/{len(shards)}] {shard.name}: "
            f"{meta['n_source_patches']} sources -> {meta['n_augmented']} aug "
            f"({meta['n_patches']} written)"
        )

    print(f"augmentation pass done in {time.time() - t0:.1f}s")

    if not args.no_consolidate:
        config = {
            "scarce_threshold": args.scarce_threshold,
            "target_per_cell": args.target_per_cell,
            "min_aug_per_patch": args.min_aug_per_patch,
            "max_aug_per_patch": args.max_aug_per_patch,
            "num_donors": int(donors.n),
            "stain_jitter": args.stain_jitter,
            "l2_normalize": not args.no_l2,
            "dtype": args.dtype,
            "base_foundation": args.foundation,
            "seed": args.seed,
        }
        consolidate(
            parts_root,
            final_root,
            args.foundation,
            args.split,
            config,
            restart_final=args.restart_final,
        )


if __name__ == "__main__":
    main()

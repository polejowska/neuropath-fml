#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import re
import shutil
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import timm
    from timm.data import create_transform, resolve_model_data_config
except Exception as exc:
    raise RuntimeError("Install timm first: pip install timm") from exc

try:
    import webdataset as wds
except Exception as exc:
    raise RuntimeError("Install webdataset first: pip install webdataset") from exc

from torch.utils.data import DataLoader

DEFAULT_MAPPING_CSV = "data/BraTS-Path-2026-Train-Patch-Patient-Slide-Mapping.csv"
DEFAULT_TRAIN_GLOB = "data/train/shard-*.tar"
DEFAULT_VAL_GLOB = "data/val-shard-*.tar"
DEFAULT_ARTIFACTS = "artifacts"

FOUNDATION_SPECS = {
    "h0mini": {
        "hf_id": "hf-hub:bioptimus/H0-mini",
        "input_size": 224,
        "vit_patch_size": 14,
        "embedding_dim": 768,
    },
    "hoptimus1": {
        # Gated Hugging Face model: run `huggingface-cli login` and accept the
        # bioptimus/H-optimus-1 terms before first use.
        "hf_id": "hf-hub:bioptimus/H-optimus-1",
        "input_size": 224,
        "vit_patch_size": 14,
        "embedding_dim": 1536,
        # Bioptimus model-card normalization for H-optimus-1. Keep this explicit
        # instead of relying on timm defaults.
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
    },
    "ctranspath": {
        "hf_id": "hf-hub:1aurent/swin_tiny_patch4_window7_224.CTransPath",
        "input_size": 224,
        "vit_patch_size": 4,
        "embedding_dim": 768,
    },
    "uni2": {
        "hf_id": "hf-hub:MahmoodLab/UNI2-h",
        "input_size": 224,
        "vit_patch_size": 14,
        "embedding_dim": 1536,
    },
    "virchow2": {
        "hf_id": "hf-hub:paige-ai/Virchow2",
        "input_size": 224,
        "vit_patch_size": 14,
        "embedding_dim": 2560,
    },
}

# Crop position names in the order produced by _crop_boxes().
CROP_NAMES: Dict[int, List[str]] = {
    1: ["center"],
    3: ["center", "top_left", "bottom_right"],
    5: ["center", "top_left", "top_right", "bottom_left", "bottom_right"],
    9: [
        "top_left",
        "top_center",
        "top_right",
        "mid_left",
        "center",
        "mid_right",
        "bottom_left",
        "bottom_center",
        "bottom_right",
    ],
}

# All valid single-crop names across mc5 and mc9 (used in regex).
_ALL_CROP_NAMES = sorted({n for names in CROP_NAMES.values() for n in names})
_CROP_NAMES_RE = "|".join(_ALL_CROP_NAMES)

# Track (w, h, crop_size, n_crops) combos already warned about crop collapse
# so the warning fires at most once per unique combination per process.
_warned_crop_collapse: set = set()


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_foundation_variant(name: str) -> Dict[str, Any]:
    """Parse a foundation/variant name into extraction parameters.

    Returns a dict with:
      artifact_name   : output directory name
      base            : base model key in FOUNDATION_SPECS
      n_crops         : total crop positions in the grid
      combine         : "single" | "resize_only" | "single_crop" | "percrop" | "mean" | "concat"
      crop_index      : int | None  (only for combine=="single_crop")
      crop_name       : str | None  (human-readable position)
    """
    name = str(name).strip()

    # Base model in this script means resize-only full-field extraction.
    # Artifact name intentionally stays as the base name, e.g. "h0mini".
    if name in FOUNDATION_SPECS:
        return {
            "artifact_name": name,
            "base": name,
            "n_crops": 1,
            "combine": "resize_only",
            "crop_index": None,
            "crop_name": None,
        }

    base_pat = r"(?P<base>h0mini|hoptimus1|ctranspath|uni2|virchow2)"

    # Full-patch resize-only variant: h0mini_resize224, uni2_resize224, etc.
    # This intentionally bypasses timm's eval transform, because those transforms
    # commonly resize then centre-crop. Here we keep the full field of view and
    # squeeze the whole input image to the model's 224x224 input.
    m = re.match(rf"^{base_pat}_resize(?P<size>\d+)$", name)
    if m:
        base = m.group("base")
        resize_size = int(m.group("size"))
        expected_size = int(FOUNDATION_SPECS[base]["input_size"])
        if resize_size != expected_size:
            raise ValueError(
                f"Resize-only variant {name!r} requested {resize_size}, but {base} "
                f"expects input_size={expected_size}. Use {base}_resize{expected_size}."
            )
        # Compatibility alias: accept h0mini_resize224, but write under h0mini
        # so partial outputs from the older resize-only script can be resumed.
        return {
            "artifact_name": base,
            "base": base,
            "n_crops": 1,
            "combine": "resize_only",
            "crop_index": None,
            "crop_name": None,
        }

    # Single specific crop: h0mini_mc5_center, uni2_mc5_top_right, etc.
    m = re.match(
        rf"^{base_pat}_mc(?P<n>\d+)_(?P<crop>{_CROP_NAMES_RE})$",
        name,
    )
    if m:
        base = m.group("base")
        n_crops = int(m.group("n"))
        crop_name = m.group("crop")
        if n_crops not in CROP_NAMES:
            raise ValueError(
                f"Unsupported n_crops={n_crops}; supported: {sorted(CROP_NAMES)}"
            )
        valid = CROP_NAMES[n_crops]
        if crop_name not in valid:
            raise ValueError(
                f"Crop '{crop_name}' not valid for mc{n_crops}. "
                f"Valid positions: {valid}"
            )
        crop_index = valid.index(crop_name)
        return {
            "artifact_name": name,
            "base": base,
            "n_crops": n_crops,
            "combine": "single_crop",
            "crop_index": crop_index,
            "crop_name": crop_name,
        }

    # All crops together (per-crop storage): h0mini_mc5, uni2_mc9, etc.
    m = re.match(rf"^{base_pat}_mc(?P<n>\d+)$", name)
    if m:
        base = m.group("base")
        n_crops = int(m.group("n"))
        if n_crops not in CROP_NAMES:
            raise ValueError(f"Unsupported n_crops={n_crops}")
        return {
            "artifact_name": name,
            "base": base,
            "n_crops": n_crops,
            "combine": "percrop",
            "crop_index": None,
            "crop_name": None,
        }

    # Legacy combined: h0mini_mc5_mean, h0mini_mc5_concat.
    m = re.match(rf"^{base_pat}_mc(?P<n>\d+)_(?P<combine>mean|concat)$", name)
    if m:
        base = m.group("base")
        n_crops = int(m.group("n"))
        combine = m.group("combine")
        if n_crops not in CROP_NAMES:
            raise ValueError(f"Unsupported n_crops={n_crops}")
        return {
            "artifact_name": name,
            "base": base,
            "n_crops": n_crops,
            "combine": combine,
            "crop_index": None,
            "crop_name": None,
        }

    raise ValueError(
        f"Unknown foundation/variant {name!r}.\n"
        f"Examples:\n"
        f"  h0mini                    full patch -> 224x224 resize, no crop; artifact h0mini\n"
        f"  hoptimus1                 H-optimus-1 full patch -> 224x224 resize, no crop; artifact hoptimus1\n"
        f"  hoptimus1_mc5_center      H-optimus-1 centre crop from mc5 grid; artifact hoptimus1_mc5_center\n"
        f"  h0mini_resize224          alias for h0mini in this script; artifact h0mini\n"
        f"  h0mini_mc5_center         centre crop only\n"
        f"  h0mini_mc5_top_right      top-right crop only\n"
        f"  h0mini_mc5                all 5 crops stored as [n,5,dim]\n"
        f"  h0mini_mc5_mean           mean of 5 crops (legacy)\n"
        f"Valid single-crop positions for mc5: {CROP_NAMES[5]}\n"
        f"Valid single-crop positions for mc9: {CROP_NAMES[9]}"
    )


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def set_speed_flags() -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def discover_shards(pattern: str) -> List[Path]:
    shards = sorted(Path().glob(pattern))
    if not shards:
        import glob

        shards = [Path(p) for p in sorted(glob.glob(pattern))]
    return shards


def read_mapping(path: str | Path) -> Dict[str, Tuple[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {p}")
    df = pd.read_csv(p)
    required = {"Name", "Patient", "Slide"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Mapping CSV must contain {required}; missing {missing}")
    out: Dict[str, Tuple[str, str]] = {}
    for name, patient, slide in zip(df["Name"].astype(str), df["Patient"], df["Slide"]):
        out[name] = (str(int(patient)), str(int(slide)))
    return out


def safe_patient(patient: Any) -> str:
    if patient is None or str(patient) in {"", "-1", "nan", "None", "unmapped"}:
        return "patient-unmapped"
    try:
        return f"patient-{int(patient):03d}"
    except Exception:
        return f"patient-{re.sub(r'[^A-Za-z0-9_.-]+', '-', str(patient))}"


def safe_slide(slide: Any) -> str:
    if slide is None or str(slide) in {"", "-1", "nan", "None", "unmapped"}:
        return "slide-unmapped"
    try:
        return f"slide-{int(slide):04d}"
    except Exception:
        return f"slide-{re.sub(r'[^A-Za-z0-9_.-]+', '-', str(slide))}"


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


def build_fallback_transform(input_size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_resize_norm_transform(input_size: int = 224, mean=None, std=None):
    from torchvision import transforms

    mean = tuple(mean) if mean is not None else (0.485, 0.456, 0.406)
    std = tuple(std) if std is not None else (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _crop_boxes(
    w: int, h: int, crop_size: int, n_crops: int
) -> List[Tuple[int, int, int, int]]:
    """Deterministic crop boxes. Order matches CROP_NAMES[n_crops].

    Emits a one-time RuntimeWarning per unique (w, h, crop_size, n_crops) combination
    when all multicrop positions collapse to the same box, which happens whenever
    crop_size >= min(w, h).  Pass --mc-crop-size smaller than the patch dimension to
    get diverse crops.
    """
    crop_size = int(min(max(1, crop_size), w, h))
    xs = [0, max(0, (w - crop_size) // 2), max(0, w - crop_size)]
    ys = [0, max(0, (h - crop_size) // 2), max(0, h - crop_size)]
    if n_crops <= 1:
        coords = [(xs[1], ys[1])]
    elif n_crops == 3:
        coords = [(xs[1], ys[1]), (xs[0], ys[0]), (xs[2], ys[2])]
    elif n_crops == 5:
        coords = [
            (xs[1], ys[1]),
            (xs[0], ys[0]),
            (xs[2], ys[0]),
            (xs[0], ys[2]),
            (xs[2], ys[2]),
        ]
    elif n_crops == 9:
        coords = [(x, y) for y in ys for x in xs]
    else:
        raise ValueError(f"Unsupported n_crops={n_crops}")
    boxes = [(x, y, x + crop_size, y + crop_size) for x, y in coords]

    # Warn once when multicrop diversity is lost.
    if n_crops > 1:
        key = (w, h, crop_size, n_crops)
        if len(set(boxes)) == 1 and key not in _warned_crop_collapse:
            _warned_crop_collapse.add(key)
            warnings.warn(
                f"[mc crop] All {n_crops} positions are identical for a {w}x{h} image "
                f"with crop_size={crop_size}.  Multicrop diversity is zero -- pass "
                f"--mc-crop-size smaller than min(patch_w, patch_h).",
                RuntimeWarning,
                stacklevel=2,
            )

    return boxes


def make_crops(img: Image.Image, n_crops: int, crop_size: int) -> List[Image.Image]:
    img = img.convert("RGB")
    w, h = img.size
    return [img.crop(box) for box in _crop_boxes(w, h, crop_size, n_crops)]


def load_foundation(foundation: str, device: torch.device, compile_model: bool = False):
    spec = FOUNDATION_SPECS[foundation]
    hf_id = spec["hf_id"]

    if foundation == "h0mini":
        from timm.layers import SwiGLUPacked

        model = timm.create_model(
            hf_id, pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
        )
    elif foundation == "hoptimus1":
        # Official Bioptimus H-optimus-1 inference kwargs. The model returns
        # a [B, 1536] feature tensor directly.
        model = timm.create_model(
            hf_id, pretrained=True, init_values=1e-5, dynamic_img_size=False
        )
    elif foundation == "uni2":
        from timm.layers import SwiGLUPacked

        model = timm.create_model(
            hf_id,
            pretrained=True,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
    elif foundation == "ctranspath":
        model = timm.create_model(hf_id, pretrained=True, num_classes=0)
    else:
        try:
            model = timm.create_model(hf_id, pretrained=True, num_classes=0)
        except TypeError:
            model = timm.create_model(hf_id, pretrained=True)

    model.eval().to(device)
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(f"[warn] torch.compile failed: {exc}", file=sys.stderr)

    try:
        data_cfg = resolve_model_data_config(model)
        transform = create_transform(**data_cfg, is_training=False)
        crop_transform = build_resize_norm_transform(
            input_size=int(spec["input_size"]),
            mean=spec.get("mean", data_cfg.get("mean")),
            std=spec.get("std", data_cfg.get("std")),
        )
    except Exception:
        transform = build_resize_norm_transform(
            spec["input_size"], mean=spec.get("mean"), std=spec.get("std")
        )
        crop_transform = build_resize_norm_transform(
            spec["input_size"], mean=spec.get("mean"), std=spec.get("std")
        )
    return model, transform, crop_transform


def _pool_tokens(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        for key in ("x_norm_clstoken", "pooled", "global_pool", "features"):
            v = out.get(key)
            if torch.is_tensor(v):
                return v
        for v in out.values():
            if torch.is_tensor(v):
                out = v
                break
    if isinstance(out, (tuple, list)):
        out = out[0]
    if torch.is_tensor(out) and out.ndim == 3:
        return out[:, 0]
    if torch.is_tensor(out) and out.ndim > 3:
        return out.flatten(1)
    return out


def extract_features(
    model, images: torch.Tensor, foundation: str, l2_normalize: bool = True
) -> torch.Tensor:
    if foundation == "virchow2":
        out = model.forward_features(images)
        if isinstance(out, dict):
            cls = out.get("x_norm_clstoken")
            patch = out.get("x_norm_patchtokens")
            emb = (
                torch.cat([cls, patch.mean(dim=1)], dim=1)
                if (cls is not None and patch is not None)
                else _pool_tokens(out)
            )
        elif torch.is_tensor(out) and out.ndim == 3:
            cls = out[:, 0]
            patch = out[:, 5:] if out.shape[1] > 260 else out[:, 1:]
            emb = torch.cat([cls, patch.mean(dim=1)], dim=1)
        else:
            emb = _pool_tokens(out)
    else:
        emb = _pool_tokens(model(images))

    if not torch.is_tensor(emb):
        raise TypeError(f"Model returned non-tensor: {type(emb)}")
    if emb.ndim != 2:
        emb = emb.flatten(1)
    if l2_normalize:
        emb = F.normalize(emb.float(), p=2, dim=1)
    return emb


class SampleConverter:
    """Pickle-safe WebDataset sample converter.

    single_crop_index=None  -> full multicrop tensor [n_crops, 3, H, W]  (or [3,H,W] for n_crops=1)
    single_crop_index=k     -> only crop k is extracted -> tensor [3, H, W]

    In resize-only variants n_crops=1 and transform is the explicit
    Resize((224,224)) + ToTensor + Normalize pipeline, so __call__ applies it to
    the full input image without any PIL crop.
    """

    def __init__(
        self,
        transform,
        n_crops: int = 1,
        crop_size: int = 224,
        single_crop_index: Optional[int] = None,
    ):
        self.transform = transform
        self.n_crops = int(n_crops)
        self.crop_size = int(crop_size)
        self.single_crop_index = single_crop_index

    def __call__(self, sample: Dict[str, Any]):
        key = sample.get("__key__", "")
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        key = str(key)

        img = None
        for k in ("jpg", "jpeg", "png", "webp"):
            if k in sample:
                img = sample[k]
                break
        if img is None:
            raise ValueError(f"sample {key}: no image field")
        if not isinstance(img, Image.Image):
            raise ValueError(f"sample {key}: not PIL.Image, got {type(img)}")

        if self.n_crops <= 1:
            # Base or resize-only path. In resize-only mode self.transform is
            # Resize((224,224)) + ToTensor + Normalize, applied to the full image.
            tensor = self.transform(img.convert("RGB"))
        elif self.single_crop_index is not None:
            # Extract exactly one crop from the mc grid -- gives [3, H, W].
            crops = make_crops(img, n_crops=self.n_crops, crop_size=self.crop_size)
            tensor = self.transform(crops[self.single_crop_index])
        else:
            # All crops -- gives [n_crops, 3, H, W].
            crops = make_crops(img, n_crops=self.n_crops, crop_size=self.crop_size)
            tensor = torch.stack([self.transform(c) for c in crops], dim=0)

        lab = -1
        for k in ("cls", "txt"):
            if k in sample:
                v = sample[k]
                if isinstance(v, bytes):
                    v = v.decode("utf-8")
                lab = int(str(v).strip())
                break
        return key, tensor, lab


def make_dataset(shards, transform, n_crops=1, crop_size=224, single_crop_index=None):
    return (
        wds.WebDataset([str(s) for s in shards], shardshuffle=False, empty_check=False)
        .decode("pil")
        .map(
            SampleConverter(
                transform,
                n_crops=n_crops,
                crop_size=crop_size,
                single_crop_index=single_crop_index,
            )
        )
    )


def chunk_shards(shards, unit_shards):
    unit_shards = max(1, int(unit_shards))
    return [shards[i : i + unit_shards] for i in range(0, len(shards), unit_shards)]


def unit_done(unit_dir):
    return unit_dir / "_UNIT_DONE.json"


def block_done(block_dir):
    return block_dir / "_DONE.json"


def completed_block_dirs(unit_dir: Path) -> List[Path]:
    if not unit_dir.exists():
        return []
    return [p for p in sorted(unit_dir.glob("block-*")) if block_done(p).exists()]


def read_completed_keys(unit_dir: Path) -> set:
    keys: set = set()
    for bdir in completed_block_dirs(unit_dir):
        try:
            meta = json.loads(block_done(bdir).read_text())
            keys.update(map(str, meta.get("keys", [])))
        except Exception:
            for part in bdir.glob("patient-*/slide-*.part.npz"):
                try:
                    d = np.load(part, allow_pickle=True)
                    keys.update(map(str, d["names"].tolist()))
                except Exception:
                    pass
    return keys


def next_block_index(unit_dir: Path) -> int:
    mx = -1
    for p in unit_dir.glob("block-*"):
        m = re.match(r"block-(\d+)$", p.name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def write_block(
    *,
    tmp_parent,
    final_block_dir,
    foundation,
    split,
    unit_name,
    block_index,
    buffers,
    block_keys,
    dtype,
    elapsed_s,
):
    tmp_dir = tmp_parent / f".{final_block_dir.name}.tmp.{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_patches = 0
    for (patient_dir, slide_stem), b in buffers.items():
        if not b["names"]:
            continue
        X = np.concatenate(b["X"], axis=0)
        y = np.concatenate(b["y"], axis=0)
        names = np.asarray(b["names"], dtype=object)
        out = tmp_dir / patient_dir / f"{slide_stem}.part.npz"
        atomic_npz(
            out,
            X=X,
            y=y,
            names=names,
            patient=np.asarray(patient_dir, dtype=object),
            slide=np.asarray(slide_stem, dtype=object),
        )
        n_files += 1
        n_patches += int(X.shape[0])

    meta = {
        "foundation": foundation,
        "split": split,
        "unit": unit_name,
        "block": f"block-{block_index:06d}",
        "n_patches": int(n_patches),
        "n_patient_slide_files": int(n_files),
        "keys": list(map(str, block_keys)),
        "dtype": dtype,
        "elapsed_s_since_unit_start": round(float(elapsed_s), 3),
    }
    atomic_json(tmp_dir / "_DONE.json", meta)
    if final_block_dir.exists():
        shutil.rmtree(final_block_dir)
    tmp_dir.replace(final_block_dir)
    return meta


def _check_feats_nan(feats_np: np.ndarray, context: str) -> np.ndarray:
    """Abort on systematic overflow (>10%); silently zero sparse bad rows."""
    finite = np.isfinite(feats_np.reshape(feats_np.shape[0], -1)).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad == 0:
        return feats_np
    frac = n_bad / max(len(feats_np), 1)
    if frac > 0.10:
        raise RuntimeError(
            f"ABORT [{context}]: {n_bad}/{len(feats_np)} ({100 * frac:.1f}%) embeddings are NaN/Inf.\n"
            f"Almost certainly float16 AMP overflow on MPS.\n"
            f"Re-run with --no-amp --batch-size 128"
        )
    tqdm.write(f"[warn] {n_bad} NaN/Inf patches ({100 * frac:.3f}%) -- zeroing.")
    out = feats_np.copy()
    out[~finite] = 0.0
    return out


def process_unit(
    *,
    unit_idx,
    unit_shards,
    split,
    foundation,
    base_foundation,
    n_crops,
    multicrop_combine,
    single_crop_index,
    crop_size,
    model,
    transform,
    mapping,
    parts_root,
    device,
    batch_size,
    num_workers,
    dtype,
    amp,
    l2_normalize,
    flush_every_batches,
):
    unit_name = f"unit-{unit_idx:06d}"
    unit_dir = parts_root / unit_name
    marker = unit_done(unit_dir)
    if marker.exists():
        meta = json.loads(marker.read_text())
        print(
            f"[skip] {foundation}/{split}/{unit_name}: complete ({meta.get('n_patches', '?')} patches)"
        )
        return meta

    unit_dir.mkdir(parents=True, exist_ok=True)
    for bdir in sorted(unit_dir.glob("block-*")):
        if not block_done(bdir).exists():
            shutil.rmtree(bdir, ignore_errors=True)

    done_keys = read_completed_keys(unit_dir)
    if done_keys:
        print(
            f"[resume] {foundation}/{split}/{unit_name}: skipping {len(done_keys):,} already-flushed patches"
        )

    n_workers = min(max(0, int(num_workers)), len(unit_shards))
    if len(unit_shards) == 1 and n_workers > 1:
        n_workers = 1

    dataset = make_dataset(
        unit_shards,
        transform,
        n_crops=n_crops,
        crop_size=crop_size,
        single_crop_index=single_crop_index,
    )
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        drop_last=False,
        persistent_workers=(n_workers > 0),
    )
    if n_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)

    t0 = time.time()
    amp_enabled = bool(amp and device.type in {"cuda", "mps"})
    amp_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.bfloat16
    flush_every = max(1, int(flush_every_batches))
    np_dtype = np.float16 if dtype == "float16" else np.float32
    context = f"{foundation}/{split}/{unit_name}"

    buffers: Dict[Tuple[str, str], Dict[str, list]] = defaultdict(
        lambda: {"X": [], "y": [], "names": []}
    )
    block_keys: List[str] = []
    block_idx = next_block_index(unit_dir)
    n_new = n_seen = n_skipped = block_batch_count = 0

    for keys, images, labels in tqdm(loader, desc=context, leave=False):
        keys_list = [str(k) for k in keys]
        n_seen += len(keys_list)
        keep_idx = (
            [i for i, k in enumerate(keys_list) if k not in done_keys]
            if done_keys
            else list(range(len(keys_list)))
        )
        if not keep_idx:
            n_skipped += len(keys_list)
            continue
        if len(keep_idx) < len(keys_list):
            n_skipped += len(keys_list) - len(keep_idx)
            kt = torch.as_tensor(keep_idx, dtype=torch.long)
            images = images.index_select(0, kt)
            labels = labels.index_select(0, kt)
            keys_list = [keys_list[i] for i in keep_idx]

        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                if images.ndim == 5:
                    # All-crops path: [B, nc, 3, H, W]
                    bsz, nc = int(images.shape[0]), int(images.shape[1])
                    flat = images.reshape(bsz * nc, *images.shape[2:])
                    ff = extract_features(
                        model, flat, base_foundation, l2_normalize=l2_normalize
                    )
                    fc = ff.reshape(bsz, nc, -1)  # [B, nc, dim]
                    if multicrop_combine == "percrop":
                        feats = fc  # [B, nc, dim]
                    elif multicrop_combine == "mean":
                        feats = (
                            F.normalize(fc.mean(dim=1).float(), p=2, dim=1)
                            if l2_normalize
                            else fc.mean(dim=1)
                        )
                    elif multicrop_combine == "concat":
                        feats = fc.reshape(bsz, nc * fc.shape[-1])
                        if l2_normalize:
                            feats = F.normalize(feats.float(), p=2, dim=1)
                    else:
                        raise ValueError(f"Unknown combine={multicrop_combine!r}")
                else:
                    # Single-crop path: [B, 3, H, W]  (includes single_crop_index variants)
                    feats = extract_features(
                        model, images, base_foundation, l2_normalize=l2_normalize
                    )

        feats_np = feats.detach().cpu().float().numpy()
        feats_np = _check_feats_nan(feats_np, context)
        feats_np = feats_np.astype(np_dtype, copy=False)
        labels_np = labels.detach().cpu().numpy().astype(np.int16, copy=False)

        if mapping is None:
            patients = slides = ["unmapped"] * len(keys_list)
        else:
            patients, slides = [], []
            for k in keys_list:
                ps = mapping.get(k)
                if ps is None:
                    patients.append("unmapped")
                    slides.append("unmapped")
                else:
                    patients.append(ps[0])
                    slides.append(ps[1])

        g2i: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for i, (p, s) in enumerate(zip(patients, slides)):
            g2i[(safe_patient(p), safe_slide(s))].append(i)
        for group, idxs in g2i.items():
            ia = np.asarray(idxs, dtype=np.int64)
            b = buffers[group]
            b["X"].append(feats_np[ia])
            b["y"].append(labels_np[ia])
            b["names"].extend([keys_list[i] for i in idxs])
        block_keys.extend(keys_list)
        n_new += len(keys_list)
        block_batch_count += 1

        if block_batch_count >= flush_every:
            fbd = unit_dir / f"block-{block_idx:06d}"
            meta = write_block(
                tmp_parent=unit_dir,
                final_block_dir=fbd,
                foundation=foundation,
                split=split,
                unit_name=unit_name,
                block_index=block_idx,
                buffers=buffers,
                block_keys=block_keys,
                dtype=dtype,
                elapsed_s=time.time() - t0,
            )
            done_keys.update(block_keys)
            tqdm.write(
                f"[block] {context}/{meta['block']}: {meta['n_patches']:,} patches"
            )
            buffers = defaultdict(lambda: {"X": [], "y": [], "names": []})
            block_keys = []
            block_batch_count = 0
            block_idx += 1
            gc.collect()

    if block_keys:
        fbd = unit_dir / f"block-{block_idx:06d}"
        meta = write_block(
            tmp_parent=unit_dir,
            final_block_dir=fbd,
            foundation=foundation,
            split=split,
            unit_name=unit_name,
            block_index=block_idx,
            buffers=buffers,
            block_keys=block_keys,
            dtype=dtype,
            elapsed_s=time.time() - t0,
        )
        tqdm.write(f"[block] {context}/{meta['block']}: {meta['n_patches']:,} patches")

    elapsed = time.time() - t0
    all_bmetas = [
        json.loads(block_done(b).read_text())
        for b in completed_block_dirs(unit_dir)
        if True
    ]
    n_total = int(sum(int(m.get("n_patches", 0)) for m in all_bmetas))

    meta = {
        "unit": unit_name,
        "split": split,
        "foundation": foundation,
        "shards": [str(s) for s in unit_shards],
        "n_shards": len(unit_shards),
        "n_patches": n_total,
        "n_seen_this_run": n_seen,
        "n_new_this_run": n_new,
        "n_skipped_this_run": n_skipped,
        "n_blocks": len(all_bmetas),
        "elapsed_s_this_run": round(elapsed, 3),
        "new_patches_per_s_this_run": round(n_new / max(elapsed, 1e-9), 3),
        "dtype": dtype,
        "base_foundation": base_foundation,
        "n_crops": n_crops,
        "multicrop_combine": multicrop_combine,
        "preprocess_mode": (
            "resize_only_full_image_to_model_input"
            if multicrop_combine == "resize_only"
            else multicrop_combine
        ),
        "source_image_transform": (
            "full PIL image -> Resize((input_size,input_size)) -> ToTensor -> Normalize; no crop"
            if multicrop_combine == "resize_only"
            else "see variant"
        ),
        "single_crop_index": single_crop_index,
        "crop_names": CROP_NAMES.get(n_crops, []),
        "crop_size": crop_size,
        "batch_size": batch_size,
        "num_workers": n_workers,
    }
    atomic_json(marker, meta)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


def collect_part_groups(parts_root: Path):
    groups: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for unit_dir in sorted(parts_root.glob("unit-*")):
        if not unit_done(unit_dir).exists():
            continue
        for block_dir in completed_block_dirs(unit_dir):
            for part in block_dir.glob("patient-*/slide-*.part.npz"):
                groups[(part.parent.name, part.stem.replace(".part", ""))].append(part)
    return groups


def consolidate(*, split, foundation, parts_root, final_root, restart_final=False):
    manifest_path = final_root / "manifest.json"
    if manifest_path.exists() and not restart_final:
        meta = json.loads(manifest_path.read_text())
        print(
            f"[skip] consolidated {foundation}/{split}: {meta.get('n_patches', '?')} patches"
        )
        return meta

    if restart_final and final_root.exists():
        shutil.rmtree(final_root)
    final_root.mkdir(parents=True, exist_ok=True)

    groups = collect_part_groups(parts_root)
    if not groups:
        raise RuntimeError(f"No completed part files under {parts_root}")

    n_patches = n_files = 0
    t0 = time.time()
    for (patient_dir, slide_stem), part_paths in tqdm(
        sorted(groups.items()), desc=f"consolidate {foundation}/{split}"
    ):
        out_path = final_root / patient_dir / f"{slide_stem}.npz"
        if out_path.exists() and not restart_final:
            try:
                n_patches += int(len(np.load(out_path, allow_pickle=True)["names"]))
                n_files += 1
                continue
            except Exception:
                out_path.unlink(missing_ok=True)

        Xs, ys, names = [], [], []
        for p in sorted(part_paths):
            d = np.load(p, allow_pickle=True)
            Xs.append(d["X"])
            ys.append(d["y"])
            names.extend(d["names"].tolist())
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        names_arr = np.asarray(names, dtype=object)

        order = np.argsort(names_arr.astype(str), kind="mergesort")
        X = X[order]
        y = y[order]
        names_arr = names_arr[order]
        if len(names_arr) > 1:
            _, ulr = np.unique(names_arr[::-1].astype(str), return_index=True)
            keep = np.sort(len(names_arr) - 1 - ulr)
            X = X[keep]
            y = y[keep]
            names_arr = names_arr[keep]

        atomic_npz(
            out_path,
            X=X,
            y=y,
            names=names_arr,
            patient=np.asarray(patient_dir, dtype=object),
            slide=np.asarray(slide_stem, dtype=object),
        )
        n_patches += int(X.shape[0])
        n_files += 1

    unit_metas = [
        json.loads(unit_done(u).read_text())
        for u in sorted(parts_root.glob("unit-*"))
        if unit_done(u).exists()
    ]
    um0 = unit_metas[0] if unit_metas else {}

    meta = {
        "split": split,
        "foundation": foundation,
        "n_patient_slide_files": n_files,
        "n_patches": n_patches,
        "parts_root": str(parts_root),
        "final_root": str(final_root),
        "n_completed_units": len(unit_metas),
        "n_crops": um0.get("n_crops", 1),
        "multicrop_combine": um0.get("multicrop_combine", "single"),
        "preprocess_mode": um0.get("preprocess_mode"),
        "source_image_transform": um0.get("source_image_transform"),
        "single_crop_index": um0.get("single_crop_index"),
        "crop_names": um0.get("crop_names", []),
        "elapsed_s_consolidate": round(time.time() - t0, 3),
    }
    atomic_json(manifest_path, meta)
    return meta


def extract_split_foundation(args, split, foundation, shards, mapping, device):
    variant = parse_foundation_variant(foundation)
    artifact_name = variant["artifact_name"]
    base_foundation = variant["base"]
    n_crops = variant["n_crops"]
    combine = variant["combine"]
    single_crop_index = variant["crop_index"]  # None unless combine=="single_crop"
    crop_name = variant["crop_name"]

    if not shards:
        print(f"[warn] no {split} shards; skipping {artifact_name}/{split}")
        return

    artifacts = Path(args.artifacts)
    parts_root = artifacts / "embedding_parts" / artifact_name / split
    final_root = artifacts / "embeddings_by_patient_slide" / artifact_name / split

    if args.restart:
        if parts_root.exists():
            shutil.rmtree(parts_root)
        if final_root.exists():
            shutil.rmtree(final_root)

    if (
        (final_root / "manifest.json").exists()
        and not args.restart
        and not args.force_reextract
    ):
        print(f"[skip] {foundation}/{split}: manifest exists")
        return

    unit_size = args.val_unit_shards if split == "val" else args.unit_shards
    units = chunk_shards(shards, unit_size)

    # Describe what we are actually doing.
    if combine == "resize_only":
        mode_note = (
            "full image resized directly to model input 224x224, no crop -> [n, dim]"
        )
    elif combine == "single_crop":
        mode_note = f"single crop: '{crop_name}' (index {single_crop_index} of mc{n_crops} grid) -> [n, dim]"
    elif combine == "percrop":
        mode_note = f"all {n_crops} crops stored separately -> [n, {n_crops}, dim]"
    else:
        mode_note = f"combine={combine}, n_crops={n_crops} -> [n, dim]"

    print(f"\n=== Extracting {artifact_name}/{split} ===")
    if foundation != artifact_name:
        print(f"requested: {foundation}  -> artifact: {artifact_name}")
    print(f"mode:      {mode_note}")
    print(f"shards:    {len(shards)},  units: {len(units)} x up to {unit_size}")
    print(
        f"amp:       {'disabled (float32 -- safe for MPS multicrop)' if args.no_amp else 'enabled'}"
    )

    model, single_transform, crop_transform = load_foundation(
        base_foundation, device, compile_model=args.compile
    )
    # Use the explicit Resize((224,224)) transform when:
    #   1. resize-only mode keeps the entire source patch and resizes it directly;
    #   2. multicrop mode manually slices PIL crops before model normalization.
    # For the original base mode, preserve timm's resolved eval transform.
    if combine == "resize_only":
        transform = crop_transform
    elif n_crops > 1:
        transform = crop_transform
    else:
        transform = single_transform

    for unit_idx, unit in enumerate(units):
        meta = process_unit(
            unit_idx=unit_idx,
            unit_shards=unit,
            split=split,
            foundation=artifact_name,
            base_foundation=base_foundation,
            n_crops=n_crops,
            multicrop_combine=combine,
            single_crop_index=single_crop_index,
            crop_size=args.mc_crop_size,
            model=model,
            transform=transform,
            mapping=mapping,
            parts_root=parts_root,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dtype=args.dtype,
            amp=not args.no_amp,
            l2_normalize=not args.no_l2,
            flush_every_batches=args.flush_every_batches,
        )
        print(
            f"[done] {artifact_name}/{split}/{meta['unit']}: "
            f"{meta['n_patches']:,} total, {meta.get('n_new_this_run', 0):,} new, "
            f"{meta.get('new_patches_per_s_this_run', 0):.1f} patches/s"
        )

    if not args.no_consolidate:
        meta = consolidate(
            split=split,
            foundation=artifact_name,
            parts_root=parts_root,
            final_root=final_root,
            restart_final=args.restart_final,
        )
        comb = meta.get("multicrop_combine", "single")
        sc = meta.get("single_crop_index")
        if comb == "single_crop" and sc is not None:
            shape_note = f"[n, dim]  (crop '{CROP_NAMES.get(meta.get('n_crops', 1), ['?'])[sc]}')"
        elif comb == "percrop":
            shape_note = f"[n, {meta.get('n_crops', 1)}, dim]"
        elif comb == "resize_only":
            shape_note = "[n, dim]  (full-image resize-only, no crop)"
        else:
            shape_note = "[n, dim]"
        print(
            f"[consolidated] {artifact_name}/{split}: "
            f"{meta['n_patches']:,} patches in {meta['n_patient_slide_files']} files  X: {shape_note}"
        )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train-glob", default=DEFAULT_TRAIN_GLOB)
    parser.add_argument("--val-glob", default=DEFAULT_VAL_GLOB)
    parser.add_argument("--mapping-csv", default=DEFAULT_MAPPING_CSV)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    parser.add_argument(
        "--foundations",
        default="h0mini",
        help=(
            "Comma-separated variants. Examples:\n"
            "  h0mini                  full patch resized to 224x224, no crop -> [n,dim]; artifact h0mini\n"
            "  hoptimus1               H-optimus-1 full patch resized to 224x224, no crop -> [n,1536]; artifact hoptimus1\n"
            "  hoptimus1_mc5_center    H-optimus-1 centre crop from mc5 grid -> [n,1536]\n"
            "  uni2                    full patch resized to 224x224, no crop -> [n,dim]; artifact uni2\n"
            "  h0mini_resize224        accepted alias, but still writes under artifact h0mini\n"
            "  h0mini_mc5_center       only centre crop from mc5 grid -> [n,dim]\n"
            "  h0mini_mc5_top_right    only top-right crop            -> [n,dim]\n"
            "  h0mini_mc5              all 5 crops stored together    -> [n,5,dim]\n"
            "  h0mini_mc5_mean         mean of 5 crops (legacy)       -> [n,dim]\n"
            f"  Valid single-crop positions (mc5): {CROP_NAMES[5]}\n"
            f"  Valid single-crop positions (mc9): {CROP_NAMES[9]}"
        ),
    )
    parser.add_argument("--splits", default="train,val")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="For single-crop variants on MPS, 128 is safe. "
        "For all-crops (mc5) use --no-amp.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--unit-shards", type=int, default=1)
    parser.add_argument("--val-unit-shards", type=int, default=1)
    parser.add_argument(
        "--flush-every-batches",
        type=int,
        default=50,
        help="Flush a checkpoint block every N batches. "
        "Lower = more crash-safe but more small files on disk. "
        "Default 50 gives ~6 400 patches/block at batch-size 128.",
    )
    parser.add_argument(
        "--mc-crop-size",
        type=int,
        default=224,
        help="Side length (px) of each crop cut from the patch before "
        "resizing to the model input size. Must be < min(patch_w, patch_h) "
        "for multicrop variants to have any diversity (BraTS-Path patches "
        "are 512x512, so the default 224 is fine).",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable autocast. Recommended for any multicrop variant on MPS.",
    )
    parser.add_argument("--no-l2", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--restart-final", action="store_true")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--no-consolidate", action="store_true")
    args = parser.parse_args()

    set_speed_flags()
    device = auto_device() if args.device == "auto" else torch.device(args.device)
    print(
        f"device: {device}  python: {platform.python_version()}  platform: {platform.platform()}"
    )

    foundations = parse_csv_list(args.foundations)
    splits = parse_csv_list(args.splits)
    for f in foundations:
        v = parse_foundation_variant(f)  # validate early, print what will happen
        print(
            f"  {f!r:40s}  artifact={v['artifact_name']}  base={v['base']}  combine={v['combine']}"
            + (
                f"  crop_index={v['crop_index']} ({v['crop_name']})"
                if v["crop_index"] is not None
                else ""
            )
        )

    train_shards = discover_shards(args.train_glob)
    val_shards = discover_shards(args.val_glob)
    print(f"train shards: {len(train_shards)},  val shards: {len(val_shards)}")

    # Load mapping once -- only needed for train.
    mapping = None
    if "train" in splits:
        mapping = read_mapping(args.mapping_csv)
        print(f"mapping rows: {len(mapping):,}")

    # Val is extracted first so probe sweeps can start as soon as any train
    # patient-slide files are consolidated, without waiting for the full train run.
    for foundation in foundations:
        if "val" in splits:
            extract_split_foundation(args, "val", foundation, val_shards, None, device)
        if "train" in splits:
            extract_split_foundation(
                args, "train", foundation, train_shards, mapping, device
            )


if __name__ == "__main__":
    main()

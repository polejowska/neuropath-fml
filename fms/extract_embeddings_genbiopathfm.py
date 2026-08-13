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
        # Output token layout: [B, 261, 1280]
        #   token 0     : CLS
        #   tokens 1-4  : register tokens (skip)
        #   tokens 5-260: patch tokens (256 total)
        # Embedding = cat([cls, mean(patch_tokens)], dim=-1) -> [B, 2560]
    },
    # ---------------------------------------------------------------------------
    # ProvGigaPath tile encoder
    # ---------------------------------------------------------------------------
    # ViT-g/14 pretrained with DINOv2 on 1.3B 256x256 pathology tiles.
    # Gated model: request access at https://huggingface.co/prov-gigapath/prov-gigapath
    # then set  export HF_TOKEN=<your_token>  before running.
    #
    # Official preprocessing (model card + README):
    #   Resize(256, BICUBIC) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)
    # This differs from other models: it uses "standard_eval" mode, not "resize_only".
    # For BraTS-Path 512x512 patches: 512->256 resize, then 256->224 center crop.
    #
    # Forward pass: returns [B, 1536] class-token embedding directly.
    # No special MLP layers needed; plain timm.create_model loads cleanly.
    #
    # Reference: https://huggingface.co/prov-gigapath/prov-gigapath
    #            https://github.com/prov-gigapath/prov-gigapath
    "provgigapath": {
        "hf_id": "hf-hub:prov-gigapath/prov-gigapath",
        "input_size": 224,
        "vit_patch_size": 14,
        "embedding_dim": 1536,
    },
    # ---------------------------------------------------------------------------
    # Phikon-v2 tile encoder
    # ---------------------------------------------------------------------------
    # ViT-Large/16 trained with DINOv2 SSL on PANCAN-XL (450M tiles, 60K WSIs).
    # Public model, NO HuggingFace token required.
    #
    # Loaded via transformers.AutoModel (NOT timm) -- the only model in this
    # script that uses the HuggingFace transformers library.
    # Forward call: model(pixel_values=images).last_hidden_state[:, 0, :]
    #   → CLS token [B, 1024]
    #
    # Preprocessing matches the official AutoImageProcessor behaviour:
    #   Resize((224, 224), BICUBIC) -> ToTensor -> Normalize(ImageNet)
    # This is identical to the "resize_only" pipeline already used for h0mini,
    # uni2 and virchow2 (combine="resize_only").
    #
    # Reference: https://huggingface.co/owkin/phikon-v2
    #            https://arxiv.org/abs/2409.09173
    "phikonv2": {
        "hf_id": "owkin/phikon-v2",  # loaded via transformers.AutoModel, NOT timm
        "input_size": 224,
        "vit_patch_size": 16,
        "embedding_dim": 1024,
    },
    # ---------------------------------------------------------------------------
    # GenBio-PathFM tile encoder
    # ---------------------------------------------------------------------------
    # 1.1B-parameter histopathology foundation model from GenBio AI.
    # Loaded via transformers.AutoModel.from_pretrained(..., trust_remote_code=True)
    # -- NOT timm.  The remote code model returns the tile-level CLS embedding
    # directly from model(x): [B, 4608].  Patch tokens are available through
    # model.forward_with_patches(x), but this extractor stores only CLS features.
    #
    # Official preprocessing (GitHub README + HuggingFace model card):
    #   Resize((224, 224)) -> ToTensor -> Normalize(
    #       mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187)
    #   )
    #
    # Reference: https://github.com/genbio-ai/genbio-pathfm
    #            https://huggingface.co/genbio-ai/genbio-pathfm
    "genbiopathfm": {
        "hf_id": "genbio-ai/genbio-pathfm",  # loaded via transformers.AutoModel, NOT timm
        "input_size": 224,
        "vit_patch_size": 16,
        "embedding_dim": 4608,
    },
    # ---------------------------------------------------------------------------
    # MUSK tile encoder
    # ---------------------------------------------------------------------------
    # MUSK: Multimodal transformer with Unified maSK modeling (Nature 2025).
    # Architecture: ViT-Large/16 @ 384x384, embedding_dim=1024.
    # Gated model: request access at https://huggingface.co/xiangjx/musk
    # then set  export HF_TOKEN=<your_token>  before running.
    #
    # Requires the musk package (registers "musk_large_patch16_384" with timm):
    #   pip install git+https://github.com/lilab-stanford/MUSK.git
    # Loading uses musk.utils.load_model_and_may_interpolate, NOT timm pretrained=True.
    #
    # Official preprocessing (GitHub README + demo.ipynb):
    #   Resize(384, BICUBIC, antialias=True) -> CenterCrop(384) -> ToTensor
    #   -> Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
    #   where IMAGENET_INCEPTION_MEAN = IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
    # Note: BEiT/DALL-E style normalization, NOT standard ImageNet.
    #
    # Forward call (feature extraction for MIL/classification tasks):
    #   out = model(image=img, with_head=False, out_norm=False,
    #               ms_aug=True, return_global=True)
    #   embedding = out[0]  # (vision_cls, text_cls) tuple -> vision_cls [B, 1024]
    #   ms_aug=True: multiscale augmentation inside the model forward pass.
    #                Recommended for MIL/classification (per official README).
    #                Increase GPU memory usage; reduce --batch-size if OOM.
    #   return_global=True: return CLS token only; excludes patch tokens.
    #   out_norm=False: this script handles L2 normalization (use --no-l2 to skip).
    #
    # Reference: https://huggingface.co/xiangjx/musk
    #            https://github.com/lilab-stanford/MUSK
    "musk": {
        "hf_id": "hf-hub:xiangjx/musk",  # used by musk.utils.load_model_and_may_interpolate
        "input_size": 384,
        "vit_patch_size": 16,
        "embedding_dim": 1024,
    },
}

# Models that use direct Resize((input_size, input_size)) preprocessing.
# combine = "resize_only"
_RESIZE_ONLY_FOUNDATIONS: frozenset = frozenset(
    {"h0mini", "ctranspath", "uni2", "virchow2", "phikonv2", "genbiopathfm"}
)

# Models that use the official timm eval transform (Resize then CenterCrop).
# combine = "standard_eval"
# The exact transform is derived from timm's resolve_model_data_config+create_transform,
# with build_standard_eval_transform as a hardcoded fallback.
_STANDARD_EVAL_FOUNDATIONS: frozenset = frozenset({"provgigapath"})

# Models that use their own custom preprocessing distinct from both resize_only and
# standard_eval.  combine = "musk_eval".
# MUSK: Resize(384,BICUBIC,antialias=True)+CenterCrop(384)+Normalize((0.5,0.5,0.5)).
_MUSK_EVAL_FOUNDATIONS: frozenset = frozenset({"musk"})

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
_warned_crop_collapse: set = set()


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


_FOUNDATION_ALIASES: Dict[str, str] = {
    "genbio-pathfm": "genbiopathfm",
    "genbio_pathfm": "genbiopathfm",
}


def canonicalize_foundation_variant_name(name: str) -> str:
    """Map user-friendly aliases to the internal artifact/base name.

    Supported examples:
      genbio-pathfm        -> genbiopathfm
      genbio_pathfm        -> genbiopathfm
      genbio-pathfm_mc5    -> genbiopathfm_mc5
    """
    name = str(name).strip()
    for alias, canonical in _FOUNDATION_ALIASES.items():
        if name == alias:
            return canonical
        if name.startswith(alias + "_"):
            return canonical + name[len(alias) :]
    return name


def parse_foundation_variant(name: str) -> Dict[str, Any]:
    """Parse a foundation/variant name into extraction parameters.

    Returns a dict with:
      artifact_name   : output directory name
      base            : base model key in FOUNDATION_SPECS
      n_crops         : total crop positions in the grid
      combine         : "resize_only" | "standard_eval" | "musk_eval" | "single_crop" | "percrop" | "mean" | "concat"
      crop_index      : int | None  (only for combine=="single_crop")
      crop_name       : str | None  (human-readable position)

    combine values
    --------------
    resize_only    : full image resized directly to model input_size (no crop)
                     uses build_resize_norm_transform (h0mini, ctranspath, uni2, virchow2)
    standard_eval  : official timm eval transform for the model, e.g. Resize(256)+CenterCrop(224)
                     uses single_transform from create_transform (provgigapath)
    musk_eval      : official MUSK preprocessing: Resize(384,BICUBIC,antialias)+CenterCrop(384)
                     +Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD)
                     uses build_musk_transform; ms_aug=True applied inside model forward pass
    single_crop    : single position from an mc grid
    percrop        : all positions stored separately [n, n_crops, dim]
    mean / concat  : legacy aggregated multicrop
    """
    name = canonicalize_foundation_variant_name(name)

    # -------------------------------------------------------------------------
    # Base model: e.g. "h0mini", "provgigapath", "musk", "genbiopathfm" -- no variant suffix.
    # -------------------------------------------------------------------------
    if name in FOUNDATION_SPECS:
        if name in _STANDARD_EVAL_FOUNDATIONS:
            combine = "standard_eval"
        elif name in _MUSK_EVAL_FOUNDATIONS:
            combine = "musk_eval"
        else:
            combine = "resize_only"
        return {
            "artifact_name": name,
            "base": name,
            "n_crops": 1,
            "combine": combine,
            "crop_index": None,
            "crop_name": None,
        }

    # -------------------------------------------------------------------------
    # Pattern helpers
    # -------------------------------------------------------------------------
    base_pat = r"(?P<base>h0mini|ctranspath|uni2|virchow2|provgigapath|musk|phikonv2|genbiopathfm)"

    # Full-patch resize-only variant: h0mini_resize224, provgigapath_resize224, etc.
    # Note: provgigapath_resize224 explicitly requests resize-only (deviates from
    # the official preprocessing), so combine="resize_only" is intentional here.
    # musk_resize384: likewise deviates from official MUSK preprocessing (no antialias,
    # no BEiT normalization); use base "musk" for official preprocessing.
    m = re.match(rf"^{base_pat}_resize(?P<size>\d+)$", name)
    if m:
        base = m.group("base")
        resize_size = int(m.group("size"))
        expected = int(FOUNDATION_SPECS[base]["input_size"])
        if resize_size != expected:
            raise ValueError(
                f"Resize-only variant {name!r} requested {resize_size}, but {base} "
                f"expects input_size={expected}. Use {base}_resize{expected}."
            )
        return {
            "artifact_name": base,
            "base": base,
            "n_crops": 1,
            "combine": "resize_only",
            "crop_index": None,
            "crop_name": None,
        }

    # Single specific crop: h0mini_mc5_center, provgigapath_mc5_top_right,
    #                       musk_mc5_top_right, etc.
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

    # All crops together (per-crop storage): h0mini_mc5, provgigapath_mc9,
    #                                        musk_mc5, etc.
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
        f"  h0mini_resize224          alias for h0mini in this script; artifact h0mini\n"
        f"  h0mini_mc5_center         centre crop only\n"
        f"  h0mini_mc5_top_right      top-right crop only\n"
        f"  h0mini_mc5                all 5 crops stored as [n,5,dim]\n"
        f"  h0mini_mc5_mean           mean of 5 crops (legacy)\n"
        f"  provgigapath              Resize(256)+CenterCrop(224); artifact provgigapath\n"
        f"  provgigapath_resize224    resize-only alias (deviates from official preprocessing)\n"
        f"  provgigapath_mc5_center   centre crop only\n"
        f"  phikonv2                  Resize(224)+ToTensor+Normalize(ImageNet); artifact phikonv2\n"
        f"                              public model, no HF_TOKEN needed\n"
        f"  phikonv2_mc5_center       mc5 centre crop (phikonv2 preprocessing)\n"
        f"  phikonv2_mc5              all 5 crops stored as [n,5,dim]\n"
        f"  genbiopathfm              Resize(224)+GenBio PathFM Normalize -> [n,4608]\n"
        f"                              aliases: genbio-pathfm, genbio_pathfm\n"
        f"  genbiopathfm_mc5_center   mc5 centre crop (GenBio PathFM preprocessing)\n"
        f"  genbiopathfm_mc5          all 5 crops stored as [n,5,4608]\n"
        f"  musk                      Resize(384)+CenterCrop(384); artifact musk\n"
        f"                              requires: pip install git+https://github.com/lilab-stanford/MUSK.git\n"
        f"                              gated HF model: set HF_TOKEN before running\n"
        f"  musk_mc5_center           mc5 centre crop (MUSK preprocessing per crop)\n"
        f"  musk_mc5                  all 5 crops stored as [n,5,dim]\n"
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


GENBIOPATHFM_MEAN: Tuple[float, float, float] = (0.697, 0.575, 0.728)
GENBIOPATHFM_STD: Tuple[float, float, float] = (0.188, 0.240, 0.187)


def build_genbiopathfm_transform(input_size: int = 224):
    """Official GenBio-PathFM preprocessing.

    GitHub README / HuggingFace model card:
      transforms.Resize((224, 224))
      transforms.ToTensor()
      transforms.Normalize(mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187))

    `transforms.Resize((224, 224))` uses torchvision's default bilinear interpolation;
    do not route this through build_resize_norm_transform(), which uses BICUBIC.
    """
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                (input_size, input_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=GENBIOPATHFM_MEAN, std=GENBIOPATHFM_STD),
        ]
    )


def build_standard_eval_transform(
    input_size: int = 224, resize_to: int = 256, mean=None, std=None
):
    """Resize(resize_to, BICUBIC) -> CenterCrop(input_size) -> ToTensor -> Normalize.

    Used as a hardcoded fallback for ProvGigaPath when timm's create_transform is
    unavailable.  Matches the official model-card preprocessing exactly:
      transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC)
      transforms.CenterCrop(224)
      transforms.ToTensor()
      transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
    Reference: https://huggingface.co/prov-gigapath/prov-gigapath
    """
    from torchvision import transforms

    mean = tuple(mean) if mean is not None else (0.485, 0.456, 0.406)
    std = tuple(std) if std is not None else (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Resize(
                resize_to, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_musk_transform(input_size: int = 384):
    """Official MUSK preprocessing transform.

    Matches the official MUSK preprocessing exactly as documented in the GitHub README
    and demo.ipynb:
      torchvision.transforms.Resize(384, interpolation=3, antialias=True)
        [interpolation=3 is BICUBIC]
      torchvision.transforms.CenterCrop((384, 384))
      torchvision.transforms.ToTensor()
      torchvision.transforms.Normalize(
          mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
      )
    where IMAGENET_INCEPTION_MEAN = IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
    (BEiT/DALL-E style normalization, NOT standard ImageNet mean/std).

    For BraTS-Path 512x512 patches:
      Resize(384, BICUBIC): both dims 512->384 (square patch, both edges equal,
        so both scale to 384).
      CenterCrop(384): effectively a no-op for the resulting 384x384 image.
    The pre-crop is applied per-tile by our pipeline (n_crops > 1 variants), with
    each crop region resized to 384x384 before normalization.

    Source: https://github.com/lilab-stanford/MUSK (README, demo.ipynb)
            https://huggingface.co/xiangjx/musk
    """
    from torchvision import transforms

    try:
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

        _mean = tuple(IMAGENET_INCEPTION_MEAN)
        _std = tuple(IMAGENET_INCEPTION_STD)
    except ImportError:
        # Hardcoded fallback: (0.5, 0.5, 0.5) / (0.5, 0.5, 0.5)
        _mean = (0.5, 0.5, 0.5)
        _std = (0.5, 0.5, 0.5)
    return transforms.Compose(
        [
            transforms.Resize(
                input_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std),
        ]
    )


def _crop_boxes(
    w: int, h: int, crop_size: int, n_crops: int
) -> List[Tuple[int, int, int, int]]:
    """Deterministic crop boxes. Order matches CROP_NAMES[n_crops]."""
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
    """Load a pathology foundation model with the correct architecture options.

    Each model requires specific timm constructor arguments to match its checkpoint.
    Loading without the correct mlp_layer/act_layer causes weight shape mismatches.

    ProvGigaPath requires no special constructor kwargs; plain timm.create_model works.
    Its official preprocessing (Resize(256)+CenterCrop(224)) is derived from the timm
    eval transform (create_transform), with build_standard_eval_transform as fallback.
    Reference: https://huggingface.co/prov-gigapath/prov-gigapath

    GenBio-PathFM uses transformers.AutoModel with trust_remote_code=True and a
    model-specific 224x224 resize + GenBio PathFM mean/std transform.
    Reference: https://github.com/genbio-ai/genbio-pathfm
               https://huggingface.co/genbio-ai/genbio-pathfm

    MUSK requires the external `musk` package (from https://github.com/lilab-stanford/MUSK)
    which registers "musk_large_patch16_384" with timm.  The model weights are loaded
    via musk.utils.load_model_and_may_interpolate, NOT via timm's pretrained=True.
    Both single_transform and crop_transform are set to build_musk_transform(384).
    Reference: https://huggingface.co/xiangjx/musk
               https://github.com/lilab-stanford/MUSK
    """
    spec = FOUNDATION_SPECS[foundation]
    hf_id = spec["hf_id"]

    if foundation == "h0mini":
        from timm.layers import SwiGLUPacked

        model = timm.create_model(
            hf_id,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
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

    elif foundation == "virchow2":
        # Virchow2 uses SwiGLU MLP.  Without mlp_layer=SwiGLUPacked the fc2
        # weights are twice as wide as the checkpoint and load fails.
        # Reference: https://huggingface.co/paige-ai/Virchow2
        from timm.layers import SwiGLUPacked

        model = timm.create_model(
            hf_id,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    elif foundation == "provgigapath":
        # ProvGigaPath tile encoder: ViT-g/14 pretrained with DINOv2.
        # No special MLP layer kwargs needed; the checkpoint uses a standard GELU MLP.
        # Returns [B, 1536] class-token embedding directly (num_classes=0 in hub config).
        # IMPORTANT: this is a gated model -- set HF_TOKEN before running.
        # Reference: https://huggingface.co/prov-gigapath/prov-gigapath
        model = timm.create_model(hf_id, pretrained=True)

    elif foundation == "phikonv2":
        # Phikon-v2: ViT-Large/16 trained with DINOv2 on PANCAN-XL (450M tiles, 60K WSIs).
        # Loaded via transformers.AutoModel -- NOT timm.create_model.
        # Public model; no HF_TOKEN required.
        #
        # The AutoImageProcessor for this model applies:
        #   Resize((224, 224), BICUBIC) -> ToTensor -> Normalize(ImageNet mean/std)
        # We replicate this exactly with build_resize_norm_transform(224) below,
        # which is also used by the "resize_only" path for h0mini / uni2 / virchow2.
        #
        # Reference: https://huggingface.co/owkin/phikon-v2
        try:
            from transformers import AutoModel as _HFAutoModel
        except ImportError as exc:
            raise RuntimeError(
                "Phikon-v2 requires the transformers library.\n"
                "Install with:  pip install transformers"
            ) from exc
        model = _HFAutoModel.from_pretrained(hf_id)
        model.eval().to(device)

        if compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:
                print(
                    f"[warn] torch.compile failed for phikonv2: {exc}", file=sys.stderr
                )

        # Both single_transform and crop_transform: Resize(224)+ToTensor+Normalize(ImageNet)
        # This matches the official AutoImageProcessor behaviour exactly.
        _t = build_resize_norm_transform(int(spec["input_size"]))
        return model, _t, _t

    elif foundation == "genbiopathfm":
        # GenBio-PathFM: 1.1B-parameter histopathology FM.
        # Loaded through HuggingFace transformers with trust_remote_code=True.
        # The remote-code model's forward(x) returns the CLS embedding directly:
        # [B, 4608].  Patch tokens are available via forward_with_patches(x), but
        # this extractor stores only tile-level CLS features.
        # Reference: https://github.com/genbio-ai/genbio-pathfm
        #            https://huggingface.co/genbio-ai/genbio-pathfm
        try:
            from transformers import AutoModel as _HFAutoModel
        except ImportError as exc:
            raise RuntimeError(
                "GenBio-PathFM requires the transformers library.\n"
                "Install/update with:  pip install -U transformers\n"
                "GenBio AI tested inference with transformers==4.57.1."
            ) from exc
        model = _HFAutoModel.from_pretrained(hf_id, trust_remote_code=True)
        model.eval().to(device)

        if compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:
                print(
                    f"[warn] torch.compile failed for genbiopathfm: {exc}",
                    file=sys.stderr,
                )

        # Both single_transform and crop_transform use the official GenBio-PathFM
        # preprocessing: Resize((224,224), default BILINEAR)+ToTensor+Normalize(GenBio mean/std).
        _t = build_genbiopathfm_transform(int(spec["input_size"]))
        return model, _t, _t

    elif foundation == "musk":
        # MUSK: Multimodal transformer with Unified maSK modeling (Nature 2025).
        # ViT-Large/16 @ 384x384; embedding_dim=1024.
        #
        # Loading steps (both are required):
        #   1. Import musk.modeling to register "musk_large_patch16_384" with timm.
        #   2. Create the model skeleton via timm.create_model (no pretrained flag).
        #   3. Load weights from HuggingFace via musk.utils.load_model_and_may_interpolate.
        #
        # The model is a gated HF model -- set HF_TOKEN before running.
        # The musk package must be installed:
        #   pip install git+https://github.com/lilab-stanford/MUSK.git
        #
        # Reference: https://huggingface.co/xiangjx/musk
        #            https://github.com/lilab-stanford/MUSK
        try:
            import musk.modeling as _musk_modeling_register  # noqa: F401
            from musk import utils as _musk_utils

            # importing musk.modeling registers "musk_large_patch16_384" with timm
        except ImportError as exc:
            raise RuntimeError(
                "MUSK requires the musk package. Install with:\n"
                "  pip install git+https://github.com/lilab-stanford/MUSK.git\n"
                "Or clone and install locally:\n"
                "  git clone https://github.com/lilab-stanford/MUSK && "
                "pip install -e ./MUSK\n"
                "Reference: https://github.com/lilab-stanford/MUSK"
            ) from exc
        # Create the model skeleton (pretrained=False; weights loaded separately below)
        model = timm.create_model("musk_large_patch16_384")
        # Load pretrained weights from HuggingFace Hub via the official MUSK loader.
        # 'model|module' is the state-dict key prefix filter (standard MUSK checkpoint format).
        # '' is the target prefix (no prefix stripping in the destination model).
        _musk_utils.load_model_and_may_interpolate(
            "hf_hub:xiangjx/musk", model, "model|module", ""
        )
        model.eval().to(device)

        # Build MUSK-specific transforms (both single and crop variants use the same
        # Resize(384)+CenterCrop(384)+Normalize((0.5,0.5,0.5)) pipeline).
        musk_transform = build_musk_transform(int(spec["input_size"]))

        if compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
            except Exception as exc:
                print(f"[warn] torch.compile failed for musk: {exc}", file=sys.stderr)

        # Return same transform for both single and crop paths.
        return model, musk_transform, musk_transform

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
        # single_transform: official timm eval transform for this model.
        # For provgigapath: Resize(256, BICUBIC) -> CenterCrop(224) -> ToTensor -> Normalize
        # For h0mini/uni2/virchow2: also Resize+CenterCrop by default from timm, but
        # the caller overrides with crop_transform (Resize((224,224))) for resize_only mode.
        single_transform = create_transform(**data_cfg, is_training=False)
        crop_transform = build_resize_norm_transform(
            input_size=int(spec["input_size"]),
            mean=data_cfg.get("mean"),
            std=data_cfg.get("std"),
        )
    except Exception:
        if foundation == "provgigapath":
            # Hardcoded fallback matching the official model card exactly.
            # Reference: https://huggingface.co/prov-gigapath/prov-gigapath
            single_transform = build_standard_eval_transform(
                input_size=int(spec["input_size"]),
                resize_to=256,
            )
        else:
            single_transform = build_fallback_transform(spec["input_size"])
        crop_transform = build_resize_norm_transform(spec["input_size"])

    return model, single_transform, crop_transform


def _pool_tokens(out: Any) -> torch.Tensor:
    """Generic token pooling for models that return dicts or 3-D tensors."""
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
    """Extract embeddings for a batch of images.

    Virchow2 uses the official token-indexing recipe from the model card:
      output shape  [B, 261, 1280]  (CLS + 4 register tokens + 256 patch tokens)
      class_token   output[:, 0]         [B, 1280]
      patch_tokens  output[:, 5:]        [B, 256, 1280]  (skip register tokens 1-4)
      embedding     cat([cls, mean(patch)], dim=-1)  [B, 2560]
    Reference: https://huggingface.co/paige-ai/Virchow2

    ProvGigaPath uses standard class-token pooling:
      output shape  [B, 1536]  (class token, returned directly by the model)
    No token indexing required; _pool_tokens handles this transparently.
    Reference: https://huggingface.co/prov-gigapath/prov-gigapath

    GenBio-PathFM returns the CLS embedding directly from model(images):
      output shape [B, 4608]
    Reference: https://github.com/genbio-ai/genbio-pathfm
               https://huggingface.co/genbio-ai/genbio-pathfm

    MUSK uses the official forward call from the GitHub README and demo.ipynb:
      model(image=images, with_head=False, out_norm=False, ms_aug=True, return_global=True)
      returns a tuple (vision_cls, text_cls); we take index [0] -> [B, 1024]
      with_head=False  : no projection head (raw encoder output)
      out_norm=False   : normalization handled by this script (see l2_normalize arg)
      ms_aug=True      : multiscale augmentation inside the model forward pass;
                         recommended for MIL/classification tasks (per official README).
                         Note: increases effective GPU memory use; reduce --batch-size if OOM.
      return_global=True: return CLS token only; excludes patch tokens.
    Reference: https://huggingface.co/xiangjx/musk
               https://github.com/lilab-stanford/MUSK
    """
    if foundation == "virchow2":
        out = model(images)  # [B, 261, 1280]
        class_token = out[:, 0]  # [B, 1280]
        patch_tokens = out[:, 5:]  # [B, 256, 1280]  (skip 4 register tokens)
        emb = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=1)  # [B, 2560]

    elif foundation == "phikonv2":
        # Phikon-v2 uses transformers.AutoModel (ViTModel).
        # Official feature extraction from the model card:
        #   outputs = model(**inputs)
        #   features = outputs.last_hidden_state[:, 0, :]  # CLS token -> [B, 1024]
        # We pass pixel_values directly (our pipeline already applied the transform).
        # Reference: https://huggingface.co/owkin/phikon-v2
        outputs = model(pixel_values=images)
        emb = outputs.last_hidden_state[:, 0, :]  # CLS token [B, 1024]

    elif foundation == "genbiopathfm":
        # Official GenBio-PathFM forward call for tile-level CLS features.
        # model(images) -> [B, 4608].
        emb = model(images)

    elif foundation == "musk":
        # Official MUSK forward call for feature extraction (MIL/classification tasks).
        # ms_aug=True applies multiscale augmentation internally.
        # Returns (vision_cls, text_cls); we take vision_cls.
        # Reference: https://github.com/lilab-stanford/MUSK
        out = model(
            image=images,
            with_head=False,
            out_norm=False,  # normalization handled below via l2_normalize
            ms_aug=True,  # multiscale augmentation for MIL/classification
            return_global=True,  # CLS token only (no patch tokens)
        )
        emb = out[0]  # (vision_cls, text_cls)[0] -> [B, 1024]

    else:
        # provgigapath: model(images) returns [B, 1536] directly.
        # h0mini, uni2, ctranspath: _pool_tokens extracts class token as needed.
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
    Resize((input_size,input_size)) + ToTensor + Normalize pipeline. GenBio-PathFM
    uses its own official mean/std rather than ImageNet normalization.

    In standard_eval variants (provgigapath base) n_crops=1 and transform is the
    timm eval transform (Resize(256)+CenterCrop(224)+ToTensor+Normalize).

    In musk_eval variants (musk base) n_crops=1 and transform is the MUSK transform
    (Resize(384,BICUBIC,antialias)+CenterCrop(384)+ToTensor+Normalize((0.5,0.5,0.5))).
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
            tensor = self.transform(img.convert("RGB"))
        elif self.single_crop_index is not None:
            crops = make_crops(img, n_crops=self.n_crops, crop_size=self.crop_size)
            tensor = self.transform(crops[self.single_crop_index])
        else:
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
                        feats = fc
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
                    # Single-crop path: [B, 3, H, W]
                    # Covers resize_only, standard_eval, musk_eval, and single_crop variants.
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
        json.loads(block_done(b).read_text()) for b in completed_block_dirs(unit_dir)
    ]
    n_total = int(sum(int(m.get("n_patches", 0)) for m in all_bmetas))

    # Preprocess mode description written to the unit manifest.
    if multicrop_combine == "resize_only":
        preprocess_mode = "resize_only_full_image_to_model_input"
        if base_foundation == "genbiopathfm":
            source_image_transform = (
                "full PIL image -> Resize((224,224), default BILINEAR) -> ToTensor -> "
                "Normalize(mean=(0.697,0.575,0.728),std=(0.188,0.240,0.187)); "
                "official GenBio-PathFM preprocessing; no crop"
            )
        else:
            source_image_transform = "full PIL image -> Resize((input_size,input_size)) -> ToTensor -> Normalize; no crop"
    elif multicrop_combine == "standard_eval":
        preprocess_mode = "standard_eval_resize256_centercrop224"
        source_image_transform = (
            "full PIL image -> Resize(256,BICUBIC) -> CenterCrop(224) -> ToTensor -> "
            "Normalize(ImageNet); official ProvGigaPath preprocessing"
        )
    elif multicrop_combine == "musk_eval":
        # Official MUSK preprocessing: Resize(384,BICUBIC,antialias)+CenterCrop(384)
        # +Normalize(IMAGENET_INCEPTION_MEAN=(0.5,0.5,0.5)).
        # ms_aug=True is applied inside the model forward pass (not the transform).
        # Reference: https://github.com/lilab-stanford/MUSK
        preprocess_mode = "musk_eval_resize384_centercrop384"
        source_image_transform = (
            "full PIL image -> Resize(384,BICUBIC,antialias=True) -> CenterCrop(384) -> "
            "ToTensor -> Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)); "
            "official MUSK preprocessing (IMAGENET_INCEPTION_MEAN/STD); "
            "ms_aug=True applied internally in model.forward()"
        )
    else:
        preprocess_mode = multicrop_combine
        source_image_transform = "see variant"

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
        "preprocess_mode": preprocess_mode,
        "source_image_transform": source_image_transform,
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
    single_crop_index = variant["crop_index"]
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

    if combine == "resize_only":
        dim = FOUNDATION_SPECS[base_foundation]["embedding_dim"]
        if base_foundation == "genbiopathfm":
            mode_note = (
                "full image Resize((224,224), default BILINEAR)+Normalize(GenBio PathFM mean/std), "
                f"official preprocessing -> [n, {dim}]  "
                "(genbio-pathfm: https://huggingface.co/genbio-ai/genbio-pathfm)"
            )
        else:
            mode_note = (
                f"full image resized directly to model input, no crop -> [n, {dim}]"
            )
    elif combine == "standard_eval":
        # Official ProvGigaPath preprocessing: Resize(256,BICUBIC)+CenterCrop(224).
        # For 512x512 BraTS-Path patches: 512->256 then center-crop to 224x224.
        mode_note = (
            "full image Resize(256,BICUBIC)+CenterCrop(224), official preprocessing -> [n, dim]  "
            "(provgigapath: https://huggingface.co/prov-gigapath/prov-gigapath)"
        )
    elif combine == "musk_eval":
        # Official MUSK preprocessing: Resize(384,BICUBIC,antialias)+CenterCrop(384).
        # For 512x512 BraTS-Path patches: 512->384 (square, no crop needed) -> [n, dim].
        # ms_aug=True is applied inside the MUSK model.forward() at inference time.
        # NOTE: ms_aug=True increases GPU memory usage. Reduce --batch-size if OOM
        #       (recommended: --batch-size 64 or lower with MUSK on 16 GB VRAM).
        mode_note = (
            "full image Resize(384,BICUBIC,antialias)+CenterCrop(384), official MUSK preprocessing "
            "-> [n, 1024]; ms_aug=True applied internally in model.forward()  "
            "(musk: https://huggingface.co/xiangjx/musk)"
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

    # Select the correct transform based on combine mode:
    #   resize_only   -> crop_transform:   Resize((input_size,input_size))+ToTensor+Normalize (no center crop)
    #                    GenBio-PathFM uses its official mean/std in this path.
    #   standard_eval -> single_transform: timm eval transform, e.g. Resize(256)+CenterCrop(224)
    #                    This is the official ProvGigaPath preprocessing.
    #   musk_eval     -> single_transform: build_musk_transform(384)
    #                    Resize(384,BICUBIC,antialias)+CenterCrop(384)+Normalize((0.5,0.5,0.5))
    #                    This is the official MUSK preprocessing.
    #                    (load_foundation sets both single_transform and crop_transform to
    #                    build_musk_transform for MUSK, so either would work here.)
    #   n_crops > 1   -> crop_transform:   applied to each pre-cropped tile region
    #   else          -> single_transform: timm default eval transform
    if combine == "resize_only":
        transform = crop_transform
    elif combine in ("standard_eval", "musk_eval"):
        transform = single_transform
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
            if meta.get("base_foundation") == "genbiopathfm":
                shape_note = "[n, 4608]  (Resize((224,224), default BILINEAR)+GenBio PathFM Normalize, no crop)"
            else:
                shape_note = "[n, dim]  (full-image resize-only, no crop)"
        elif comb == "standard_eval":
            shape_note = "[n, dim]  (Resize(256)+CenterCrop(224), official ProvGigaPath preprocessing)"
        elif comb == "musk_eval":
            shape_note = (
                "[n, 1024]  (Resize(384,BICUBIC,antialias)+CenterCrop(384), "
                "official MUSK preprocessing; ms_aug=True internally)"
            )
        else:
            shape_note = "[n, dim]"
        print(
            f"[consolidated] {artifact_name}/{split}: "
            f"{meta['n_patches']:,} patches in {meta['n_patient_slide_files']} files  "
            f"X: {shape_note}"
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
            "  h0mini                  full patch resized to 224x224, no crop -> [n,dim]\n"
            "  uni2                    full patch resized to 224x224, no crop -> [n,dim]\n"
            "  virchow2                full patch resized to 224x224, no crop -> [n,dim]\n"
            "  provgigapath            Resize(256,BICUBIC)+CenterCrop(224)    -> [n,1536]\n"
            "                            (official preprocessing; gated HF model)\n"
            "                            set HF_TOKEN env var before running\n"
            "  phikonv2                Resize(224)+ToTensor+Normalize(ImageNet)  -> [n,1024]\n"
            "                            ViT-L/16 DINOv2 on 450M public tiles (TCGA/CPTAC/GTEx)\n"
            "                            NO HF_TOKEN required; public model\n"
            "  genbiopathfm            Resize(224, default BILINEAR)+Normalize(GenBio PathFM mean/std) -> [n,4608]\n"
            "                            aliases: genbio-pathfm, genbio_pathfm\n"
            "                            requires: pip install -U transformers\n"
            "                            uses AutoModel(..., trust_remote_code=True)\n"
            "  musk                    Resize(384,BICUBIC,antialias)+CenterCrop(384) -> [n,1024]\n"
            "                            ms_aug=True applied internally in model.forward()\n"
            "                            (official preprocessing; gated HF model)\n"
            "                            set HF_TOKEN env var before running\n"
            "                            requires: pip install git+https://github.com/lilab-stanford/MUSK.git\n"
            "                            NOTE: ms_aug=True increases GPU memory; use --batch-size 64\n"
            "  h0mini_resize224        alias, still writes under artifact h0mini\n"
            "  provgigapath_resize224  resize-only alias (deviates from official preprocessing)\n"
            "  h0mini_mc5_center       only centre crop from mc5 grid -> [n,dim]\n"
            "  h0mini_mc5_top_right    only top-right crop            -> [n,dim]\n"
            "  h0mini_mc5              all 5 crops stored together    -> [n,5,dim]\n"
            "  h0mini_mc5_mean         mean of 5 crops (legacy)       -> [n,dim]\n"
            "  provgigapath_mc5_center mc5 centre crop                -> [n,1536]\n"
            "  genbiopathfm_mc5_center mc5 centre crop (GenBio PathFM preprocessing) -> [n,4608]\n"
            "  genbiopathfm_mc5        all 5 crops stored together    -> [n,5,4608]\n"
            "  musk_mc5_center         mc5 centre crop (MUSK preprocessing) -> [n,1024]\n"
            "  musk_mc5                all 5 crops stored together    -> [n,5,1024]\n"
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
        help="Batch size. Reduce to 64 or lower when using MUSK (ms_aug=True "
        "increases GPU memory usage inside the model forward pass).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--unit-shards", type=int, default=1)
    parser.add_argument("--val-unit-shards", type=int, default=1)
    parser.add_argument("--flush-every-batches", type=int, default=50)
    parser.add_argument("--mc-crop-size", type=int, default=224)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--no-amp", action="store_true")
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
        v = parse_foundation_variant(f)
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

    mapping = None
    if "train" in splits:
        mapping = read_mapping(args.mapping_csv)
        print(f"mapping rows: {len(mapping):,}")

    for foundation in foundations:
        if "val" in splits:
            extract_split_foundation(args, "val", foundation, val_shards, None, device)
        if "train" in splits:
            extract_split_foundation(
                args, "train", foundation, train_shards, mapping, device
            )


if __name__ == "__main__":
    main()

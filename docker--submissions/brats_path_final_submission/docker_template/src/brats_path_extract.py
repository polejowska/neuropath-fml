#!/usr/bin/env python3
"""
brats_path_extract.py -- offline, inference-time embedding extraction for the
THREE foundations actually used by the packaged ensemble: virchow2, hoptimus1,
genbiopathfm. All three are used in resize-only mode (full 512x512 patch ->
224x224, no crop), matching how the training-time embeddings were produced.

Model loading follows the Submission Kit's documented pattern (flat local
state-dict + hand-built model skeleton), NOT the earlier hf-hub/AutoModel
approach:

  virchow2, hoptimus1:
    - build the model architecture via timm's "local-dir:" scheme, pointed
      at a local directory containing a cached `config.json` -- NOT the
      "hf-hub:repo_id" scheme, which unconditionally fetches config.json
      from the Hub even when pretrained=False, and therefore fails outright
      under `docker run --network none`. "local-dir:" runs through the
      identical local parsing code, just sourced from disk.
    - load a flat local state dict (model.safetensors / pytorch_model.bin)
      from src/foundation_model_weights/<foundation>/, matching the kit's
      documented "Allowed local foundation-model files" pattern exactly
    - this is what timm's pretrained=True already does internally, so the
      resulting weights are identical -- just resolved from local files
      instead of the Hub, with zero network calls at container runtime

  genbiopathfm:
    - Now follows the SAME pattern as the other two: the model class is
      vendored directly as plain local Python (src/genbio_pathfm_model.py,
      copied from the official genbio-ai/genbio-pathfm repo's
      `genbio_pathfm/model.py` -- the "Option 2: pip package" code path in
      that repo's own README, as opposed to "Option 1: HuggingFace
      AutoModel"). A flat `.pth` state dict is loaded into it directly.
    - No transformers.AutoModel, no trust_remote_code, no config.json, no HF
      snapshot directory, no repo-provided code executed at runtime that
      wasn't already reviewed and shipped as part of this image's own source.

Preprocessing uses torchvision.transforms.v2, matching
src/webdataset_loader.py's DEFAULT_IMAGE_TRANSFORMS convention (ToImage ->
Resize -> ToDtype(scale=True) -> Normalize), rather than the legacy
torchvision.transforms API used in an earlier draft of this file.

Provenance notes (extraction recipe correctness, unchanged from before):
  - hoptimus1: bioptimus/H-optimus-1, init_values=1e-5, dynamic_img_size=False,
    Bioptimus model-card mean/std. Output is already [B, 1536] (no token
    indexing needed).
  - virchow2: paige-ai/Virchow2, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU
    (required -- fc2 width mismatch otherwise). Official recipe:
      output = model(x) -> [B, 261, 1280]
      cls = output[:, 0]; patch = output[:, 5:] (skip 4 register tokens)
      embedding = cat([cls, mean(patch, dim=1)], dim=-1) -> [B, 2560]
  - genbiopathfm: genbio-ai/genbio-pathfm. Official recipe:
      embedding = model(x) -> [B, 4608] directly (CLS feature).
    Preprocessing: Resize((224,224), default BILINEAR) + Normalize(
      mean=(0.697,0.575,0.728), std=(0.188,0.240,0.187)).

All three foundations are used with l2_normalize=True.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torchvision.transforms import v2

try:
    import timm
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Install timm first: pip install timm") from exc


N_CLASSES = 10

FOUNDATION_SPECS: Dict[str, Dict[str, Any]] = {
    "hoptimus1": {
        "embedding_dim": 1536,
        "input_size": 224,
        "interpolation": v2.InterpolationMode.BICUBIC,
        # Bioptimus model-card normalization for H-optimus-1.
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
    },
    "virchow2": {
        "embedding_dim": 2560,
        "input_size": 224,
        "interpolation": v2.InterpolationMode.BICUBIC,
        # No foundation-specific mean/std override -- falls back to ImageNet
        # defaults, matching the source extraction script's fallback.
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
    "genbiopathfm": {
        "embedding_dim": 4608,
        "input_size": 224,
        "interpolation": v2.InterpolationMode.BILINEAR,
        "mean": (0.697, 0.575, 0.728),
        "std": (0.188, 0.240, 0.187),
    },
}

EMBEDDING_DIMS = {k: v["embedding_dim"] for k, v in FOUNDATION_SPECS.items()}

# Local weight filenames this container is allowed to ship, per the kit's
# documented pattern. Searched in this order under
# src/foundation_model_weights/<foundation>/.
# "model.pth" is included because it's the canonical filename GenBio-PathFM
# ships under on HuggingFace Hub (genbio-ai/genbio-pathfm/model.pth).
_WEIGHT_FILENAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "checkpoint.pth",
    "model.pth",
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


def build_transform(foundation: str):
    """torchvision.transforms.v2 pipeline, matching
    src/webdataset_loader.py's DEFAULT_IMAGE_TRANSFORMS convention:
    ToImage -> Resize -> ToDtype(scale=True) -> Normalize."""
    spec = FOUNDATION_SPECS[foundation]
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(
                (spec["input_size"], spec["input_size"]),
                interpolation=spec["interpolation"],
            ),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=list(spec["mean"]), std=list(spec["std"])),
        ]
    )


def _find_weight_file(weights_dir: Path) -> Path:
    for name in _WEIGHT_FILENAMES:
        candidate = weights_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No local weight file found under {weights_dir}. Expected one of: {_WEIGHT_FILENAMES}"
    )


def _load_state_dict(weight_path: Path) -> Dict[str, torch.Tensor]:
    if weight_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(weight_path), device="cpu")
    return torch.load(weight_path, map_location="cpu")


def load_foundation(
    foundation: str,
    device: torch.device,
    weights_root: Path,
    compile_model: bool = False,
):
    """Returns (model, transform). weights_root is
    src/foundation_model_weights/ ; each foundation's files live in its own
    subdirectory (weights_root / foundation / ...)."""
    spec = FOUNDATION_SPECS[foundation]
    foundation_dir = weights_root / foundation

    if foundation == "hoptimus1":
        # IMPORTANT: timm's "hf-hub:" prefix ALWAYS fetches config.json from
        # the Hub to resolve the architecture, even with pretrained=False --
        # this happens unconditionally inside timm.create_model() regardless
        # of whether weights are downloaded. Under `docker run --network
        # none` that call fails outright (confirmed: raises
        # LocalEntryNotFoundError). timm's "local-dir:" scheme runs through
        # the identical config-parsing code path but reads config.json from
        # a local directory instead -- see
        # src/foundation_model_weights/README.md for how that file gets
        # there (fetched once at prepare-time, on a machine with network,
        # same as the weight file already is).
        if not (foundation_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Missing {foundation_dir / 'config.json'}. Run "
                f"`python prepare_submission.py --populate-weights` on a machine "
                f"with network access to fetch it (see foundation_model_weights/README.md)."
            )
        model = timm.create_model(
            f"local-dir:{foundation_dir}",
            pretrained=False,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        state_dict = _load_state_dict(_find_weight_file(foundation_dir))
        model.load_state_dict(state_dict)
        model.eval().to(device)

    elif foundation == "virchow2":
        from timm.layers import SwiGLUPacked

        # Same "local-dir:" fix as hoptimus1 above -- see the comment there.
        if not (foundation_dir / "config.json").is_file():
            raise FileNotFoundError(
                f"Missing {foundation_dir / 'config.json'}. Run "
                f"`python prepare_submission.py --populate-weights` on a machine "
                f"with network access to fetch it (see foundation_model_weights/README.md)."
            )
        model = timm.create_model(
            f"local-dir:{foundation_dir}",
            pretrained=False,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        state_dict = _load_state_dict(_find_weight_file(foundation_dir))
        model.load_state_dict(state_dict)
        model.eval().to(device)

    elif foundation == "genbiopathfm":
        # Vendored model class -- no transformers.AutoModel, no
        # trust_remote_code, no HF snapshot directory. src/genbio_pathfm_model.py
        # is a straight copy of the official genbio-ai/genbio-pathfm repo's
        # `genbio_pathfm/model.py` (the repo's own "Option 2: pip package"
        # code path). GenBio_PathFM_Inference builds the ViT skeleton, loads
        # the flat local .pth state dict (torch.load(..., weights_only=True)),
        # and moves the model to device internally.
        from .genbio_pathfm_model import GenBio_PathFM_Inference

        weight_path = _find_weight_file(foundation_dir)
        model = GenBio_PathFM_Inference(str(weight_path), device=str(device))
        model.eval()

    else:
        raise ValueError(
            f"Unsupported foundation {foundation!r}; this offline copy only supports {sorted(FOUNDATION_SPECS)}"
        )

    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:  # pragma: no cover
            print(f"[model] torch.compile failed for {foundation}: {exc}", flush=True)

    return model, build_transform(foundation)


def _pool_tokens(out: Any) -> torch.Tensor:
    """Generic token pooling (used for hoptimus1, which already returns a
    [B, dim] pooled/CLS tensor directly -- this is effectively a passthrough
    with a defensive unwrap for dict/tuple/3-D outputs)."""
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
    """Byte-identical extraction recipe per foundation, matching the official
    model-card usage for each."""
    if foundation == "virchow2":
        out = model(
            images
        )  # [B, 261, 1280]: CLS + 4 register tokens + 256 patch tokens
        class_token = out[:, 0]
        patch_tokens = out[:, 5:]  # skip the 4 register tokens
        emb = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=1)  # [B, 2560]

    elif foundation == "genbiopathfm":
        emb = model(images)  # [B, 4608] CLS feature, returned directly

    elif foundation == "hoptimus1":
        emb = _pool_tokens(model(images))  # already [B, 1536]

    else:
        raise ValueError(f"Unsupported foundation {foundation!r}")

    if not torch.is_tensor(emb):
        raise TypeError(f"Model returned non-tensor: {type(emb)}")
    if emb.ndim != 2:
        emb = emb.flatten(1)
    if l2_normalize:
        emb = F.normalize(emb.float(), p=2, dim=1)
    return emb

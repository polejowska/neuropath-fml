#!/usr/bin/env python3
"""
prepare_submission.py -- populate docker_template/ from your real trained
model directory and local machine, and (optionally) produce a local
reference prediction to diff against the built container's output.

This replaces the single-foundation preparation.py test harness: it now
covers all three foundations (virchow2, hoptimus1, genbiopathfm) and reads
everything it needs from manifest.json rather than hardcoding one head.

Usage
-----
# 1. Copy manifest.json + heads/ from your trained model directory into
#    docker_template/src/ckpts/  (already done for this handoff, but re-run
#    this if you retrain):
python prepare_submission.py --model-dir /path/to/official_models/vhg_aug_ivy14950_source_soft_full_exact_v1 --copy-heads

# 2. Populate docker_template/src/foundation_model_weights/ from your local
#    machine's HF cache (run this on the machine that has all 3 models
#    cached -- e.g. wherever you ran extraction). Extracts flat weight files
#    for all three foundations -- virchow2/hoptimus1 as .safetensors via
#    timm, genbiopathfm as model.pth via huggingface_hub -- matching the
#    kit's documented single-file pattern for every foundation:
python prepare_submission.py --populate-weights

# 3. Build a small local /input tar from your own val shards and run the
#    SAME extraction + inference code the container will run, right here, so
#    you get a local_reference_predictions.csv to diff against the
#    container's predictions.csv after `./scripts/02_run_docker_image.sh`:
python prepare_submission.py --make-sample-tar --val-glob "data/val-shard-*.tar" --n-samples 10

Then:
    cd docker_template
    IMAGE_NAME=brats_path_vhg IMAGE_TAG=latest ./scripts/01_build_image.sh
    INPUT_DIR=../test_artifacts OUTPUT_DIR=../docker_output \\
      IMAGE_NAME=brats_path_vhg IMAGE_TAG=latest ./scripts/02_run_docker_image.sh
    diff <(sort ../test_artifacts/local_reference_predictions.csv) \\
         <(sort ../docker_output/predictions.csv)
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path
from typing import List

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SCRIPT_DIR / "docker_template"
CKPTS_DIR = TEMPLATE_DIR / "src" / "ckpts"
WEIGHTS_DIR = TEMPLATE_DIR / "src" / "foundation_model_weights"


# -----------------------------------------------------------------------------
# 1. Copy manifest.json + heads/ from a (re)trained model directory
# -----------------------------------------------------------------------------


def copy_heads(model_dir: Path) -> None:
    manifest_path = model_dir / "manifest.json"
    heads_dir = model_dir / "heads"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found under {model_dir}")
    if not heads_dir.is_dir():
        raise FileNotFoundError(f"heads/ not found under {model_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {"virchow2", "hoptimus1", "genbiopathfm"}
    got = set(manifest.get("foundations", []))
    if got != expected:
        print(
            f"[warn] manifest foundations {sorted(got)} != expected {sorted(expected)}; "
            f"src/brats_path_extract.py only implements {sorted(expected)}."
        )

    CKPTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, CKPTS_DIR / "manifest.json")

    dest_heads = CKPTS_DIR / "heads"
    if dest_heads.exists():
        shutil.rmtree(dest_heads)
    shutil.copytree(heads_dir, dest_heads)

    n_files = sum(1 for _ in dest_heads.rglob("*.joblib"))
    print(f"[ckpts] copied manifest.json + {n_files} head files -> {CKPTS_DIR}")
    if n_files != int(manifest.get("n_heads", -1)):
        print(
            f"[warn] copied {n_files} .joblib files but manifest.json says n_heads={manifest.get('n_heads')}"
        )


# -----------------------------------------------------------------------------
# 2. Populate local flat weight files for all 3 foundations
# -----------------------------------------------------------------------------


# These downloads use whatever HF cache huggingface_hub/timm would normally
# resolve to (respecting HF_HOME/HF_HUB_CACHE if you've set them, or the
# standard ~/.cache/huggingface default otherwise) -- so anything you've
# already downloaded gets reused rather than re-fetched every run. The one
# thing checked for is whether that resolved location happens to sit INSIDE
# this project (which is what would leak a multi-GB raw HF cache tree into
# the Docker build context via `COPY src/`) -- see _warn_if_cache_inside_project()
# below. If you see that warning, set HF_HOME to somewhere outside this
# project (e.g. `export HF_HOME=~/.cache/huggingface`) rather than leaving
# it as-is.


def _warn_if_cache_inside_project() -> None:
    try:
        from huggingface_hub import constants as hf_constants

        cache_root = Path(hf_constants.HF_HUB_CACHE).resolve()
    except Exception:
        return
    try:
        cache_root.relative_to(SCRIPT_DIR.resolve())
    except ValueError:
        return  # not inside the project -- nothing to warn about
    print(
        f"[warn] Your HF Hub cache ({cache_root}) resolves to somewhere INSIDE "
        f"this project. That's fine for prepare_submission.py itself, but if it's "
        f"also inside docker_template/ specifically, a full raw HF cache tree "
        f"(blobs/snapshots, easily multiple GB per foundation) could end up in "
        f"the Docker build context. .dockerignore already excludes these patterns "
        f"defensively, but consider setting HF_HOME to somewhere outside this "
        f"project (e.g. `export HF_HOME=~/.cache/huggingface`) to avoid the "
        f"wasted disk/context-transfer time entirely.",
    )


def _extract_timm_state_dict(hf_id: str, dest_file: Path, **create_kwargs) -> None:
    """Load a timm model with pretrained=True (reuses your existing local HF
    cache download if you already have one -- see the note above) and save
    its state dict as a flat .safetensors file -- exactly the format
    src/brats_path_extract.py::load_foundation() loads directly with
    model.load_state_dict(). Also caches config.json alongside it, so the
    container can build the architecture via timm's "local-dir:" scheme at
    runtime instead of "hf-hub:", which fetches config.json from the Hub
    even when pretrained=False and therefore breaks under `docker run
    --network none`."""
    import timm
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    _warn_if_cache_inside_project()
    print(
        f"[hf-weights] loading {hf_id} via timm (pretrained=True) to extract its state dict..."
    )
    model = timm.create_model(hf_id, pretrained=True, **create_kwargs)
    state_dict = model.state_dict()
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    # safetensors requires contiguous tensors and cannot hold non-tensor
    # buffers; state_dict() from a timm model is all plain tensors already.
    save_file({k: v.contiguous() for k, v in state_dict.items()}, str(dest_file))
    print(
        f"[hf-weights] wrote flat state dict ({len(state_dict)} tensors) -> {dest_file}"
    )
    del model
    import gc

    gc.collect()

    # Cache config.json for the runtime "local-dir:" build path.
    repo_id = hf_id.removeprefix("hf-hub:")
    config_dest = dest_file.parent / "config.json"
    downloaded_cfg = hf_hub_download(repo_id=repo_id, filename="config.json")
    shutil.copy2(downloaded_cfg, config_dest)
    print(f"[hf-weights] cached {repo_id}/config.json -> {config_dest}")


def _download_genbiopathfm_weight(dest_file: Path) -> None:
    """genbiopathfm now uses the same flat-weight-file pattern as the other
    two foundations: download the official model.pth checkpoint via
    huggingface_hub.hf_hub_download (reuses your existing local HF cache
    download if you already have one; weights only, no repo Python code
    executed, no AutoModel/trust_remote_code involved at all) and copy it
    into place. See src/genbio_pathfm_model.py for the vendored model class
    this weight file is loaded into at container runtime."""
    from huggingface_hub import hf_hub_download

    _warn_if_cache_inside_project()
    repo_id = "genbio-ai/genbio-pathfm"
    print(
        f"[hf-weights] downloading {repo_id}/model.pth via huggingface_hub "
        f"(weights only -- no repo code is executed by this download)..."
    )
    downloaded_path = hf_hub_download(repo_id=repo_id, filename="model.pth")

    dest_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, dest_file)
    print(f"[hf-weights] copied {downloaded_path} -> {dest_file}")


def populate_weights(force_redownload: bool) -> None:
    """Populate docker_template/src/foundation_model_weights/ for all 3
    foundations. All three now use the kit's flat-single-file pattern:
    virchow2/hoptimus1 as .safetensors extracted via timm, genbiopathfm as
    model.pth downloaded directly via huggingface_hub."""
    from timm.layers import SwiGLUPacked
    import torch

    if force_redownload:
        import os

        os.environ.pop("HF_HUB_OFFLINE", None)

    hoptimus1_file = WEIGHTS_DIR / "hoptimus1" / "model.safetensors"
    hoptimus1_cfg = WEIGHTS_DIR / "hoptimus1" / "config.json"
    if hoptimus1_file.exists() and hoptimus1_cfg.exists() and not force_redownload:
        print(
            f"[hf-weights] hoptimus1: {hoptimus1_file} + config.json already exist; reusing them."
        )
    else:
        _extract_timm_state_dict(
            "hf-hub:bioptimus/H-optimus-1",
            hoptimus1_file,
            init_values=1e-5,
            dynamic_img_size=False,
        )

    virchow2_file = WEIGHTS_DIR / "virchow2" / "model.safetensors"
    virchow2_cfg = WEIGHTS_DIR / "virchow2" / "config.json"
    if virchow2_file.exists() and virchow2_cfg.exists() and not force_redownload:
        print(
            f"[hf-weights] virchow2: {virchow2_file} + config.json already exist; reusing them."
        )
    else:
        _extract_timm_state_dict(
            "hf-hub:paige-ai/Virchow2",
            virchow2_file,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    genbiopathfm_file = WEIGHTS_DIR / "genbiopathfm" / "model.pth"
    if genbiopathfm_file.exists() and not force_redownload:
        print(
            f"[hf-weights] genbiopathfm: {genbiopathfm_file} already exists; reusing it."
        )
    else:
        _download_genbiopathfm_weight(genbiopathfm_file)

    print(
        "[hf-weights] done. Populated: hoptimus1, virchow2, genbiopathfm -- "
        "all three as flat local weight files, no trust_remote_code anywhere."
    )


# -----------------------------------------------------------------------------
# 3. Sample a small local /input tar + local reference prediction
# -----------------------------------------------------------------------------


def discover_shards(pattern: str) -> List[Path]:
    import glob

    shards = [Path(p) for p in sorted(glob.glob(pattern))]
    if not shards:
        raise FileNotFoundError(f"No shards found matching {pattern!r}")
    return shards


def sample_val_shard(val_glob: str, n_samples: int, out_tar_path: Path) -> List[str]:
    import webdataset as wds

    shards = discover_shards(val_glob)
    print(f"[sample] found {len(shards)} val shard(s); reading from {shards[0]}")
    ds = wds.WebDataset([str(s) for s in shards], shardshuffle=False, empty_check=False)

    out_tar_path.parent.mkdir(parents=True, exist_ok=True)
    written_keys: List[str] = []
    with tarfile.open(out_tar_path, "w") as tar:
        for sample in ds:
            if len(written_keys) >= n_samples:
                break
            key = sample.get("__key__", "")
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            img_bytes, img_ext = None, None
            for ext in ("jpg", "jpeg", "png", "webp"):
                if ext in sample:
                    img_bytes, img_ext = sample[ext], ext
                    break
            if img_bytes is None:
                continue
            info = tarfile.TarInfo(name=f"{key}.{img_ext}")
            info.size = len(img_bytes)
            tar.addfile(info, io.BytesIO(img_bytes))
            written_keys.append(str(key))
    print(f"[sample] wrote {len(written_keys)} patches -> {out_tar_path}")
    return written_keys


def run_local_reference(
    sample_tar_dir: Path, out_csv: Path, batch_size: int, tta_aug: int
) -> None:
    """Run the EXACT container code path (docker_template/src/inference.py's
    run_inference) locally against the sample tar, so the resulting CSV is a
    genuine reference to diff the container's own predictions.csv against."""
    import os

    os.environ.setdefault(
        "HF_HUB_OFFLINE", "0"
    )  # allow local cache resolution either way
    os.environ["BRATS_PATH_BATCH_SIZE"] = str(batch_size)
    os.environ["BRATS_PATH_TTA_AUG"] = str(tta_aug)

    sys.path.insert(0, str(TEMPLATE_DIR))
    from src.inference import run_inference  # noqa: E402

    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inference(str(sample_tar_dir), str(out_dir))

    produced = out_dir / "predictions.csv"
    if produced != out_csv:
        shutil.copy2(produced, out_csv)
    print(f"[local-ref] wrote local reference predictions -> {out_csv}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Trained model directory (manifest.json + heads/). Required with --copy-heads.",
    )
    ap.add_argument(
        "--copy-heads",
        action="store_true",
        help="Copy manifest.json + heads/ from --model-dir into docker_template/src/ckpts/.",
    )
    ap.add_argument(
        "--populate-weights",
        action="store_true",
        help="Populate docker_template/src/foundation_model_weights/ for all 3 foundations "
        "(virchow2/hoptimus1 as .safetensors via timm, genbiopathfm as model.pth via huggingface_hub).",
    )
    ap.add_argument("--force-redownload", action="store_true")
    ap.add_argument(
        "--make-sample-tar",
        action="store_true",
        help="Sample N val patches into a small /input-style tar and run a local reference prediction.",
    )
    ap.add_argument("--val-glob", default="data/val-shard-*.tar")
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--out-dir", default="./test_artifacts")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--tta-aug",
        type=int,
        default=0,
        help="0 for a fast clean-only local reference; 16 to match your official TTA inference exactly.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.copy_heads:
        if args.model_dir is None:
            raise SystemExit("--copy-heads requires --model-dir")
        copy_heads(args.model_dir)

    if args.populate_weights:
        populate_weights(args.force_redownload)

    if args.make_sample_tar:
        out_dir = Path(args.out_dir).resolve()
        sample_tar = out_dir / "sample_val.tar"
        sample_val_shard(args.val_glob, args.n_samples, sample_tar)
        # run_inference expects a *directory* of shards, not a single file path.
        run_local_reference(
            sample_tar.parent,
            out_dir / "local_reference_predictions.csv",
            batch_size=args.batch_size,
            tta_aug=args.tta_aug,
        )

    if not any([args.copy_heads, args.populate_weights, args.make_sample_tar]):
        print(
            "Nothing to do -- pass --copy-heads, --populate-weights, and/or --make-sample-tar. See --help."
        )


if __name__ == "__main__":
    main()

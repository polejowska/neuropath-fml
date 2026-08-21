"""
inference.py -- high-level inference entrypoint for the packaged Virchow2 +
H-optimus-1 + GenBio-PathFM ensemble.

The Docker runner calls exactly:

    run_inference(input_dir=Path("/input"), output_dir=Path("/output"))

Flow:

  for each foundation in manifest["foundations"] (sequential -- one
      foundation's model + embeddings live in memory at a time):
    1. load the foundation's model + resize-only transform
    2. extract embeddings for every patch in every /input shard, in
       deterministic sorted-shard order
    3. keep the full embedding matrix for this foundation (all three are
       assembled into one {foundation: embeddings} bundle before scoring)

  -> load every fitted chunk head (sgd_lr_log_a3e-5 + ridge_lsqr_a10) via
     brats_path_ensemble.load_fitted_ensemble(), which reconstructs a
     FittedChunkEnsemble -- an inference-only class whose methods are
     copied VERBATIM from the real, validated training/eval script's
     BratsPath2025ChunkedSGDEnsemble (see brats_path_ensemble.py's own
     docstring for why this matters: an earlier version of this file
     reimplemented that math independently and it was NOT equivalent)
  -> ensemble.predict_proba_tta(X_bundle, tta_aug, tta_keep, tta_seed)
     (optionally row-chunked purely for memory safety -- see
     _score_ensemble_in_row_chunks below)
  -> argmax -> write predictions.csv

ADAPT: TTA is ON by default (16 masked passes, keep=0.90, seed=42), matching
the official run (`python genbiopathfm_ivy_end_to_end.py --tta-aug 16
--tta-keep 0.9 --tta-seed 42`). Set BRATS_PATH_TTA_AUG=0 in the container
environment to run clean-only inference instead (~17x faster). There is no
automatic fallback of any kind: whatever BRATS_PATH_TTA_AUG/KEEP/SEED say is
exactly what runs, regardless of how long it takes.

ADAPT: PREDICTION_FILENAME and the CSV schema (SubjectID, Prediction) match
example_base_line/'s worked reference exactly. Still confirm these match what
the BraTS-Path 2026 evaluator actually expects before submitting.
"""

from __future__ import annotations

import csv
import gc
import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import torch

try:
    from . import inference_dependencies as deps
    from . import brats_path_ensemble as ensemble_mod
except ImportError:  # Allows direct execution from src/ during local debugging.
    import inference_dependencies as deps  # type: ignore
    import brats_path_ensemble as ensemble_mod  # type: ignore

# ADAPT: Rename/move these folders if your checkpoints or foundation-model
# weights live somewhere else inside the image.
CHECKPOINT_DIR = Path(__file__).resolve().parent / "ckpts"
FOUNDATION_MODEL_DIR = Path(__file__).resolve().parent / "foundation_model_weights"

# ADAPT: Change this file name or CSV schema if the challenge output differs.
PREDICTION_FILENAME = "predictions.csv"

N_CLASSES = deps.N_CLASSES
CLASS_NAMES = deps.CLASS_NAMES


def _preview(values: list, limit: int = 3) -> str:
    shown = [str(v) for v in values[:limit]]
    if len(values) > limit:
        shown.append(f"... ({len(values) - limit} more)")
    return ", ".join(shown)


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _score_ensemble_in_row_chunks(
    ensemble: "ensemble_mod.FittedChunkEnsemble",
    X_bundle: Mapping[str, np.ndarray],
    tta_aug: int,
    tta_keep: float,
    tta_seed: int,
    chunk_rows: int,
) -> np.ndarray:
    """Call ensemble.predict_proba_tta() over row-slices of X_bundle instead
    of the whole dataset in one call -- purely a memory-safety measure for
    large test sets (holding all 3 foundations' full embeddings, plus one
    masked copy per TTA pass, in RAM at once can add up). This does NOT
    change the algorithm: _mask_bundle / _raw_avg_proba / predict_proba_tta
    are called completely unmodified per chunk, and there is no cross-row
    dependency anywhere in this ensemble (every row is scored
    independently), so chunking by rows and concatenating results is exactly
    equivalent row-for-row to a single unchunked call -- EXCEPT that each
    chunk gets its own fresh np.random.default_rng(tta_seed) at the top of
    predict_proba_tta(), so the specific TTA masks drawn for a given row
    will differ from what one unchunked call would have drawn for that same
    row. That doesn't matter here: there is no external "ground truth" mask
    sequence to match (the hidden test set was never seen during training/
    validation), only internal self-consistency for a given
    (chunk_rows, tta_seed) choice, which this preserves.
    """
    n_rows = int(next(iter(X_bundle.values())).shape[0])
    if chunk_rows <= 0 or n_rows <= chunk_rows:
        return ensemble.predict_proba_tta(
            X_bundle, tta_aug=tta_aug, tta_keep=tta_keep, tta_seed=tta_seed
        )

    parts = []
    n_chunks = -(-n_rows // chunk_rows)
    for i, start in enumerate(range(0, n_rows, chunk_rows), start=1):
        stop = min(start + chunk_rows, n_rows)
        X_chunk = {fn: X_bundle[fn][start:stop] for fn in X_bundle}
        parts.append(
            ensemble.predict_proba_tta(
                X_chunk, tta_aug=tta_aug, tta_keep=tta_keep, tta_seed=tta_seed
            )
        )
        print(
            f"[ensemble] scored row-chunk {i}/{n_chunks} ({stop:,}/{n_rows:,} rows)",
            flush=True,
        )
    return np.concatenate(parts, axis=0)


def run_inference(input_dir: Path, output_dir: Path) -> Path:
    """Required entrypoint called by run.py."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[inference] Step 1: discovering input WebDataset tar shards.", flush=True)
    from .webdataset_loader import discover_tar_shards

    shards = discover_tar_shards(input_dir)
    print(f"[inference] Step 1 output: found {len(shards)} shard(s).", flush=True)
    print(f"[inference] Input shard preview: {_preview(shards)}", flush=True)

    print("[inference] Step 2: loading manifest and configuration.", flush=True)
    manifest = ensemble_mod.load_manifest(CHECKPOINT_DIR)
    foundations = tuple(str(x) for x in manifest["foundations"])
    dimensions = ensemble_mod.foundation_dimensions(manifest)

    # ADAPT: override via environment if you need a different runtime config
    # without rebuilding the image.
    batch_size = int(os.environ.get("BRATS_PATH_BATCH_SIZE", "64"))
    tta_aug = int(os.environ.get("BRATS_PATH_TTA_AUG", "16"))
    tta_keep = float(os.environ.get("BRATS_PATH_TTA_KEEP", "0.90"))
    tta_seed = int(os.environ.get("BRATS_PATH_TTA_SEED", "42"))
    num_workers = int(os.environ.get("BRATS_PATH_NUM_WORKERS", "0"))
    # Memory-safety row-chunking for the ensemble-scoring step only (see
    # _score_ensemble_in_row_chunks above); 0 disables chunking entirely.
    score_chunk_rows = int(os.environ.get("BRATS_PATH_SCORE_CHUNK_ROWS", "50000"))

    device = _auto_device()
    print("[inference] Step 3: selecting runtime device and configuration.", flush=True)
    print(f"[inference] Step 3 output: device={device}", flush=True)
    if device.type == "cuda":
        print(
            f"[inference] CUDA device name: {torch.cuda.get_device_name(0)}", flush=True
        )
    print(
        f"[inference] foundations={foundations}, dimensions={dimensions}, "
        f"n_heads={manifest['n_heads']}",
        flush=True,
    )
    if tta_aug > 0:
        print(
            f"[inference] TTA enabled: 1 clean + {tta_aug} masked passes, "
            f"keep={tta_keep}, seed={tta_seed}, batch={batch_size}, "
            f"score_chunk_rows={score_chunk_rows or 'disabled'}",
            flush=True,
        )
    else:
        print("[inference] TTA disabled: clean inference only.", flush=True)

    print("[inference] Step 4: extracting embeddings for all foundations.", flush=True)
    X_bundle: Dict[str, np.ndarray] = {}
    reference_keys = None

    for i, foundation in enumerate(foundations, start=1):
        print(
            f"[inference] Foundation {i}/{len(foundations)}: {foundation}", flush=True
        )

        keys, X = deps.extract_foundation_embeddings(
            foundation,
            shards,
            device,
            FOUNDATION_MODEL_DIR,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        expected_dim = dimensions[foundation]
        if X.shape[1] != expected_dim:
            raise RuntimeError(
                f"{foundation}: extracted dimension {X.shape[1]} does not match "
                f"trained head dimension {expected_dim}"
            )
        print(f"[inference] {foundation}: extracted matrix shape={X.shape}", flush=True)

        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise RuntimeError(
                f"{foundation}: patch key order differs from the first foundation. "
                "This should not happen with num_workers=0 and shuffling disabled -- "
                "check BRATS_PATH_NUM_WORKERS and shard determinism."
            )
        X_bundle[foundation] = X

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[inference] {foundation}: done; feature-extractor model released.",
            flush=True,
        )

    if reference_keys is None or not X_bundle:
        raise RuntimeError("No predictions were produced (no foundations processed)")

    print(
        "[inference] Step 5: loading the fitted ensemble (all heads, all foundations).",
        flush=True,
    )
    ensemble = ensemble_mod.load_fitted_ensemble(CHECKPOINT_DIR, manifest)

    print(
        "[inference] Step 6: scoring (source_mean aggregation + TTA, exactly as trained).",
        flush=True,
    )
    t_score = time.time()
    final = _score_ensemble_in_row_chunks(
        ensemble,
        X_bundle,
        tta_aug=tta_aug,
        tta_keep=tta_keep,
        tta_seed=tta_seed,
        chunk_rows=score_chunk_rows,
    )
    print(
        f"[inference] Step 6 output: scored {len(reference_keys):,} rows in {time.time() - t_score:.1f}s",
        flush=True,
    )
    used = "tta" if tta_aug > 0 else "clean"

    prediction = final.argmax(axis=1).astype(int)

    print("[inference] Step 7: writing predictions CSV.", flush=True)
    output_path = output_dir / PREDICTION_FILENAME
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SubjectID", "Prediction"])
        for subject_id, pred in zip(reference_keys, prediction.tolist()):
            writer.writerow([subject_id, int(pred)])

    counts = dict(Counter(prediction.tolist()))
    print(
        f"[inference] Step 7 output: wrote {len(prediction)} prediction(s) ({used} pipeline) to {output_path}",
        flush=True,
    )
    print(
        "[inference] Prediction label counts: "
        + ", ".join(f"{CLASS_NAMES[c]}={counts.get(c, 0):,}" for c in range(N_CLASSES)),
        flush=True,
    )

    return output_path

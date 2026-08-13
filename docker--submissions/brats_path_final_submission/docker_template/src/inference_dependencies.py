"""
inference_dependencies.py -- embedding extraction for the packaged Virchow2 +
H-optimus-1 + GenBio-PathFM ensemble.

This module now handles ONLY on-the-fly embedding extraction from /input
WebDataset tar shards, one foundation at a time (same "one foundation ->
one model in memory" policy as training). The manifest loading, head
loading, and probability-aggregation/TTA math that used to live here has
been REMOVED and replaced by brats_path_ensemble.py, which vendors the
real, validated BratsPath2025ChunkedSGDEnsemble class from
genbiopathfm_ivy_end_to_end.py verbatim -- see that module's docstring for
why the previous version of this file's independent reimplementation was
not equivalent to the real thing.

Logging follows the kit's example_base_line convention: bracketed section
tags ([data], [model], [ensemble], ...), first-batch detail previews, and
end-of-run count summaries, plus periodic progress lines with real
totals/percentages/ETA for foundations/batches that take a long time.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from . import brats_path_extract as extract
from .webdataset_loader import build_inference_dataloader, count_shard_patches

N_CLASSES = 10
CLASS_NAMES = {
    0: "CT", 1: "DM", 2: "IC", 3: "LI", 4: "MP",
    5: "NC", 6: "PL", 7: "PN", 8: "WM", 9: "NOTA",
}

# How often to emit a progress log line, in batches.
_LOG_EVERY_N_BATCHES = 5


def _preview(values: list, limit: int = 3) -> str:
    shown = [str(v) for v in values[:limit]]
    if len(values) > limit:
        shown.append(f"... ({len(values) - limit} more)")
    return ", ".join(shown)


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# -----------------------------------------------------------------------------
# Embedding extraction (on the fly, from /input tar shards)
# -----------------------------------------------------------------------------


def extract_foundation_embeddings(
    foundation: str,
    shards: List[str],
    device: torch.device,
    weights_root: Path,
    batch_size: int,
    num_workers: int = 0,
) -> Tuple[List[str], np.ndarray]:
    """Extract embeddings for every patch in every shard, for one foundation,
    in deterministic shard order. Returns (keys, X) with X rows in the exact
    same order as keys are encountered."""
    total_patches = count_shard_patches(shards)
    total_batches = max(1, -(-total_patches // max(1, batch_size)))
    print(
        f"[data] {foundation}: {total_patches:,} patches to extract across "
        f"{len(shards)} shard(s), ~{total_batches:,} batches of {batch_size}",
        flush=True,
    )

    print(f"[model] Loading {foundation} feature extractor.", flush=True)
    model, transform = extract.load_foundation(foundation, device, weights_root, compile_model=False)
    print(f"[model] {foundation} feature extractor ready on device: {device}", flush=True)

    loader = build_inference_dataloader(
        shards=shards, batch_size=batch_size, num_workers=num_workers, image_transform=transform
    )

    keys: List[str] = []
    feats: List[np.ndarray] = []
    t0 = time.time()
    batch_idx = 0
    with torch.inference_mode():
        for batch_keys, batch_images in loader:
            batch_idx += 1
            if batch_idx == 1:
                print(
                    f"[data] {foundation}: first batch shape={tuple(batch_images.shape)}, "
                    f"dtype={batch_images.dtype}, subject preview: {_preview(list(batch_keys))}",
                    flush=True,
                )
            batch_images = batch_images.to(device, non_blocking=True)
            emb = extract.extract_features(model, batch_images, foundation, l2_normalize=True)
            feats.append(emb.detach().cpu().float().numpy())
            keys.extend(str(k) for k in batch_keys)

            if batch_idx % _LOG_EVERY_N_BATCHES == 0 or batch_idx == total_batches:
                n_done = len(keys)
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0.0
                pct = 100.0 * n_done / max(total_patches, 1)
                remaining = (total_patches - n_done) / rate if rate > 0 else float("inf")
                print(
                    f"[data] {foundation}: extracting batch {batch_idx}/{total_batches} "
                    f"({n_done:,}/{total_patches:,} patches, {pct:.1f}%) "
                    f"-- elapsed {_format_eta(elapsed)}, ~{_format_eta(remaining)} remaining "
                    f"({rate:.1f} patches/s)",
                    flush=True,
                )

    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, extract.EMBEDDING_DIMS[foundation]), dtype=np.float32)
    print(
        f"[data] {foundation}: extraction complete -- {len(keys):,} patches "
        f"in {_format_eta(time.time() - t0)}",
        flush=True,
    )

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return keys, X

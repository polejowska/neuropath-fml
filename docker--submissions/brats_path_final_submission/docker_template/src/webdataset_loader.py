"""WebDataset loader utilities for deterministic inference.

The challenge input is expected to be one or more tar shards. Inference must
not shuffle samples, so both WebDataset shard/sample shuffling and PyTorch
DataLoader shuffling are disabled here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import webdataset as wds


DEFAULT_IMAGE_TRANSFORMS = v2.Compose(
    [
        # ADAPT: Replace image resizing/normalization with your model's
        # preprocessing. Add normalization if your checkpoint expects it.
        v2.ToImage(),
        v2.Resize((512, 512)),
        v2.ToDtype(torch.float, scale=True),
    ]
)


def to_filename(value: str | bytes) -> str:
    """Convert a WebDataset __key__ value into a CSV-safe subject id."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def discover_tar_shards(input_dir: str | Path) -> list[str]:
    """Find tar shard files under input_dir in deterministic order."""
    input_path = Path(input_dir)
    shard_paths = sorted(path for path in input_path.rglob("*.tar") if path.is_file())
    if not shard_paths:
        raise FileNotFoundError(f"No .tar WebDataset shards found under: {input_path}")
    return [str(path) for path in shard_paths]


def build_inference_webdataset(
    shards: str | Iterable[str], image_transform=DEFAULT_IMAGE_TRANSFORMS
):
    """Build a non-shuffling WebDataset for inference."""
    return (
        wds.WebDataset(shards, shardshuffle=False)
        .decode("pil")
        .to_tuple("__key__", "jpg")
        .map_tuple(to_filename, image_transform)
    )


def build_inference_dataloader(
    shards: str | Iterable[str],
    batch_size: int = 32,
    num_workers: int = 0,
    image_transform=DEFAULT_IMAGE_TRANSFORMS,
) -> DataLoader:
    """Build a deterministic DataLoader for WebDataset tar-shard inference."""
    # ADAPT: Add a custom collate_fn or tune DataLoader settings here. Keep
    # shuffle=False for deterministic inference.
    dataset = build_inference_webdataset(
        shards=shards,
        image_transform=image_transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )


# ADAPT (addition, not part of the original template): cheap pre-scan so
# progress logging can report real totals/percentages/ETA instead of an
# unbounded counter. Reads tar headers only -- no image decoding.
def count_shard_patches(shards: str | Iterable[str]) -> int:
    import tarfile

    shard_list = [shards] if isinstance(shards, str) else list(shards)
    total = 0
    for shard in shard_list:
        with tarfile.open(shard, "r") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.lower().endswith(".jpg"):
                    total += 1
    return total

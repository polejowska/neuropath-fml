from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import json
import math
import os
import platform
import random
import re
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
except Exception as exc:
    raise SystemExit(
        "Missing PyTorch/torchvision. Install a PyTorch build appropriate for "
        "your CUDA/MPS environment before running this script."
    ) from exc

try:
    import timm
    from timm.data import resolve_model_data_config
except Exception:
    # Kept lazy so --help, --dry-run and --audit-only remain usable on a
    # controller node. Actual extraction checks this again before model loading.
    timm = None
    resolve_model_data_config = None


SCRIPT_VERSION = "ivygap_live_embeddings.raw.v2"
SCHEMA_VERSION = "ivygap_embedding_blocks.v2_raw"

# This exact map is used by the Ivy GAP patch writer supplied in the conversation.
BRATS_LABEL_TO_INT: Dict[str, int] = {
    "CT": 0,
    "DM": 1,
    "IC": 2,
    "LI": 3,
    "MP": 4,
    "NC": 5,
    "PL": 6,
    "PN": 7,
    "WM": 8,
    "NOTA": 9,
}
DEFAULT_IVY_CLASSES = ("CT", "IC", "MP", "NC", "PN")

FOUNDATIONS: Dict[str, Dict[str, Any]] = {
    "virchow2": {
        "hf_id": "hf-hub:paige-ai/Virchow2",
        "input_size": 224,
        "embedding_dim": 2560,
        "pooling": "concat(class_token, mean(patch_tokens_without_4_register_tokens))",
        "normalization": "timm pretrained_cfg mean/std",
    },
    "hoptimus1": {
        "hf_id": "hf-hub:bioptimus/H-optimus-1",
        "input_size": 224,
        "embedding_dim": 1536,
        "pooling": "model_output",
        "mean": (0.707223, 0.578729, 0.703617),
        "std": (0.211883, 0.230117, 0.177517),
        "normalization": "Bioptimus model-card mean/std",
    },
    "genbiopathfm": {
        "hf_id": "genbio-ai/genbio-pathfm",
        "input_size": 224,
        "embedding_dim": 4608,
        "pooling": "model_output_cls_direct",
        "mean": (0.697, 0.575, 0.728),
        "std": (0.188, 0.240, 0.187),
        "normalization": "GenBio PathFM model-card mean/std",
        "loader": "transformers_trust_remote_code",
        "resize_interpolation": "bilinear",
    },
}

# Right-anchored parser: specimen may contain hyphens, dots, and underscores.
PATCH_NAME_RE = re.compile(
    r"^(?P<specimen>.+)_(?P<image_id>\d+)_(?P<left>\d+)_(?P<top>\d+)"
    r"_d(?P<downsample>\d+)_(?P<label>[A-Za-z0-9]+)\.(?P<suffix>png|jpg|jpeg)$",
    flags=re.IGNORECASE,
)

ImageFile.LOAD_TRUNCATED_IMAGES = False


@dataclass(frozen=True)
class PatchRecord:
    patch_uid: str
    path: Path
    source_relpath: str
    filename: str
    specimen: str
    image_id: int
    left: int
    top: int
    downsample: int
    label: str
    label_int: int
    mtime_ns: int
    size_bytes: int


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv_list(text: str) -> List[str]:
    return [x.strip().upper() for x in str(text).split(",") if x.strip()]


def safe_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def atomic_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=safe_json_value),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def atomic_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez(
        tmp, **arrays
    )  # deliberately uncompressed: faster for large feature arrays
    os.replace(tmp, path)


class RunLock:
    """Advisory Unix lock. It prevents two writers targeting one model directory."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        try:
            import fcntl
        except ImportError as exc:
            raise SystemExit(
                "This script currently requires a Unix-like OS for fcntl locking."
            ) from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a+")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                f"Another ivygap_live_embeddings.py process holds {self.path}. "
                "Do not run two writers against the same --out/foundation directory."
            ) from exc
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(f"pid={os.getpid()}\nstarted_utc={now_utc()}\n")
        self.fh.flush()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fh is not None:
            try:
                import fcntl

                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            finally:
                self.fh.close()
        return False


def make_patch_uid(
    specimen: str, image_id: int, left: int, top: int, downsample: int
) -> str:
    # Natural, path-independent ID. The label is deliberately NOT included:
    # a label mismatch for the same physical tile must be visible rather than
    # silently producing two differently labelled embeddings.
    return f"ivy-gap::{specimen}::{image_id}::{left}::{top}::d{downsample}"


def parse_patch_path(
    path: Path, patch_dir: Path, st: os.stat_result
) -> Optional[PatchRecord]:
    m = PATCH_NAME_RE.match(path.name)
    if m is None:
        return None
    g = m.groupdict()
    label = str(g["label"]).upper()
    if label not in BRATS_LABEL_TO_INT:
        return None
    try:
        rel = str(path.relative_to(patch_dir))
    except ValueError:
        rel = path.name
    specimen = str(g["specimen"])
    image_id = int(g["image_id"])
    left = int(g["left"])
    top = int(g["top"])
    downsample = int(g["downsample"])
    return PatchRecord(
        patch_uid=make_patch_uid(specimen, image_id, left, top, downsample),
        path=path,
        source_relpath=rel,
        filename=path.name,
        specimen=specimen,
        image_id=image_id,
        left=left,
        top=top,
        downsample=downsample,
        label=label,
        label_int=int(BRATS_LABEL_TO_INT[label]),
        mtime_ns=int(st.st_mtime_ns),
        size_bytes=int(st.st_size),
    )


def iter_image_files(patch_dir: Path) -> Iterator[Path]:
    # The Ivy writer emits .png; jpg/jpeg are accepted for portability.
    for path in patch_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            yield path


def stable_file(
    path: Path, min_age_s: float, now_epoch: float
) -> Optional[os.stat_result]:
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size <= 0:
        return None
    if now_epoch - st.st_mtime < float(min_age_s):
        return None
    return st


def reservoir_add(
    bucket: List[PatchRecord],
    seen_count: int,
    candidate: PatchRecord,
    capacity: int,
    rng: random.Random,
) -> None:
    """Uniform reservoir sampling over all currently eligible candidates."""
    if capacity <= 0:
        return
    if len(bucket) < capacity:
        bucket.append(candidate)
        return
    slot = rng.randrange(seen_count)
    if slot < capacity:
        bucket[slot] = candidate


def scan_candidates(
    *,
    patch_dir: Path,
    classes: Sequence[str],
    embedded_uids: set[str],
    capacity_by_class: Dict[str, int],
    min_age_s: float,
    seed: int,
) -> Tuple[Dict[str, List[PatchRecord]], Dict[str, int], Dict[str, int]]:
    """Scan the live patch tree once and sample only non-committed stable files."""
    chosen: Dict[str, List[PatchRecord]] = {lab: [] for lab in classes}
    available: Dict[str, int] = {lab: 0 for lab in classes}
    stats = {
        "files_seen": 0,
        "parsed_supported": 0,
        "too_new_or_empty": 0,
        "unparseable_or_unsupported": 0,
        "already_committed": 0,
    }
    rng_by_class = {
        lab: random.Random(int(seed) + 1009 * i) for i, lab in enumerate(classes)
    }
    now_epoch = time.time()

    for path in iter_image_files(patch_dir):
        stats["files_seen"] += 1
        st = stable_file(path, min_age_s=min_age_s, now_epoch=now_epoch)
        if st is None:
            stats["too_new_or_empty"] += 1
            continue
        rec = parse_patch_path(path, patch_dir, st)
        if rec is None or rec.label not in chosen:
            stats["unparseable_or_unsupported"] += 1
            continue
        stats["parsed_supported"] += 1
        if rec.patch_uid in embedded_uids:
            stats["already_committed"] += 1
            continue
        available[rec.label] += 1
        reservoir_add(
            chosen[rec.label],
            available[rec.label],
            rec,
            capacity_by_class.get(rec.label, 0),
            rng_by_class[rec.label],
        )

    return chosen, available, stats


def read_reuse_index(
    index_path: Path,
    patch_dir: Path,
    classes: Sequence[str],
    embedded_uids: set[str],
    capacity_by_class: Dict[str, int],
    min_age_s: float,
    seed: int,
) -> Tuple[Dict[str, List[PatchRecord]], Dict[str, int], Dict[str, int]]:
    """Select from a prior embedded_index.csv to align two foundation models."""
    if not index_path.exists():
        raise FileNotFoundError(f"--reuse-index not found: {index_path}")

    chosen: Dict[str, List[PatchRecord]] = {lab: [] for lab in classes}
    available: Dict[str, int] = {lab: 0 for lab in classes}
    stats = {
        "index_rows_seen": 0,
        "index_rows_missing_source": 0,
        "index_rows_too_new_or_empty": 0,
        "index_rows_invalid": 0,
        "already_committed": 0,
    }
    rng_by_class = {
        lab: random.Random(int(seed) + 2003 * i) for i, lab in enumerate(classes)
    }
    now_epoch = time.time()

    with index_path.open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            stats["index_rows_seen"] += 1
            label = str(row.get("label", "")).upper()
            if label not in chosen:
                continue
            rel = row.get("source_relpath", "")
            source_path = patch_dir / rel
            st = stable_file(source_path, min_age_s=min_age_s, now_epoch=now_epoch)
            if st is None:
                if not source_path.exists():
                    stats["index_rows_missing_source"] += 1
                else:
                    stats["index_rows_too_new_or_empty"] += 1
                continue
            rec = parse_patch_path(source_path, patch_dir, st)
            if rec is None:
                stats["index_rows_invalid"] += 1
                continue
            if rec.patch_uid != row.get("patch_uid", rec.patch_uid):
                raise RuntimeError(
                    f"Reuse-index source identity mismatch for {source_path}. "
                    "Do not use an index from a different patch tree."
                )
            if rec.label != label:
                raise RuntimeError(
                    f"Reuse-index label mismatch for {source_path}: index={label}, filename={rec.label}."
                )
            if rec.patch_uid in embedded_uids:
                stats["already_committed"] += 1
                continue
            available[label] += 1
            reservoir_add(
                chosen[label],
                available[label],
                rec,
                capacity_by_class.get(label, 0),
                rng_by_class[label],
            )

    return chosen, available, stats


def interleave_by_class(
    chosen: Dict[str, List[PatchRecord]], classes: Sequence[str], seed: int
) -> List[PatchRecord]:
    """Shuffle within classes then round-robin for a balanced inference stream."""
    rng = random.Random(seed + 7919)
    queues: Dict[str, List[PatchRecord]] = {}
    for lab in classes:
        items = list(chosen.get(lab, []))
        rng.shuffle(items)
        queues[lab] = items

    out: List[PatchRecord] = []
    while any(queues[lab] for lab in classes):
        for lab in classes:
            if queues[lab]:
                out.append(queues[lab].pop())
    return out


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def speed_flags() -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def model_key_for(foundation: str) -> str:
    """Identity for raw pooled features produced by this raw-only extractor."""
    spec = FOUNDATIONS[foundation]
    return (
        f"{foundation}__ivy_fullpatch_resize{spec['input_size']}"
        f"__{spec['pooling']}__raw_pooled_features__v2"
    )


def static_embedding_config(foundation: str) -> Dict[str, Any]:
    spec = FOUNDATIONS[foundation]
    return {
        "schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "foundation": foundation,
        "hf_id": spec["hf_id"],
        "model_key": model_key_for(foundation),
        "embedding_dim": int(spec["embedding_dim"]),
        "input_mode": "full_ivy_patch_resize_directly_to_224_no_crop",
        "input_size": int(spec["input_size"]),
        "normalization": spec["normalization"],
        "pooling": spec["pooling"],
        "feature_postprocessing": "none; raw pooled model features",
        "brats_label_to_int": BRATS_LABEL_TO_INT,
        "supported_ivy_classes": list(DEFAULT_IVY_CLASSES),
    }


def ensure_embedding_config(model_root: Path, new_config: Dict[str, Any]) -> None:
    path = model_root / "embedding_config.json"
    if not path.exists():
        atomic_json(path, new_config)
        return
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read existing config: {path}") from exc

    must_match = [
        "schema_version",
        "foundation",
        "hf_id",
        "model_key",
        "embedding_dim",
        "input_mode",
        "input_size",
        "pooling",
        "feature_postprocessing",
        "brats_label_to_int",
    ]
    mismatch = {
        k: (old.get(k), new_config.get(k))
        for k in must_match
        if old.get(k) != new_config.get(k)
    }
    if mismatch:
        raise RuntimeError(
            f"Existing embedding directory {model_root} has incompatible configuration:\n"
            f"{json.dumps(mismatch, indent=2)}\n"
            "Use a different --out directory; do not mix incompatible embeddings."
        )


def connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=120.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            model_key TEXT NOT NULL,
            patch_uid TEXT NOT NULL,
            label TEXT NOT NULL,
            label_int INTEGER NOT NULL,
            filename TEXT NOT NULL,
            source_relpath TEXT NOT NULL,
            specimen TEXT NOT NULL,
            image_id INTEGER NOT NULL,
            left_px INTEGER NOT NULL,
            top_px INTEGER NOT NULL,
            downsample INTEGER NOT NULL,
            source_mtime_ns INTEGER NOT NULL,
            source_size_bytes INTEGER NOT NULL,
            block_relpath TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            committed_utc TEXT NOT NULL,
            PRIMARY KEY (model_key, patch_uid)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_model_label "
        "ON embeddings(model_key, label)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_embeddings_model_block "
        "ON embeddings(model_key, block_relpath)"
    )
    conn.commit()
    return conn


def db_existing_uids(conn: sqlite3.Connection, model_key: str) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT patch_uid FROM embeddings WHERE model_key = ?", (model_key,)
        )
    }


def db_counts(
    conn: sqlite3.Connection, model_key: str, classes: Sequence[str]
) -> Dict[str, int]:
    counts = {lab: 0 for lab in classes}
    for row in conn.execute(
        "SELECT label, COUNT(*) AS n FROM embeddings WHERE model_key = ? GROUP BY label",
        (model_key,),
    ):
        if str(row["label"]) in counts:
            counts[str(row["label"])] = int(row["n"])
    return counts


def db_insert_records(
    conn: sqlite3.Connection,
    model_key: str,
    block_relpath: str,
    records: Sequence[PatchRecord],
) -> None:
    rows = [
        (
            model_key,
            r.patch_uid,
            r.label,
            r.label_int,
            r.filename,
            r.source_relpath,
            r.specimen,
            r.image_id,
            r.left,
            r.top,
            r.downsample,
            r.mtime_ns,
            r.size_bytes,
            block_relpath,
            i,
            now_utc(),
        )
        for i, r in enumerate(records)
    ]
    if len({r.patch_uid for r in records}) != len(records):
        raise RuntimeError(
            f"Duplicate patch_uid inside block {block_relpath}; refusing to commit."
        )

    before = conn.total_changes
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(
            """
            INSERT OR IGNORE INTO embeddings(
                model_key, patch_uid, label, label_int, filename, source_relpath,
                specimen, image_id, left_px, top_px, downsample, source_mtime_ns,
                source_size_bytes, block_relpath, row_index, committed_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        added = conn.total_changes - before
        if added != len(rows):
            raise RuntimeError(
                f"SQLite found {len(rows) - added} duplicate patch IDs while committing {block_relpath}. "
                "Another process may have written to this directory; the transaction was rolled back."
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def reconcile_blocks(conn: sqlite3.Connection, model_root: Path, model_key: str) -> int:
    """Register blocks present on disk but absent from SQLite after an interrupted run."""
    blocks_dir = model_root / "blocks"
    if not blocks_dir.exists():
        return 0

    recovered = 0
    for path in sorted(blocks_dir.glob("*.npz")):
        rel = str(path.relative_to(model_root))
        n_registered = int(
            conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_key = ? AND block_relpath = ?",
                (model_key, rel),
            ).fetchone()[0]
        )
        try:
            with np.load(path, allow_pickle=False) as d:
                stored_key = str(d["model_key"].item())
                uids = d["patch_uid"].astype(str).tolist()
                labels = d["label"].astype(str).tolist()
                label_ints = d["label_int"].astype(np.int16).tolist()
                filenames = d["filename"].astype(str).tolist()
                relpaths = d["source_relpath"].astype(str).tolist()
                specimens = d["specimen"].astype(str).tolist()
                image_ids = d["image_id"].astype(np.int64).tolist()
                lefts = d["left"].astype(np.int64).tolist()
                tops = d["top"].astype(np.int64).tolist()
                downsamples = d["downsample"].astype(np.int16).tolist()
                mtimes = d["source_mtime_ns"].astype(np.int64).tolist()
                sizes = d["source_size_bytes"].astype(np.int64).tolist()
        except Exception as exc:
            raise RuntimeError(f"Cannot read embedding block {path}") from exc

        if stored_key != model_key:
            continue
        n = len(uids)
        if n_registered == n:
            continue
        if n_registered not in {0, n}:
            raise RuntimeError(
                f"Partial registry state for {path}: database has {n_registered}, block has {n}. "
                "Stop and inspect before continuing."
            )
        if not all(
            len(v) == n
            for v in (
                labels,
                label_ints,
                filenames,
                relpaths,
                specimens,
                image_ids,
                lefts,
                tops,
                downsamples,
                mtimes,
                sizes,
            )
        ):
            raise RuntimeError(f"Metadata length mismatch in {path}")

        records = [
            PatchRecord(
                patch_uid=uids[i],
                path=Path(filenames[i]),  # not used during reconciliation
                source_relpath=relpaths[i],
                filename=filenames[i],
                specimen=specimens[i],
                image_id=int(image_ids[i]),
                left=int(lefts[i]),
                top=int(tops[i]),
                downsample=int(downsamples[i]),
                label=labels[i],
                label_int=int(label_ints[i]),
                mtime_ns=int(mtimes[i]),
                size_bytes=int(sizes[i]),
            )
            for i in range(n)
        ]
        db_insert_records(conn, model_key, rel, records)
        recovered += n
        print(f"[reconcile] recovered {n:,} rows from {rel}", flush=True)
    return recovered


def export_index_and_counts(
    conn: sqlite3.Connection,
    model_root: Path,
    model_key: str,
    classes: Sequence[str],
) -> Dict[str, int]:
    """Write a flat, portable index without reading every NPZ again."""
    index_path = model_root / "embedded_index.csv"
    fields = [
        "model_key",
        "patch_uid",
        "label",
        "label_int",
        "filename",
        "source_relpath",
        "specimen",
        "image_id",
        "left",
        "top",
        "downsample",
        "source_mtime_ns",
        "source_size_bytes",
        "block_relpath",
        "row_index",
        "committed_utc",
    ]

    query = """
        SELECT model_key, patch_uid, label, label_int, filename, source_relpath,
               specimen, image_id, left_px, top_px, downsample, source_mtime_ns,
               source_size_bytes, block_relpath, row_index, committed_utc
        FROM embeddings
        WHERE model_key = ?
        ORDER BY label_int, specimen, image_id, left_px, top_px
    """

    def rows() -> Iterator[Dict[str, Any]]:
        for r in conn.execute(query, (model_key,)):
            yield {
                "model_key": r["model_key"],
                "patch_uid": r["patch_uid"],
                "label": r["label"],
                "label_int": r["label_int"],
                "filename": r["filename"],
                "source_relpath": r["source_relpath"],
                "specimen": r["specimen"],
                "image_id": r["image_id"],
                "left": r["left_px"],
                "top": r["top_px"],
                "downsample": r["downsample"],
                "source_mtime_ns": r["source_mtime_ns"],
                "source_size_bytes": r["source_size_bytes"],
                "block_relpath": r["block_relpath"],
                "row_index": r["row_index"],
                "committed_utc": r["committed_utc"],
            }

    atomic_csv(index_path, fields, rows())
    counts = db_counts(conn, model_key, classes)
    atomic_csv(
        model_root / "class_counts.csv",
        ["label", "label_int", "embedded_count"],
        (
            {
                "label": lab,
                "label_int": BRATS_LABEL_TO_INT[lab],
                "embedded_count": counts.get(lab, 0),
            }
            for lab in classes
        ),
    )
    return counts


def append_run_history(model_root: Path, item: Dict[str, Any]) -> None:
    path = model_root / "run_history.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, sort_keys=True, default=safe_json_value) + "\n")


def build_full_resize_transform(foundation: str, model: torch.nn.Module):
    spec = FOUNDATIONS[foundation]
    if foundation == "hoptimus1":
        mean, std = spec["mean"], spec["std"]
        interpolation = InterpolationMode.BICUBIC
    elif foundation == "genbiopathfm":
        # Official GenBio-PathFM preprocessing (GitHub README / HF model card):
        #   transforms.Resize((224, 224))   # torchvision default = BILINEAR
        #   transforms.ToTensor()
        #   transforms.Normalize(mean=(0.697, 0.575, 0.728), std=(0.188, 0.240, 0.187))
        # Unlike Virchow2/H-optimus-1, GenBio-PathFM's official recipe intentionally
        # uses bilinear resizing, not bicubic, so this script honors that here rather
        # than forcing BICUBIC for consistency with the other foundations.
        mean, std = spec["mean"], spec["std"]
        interpolation = InterpolationMode.BILINEAR
    else:
        if resolve_model_data_config is None:
            raise RuntimeError("Missing timm. Install with: pip install 'timm>=0.9.11'")
        try:
            data_cfg = resolve_model_data_config(model)
            mean = tuple(data_cfg.get("mean", (0.485, 0.456, 0.406)))
            std = tuple(data_cfg.get("std", (0.229, 0.224, 0.225)))
        except Exception as exc:
            raise RuntimeError(
                "Could not resolve Virchow2 preprocessing from timm model config."
            ) from exc
        interpolation = InterpolationMode.BICUBIC

    # Explicitly avoids timm's potential resize+centre-crop eval transform:
    # source image -> 224x224 full-field resize -> tensor -> normalize.
    try:
        resize = transforms.Resize(
            (int(spec["input_size"]), int(spec["input_size"])),
            interpolation=interpolation,
            antialias=True,
        )
    except TypeError:  # older torchvision
        resize = transforms.Resize(
            (int(spec["input_size"]), int(spec["input_size"])),
            interpolation=interpolation,
        )
    transform = transforms.Compose(
        [resize, transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return transform, tuple(float(v) for v in mean), tuple(float(v) for v in std)


def load_foundation(
    foundation: str,
    device: torch.device,
    compile_model: bool,
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
    spec = FOUNDATIONS[foundation]

    if foundation == "genbiopathfm":
        # GenBio-PathFM is loaded via HuggingFace transformers with
        # trust_remote_code=True -- NOT timm. It is a public model (no gated
        # access / huggingface-cli login needed), but requires an up-to-date
        # transformers install because the model class ships as remote code.
        # Reference: https://github.com/genbio-ai/genbio-pathfm
        #            https://huggingface.co/genbio-ai/genbio-pathfm
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "GenBio-PathFM requires the transformers library.\n"
                "Install/update with:  pip install -U transformers\n"
                "GenBio AI tested inference with transformers==4.57.1."
            ) from exc
        try:
            model = AutoModel.from_pretrained(spec["hf_id"], trust_remote_code=True)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {foundation} via transformers.AutoModel(trust_remote_code=True). "
                "Confirm `transformers` is up to date (pip install -U transformers)."
            ) from exc
    else:
        if timm is None:
            raise RuntimeError("Missing timm. Install with: pip install 'timm>=0.9.11'")
        try:
            if foundation == "virchow2":
                # Required by the official Virchow2 timm example.
                from timm.layers import SwiGLUPacked

                model = timm.create_model(
                    spec["hf_id"],
                    pretrained=True,
                    mlp_layer=SwiGLUPacked,
                    act_layer=torch.nn.SiLU,
                )
            elif foundation == "hoptimus1":
                # Official H-optimus-1 inference kwargs.
                model = timm.create_model(
                    spec["hf_id"],
                    pretrained=True,
                    init_values=1e-5,
                    dynamic_img_size=False,
                )
            else:  # defensive; argparse limits the choices
                raise KeyError(foundation)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {foundation} from Hugging Face. Confirm that you accepted "
                "the gated model terms and ran `huggingface-cli login` in this environment."
            ) from exc

    model = model.eval().to(device)
    if compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as exc:
            print(
                f"[warn] torch.compile failed; continuing uncompiled: {exc}",
                file=sys.stderr,
            )

    transform, mean, std = build_full_resize_transform(foundation, model)
    runtime = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "timm_version": getattr(timm, "__version__", "unknown"),
        "device": str(device),
        "mean": list(mean),
        "std": list(std),
        "python": platform.python_version(),
    }
    return model, transform, runtime


def _unwrap_model_output(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        # Current timm wrappers may use one of these names.
        for key in ("x_norm_clstoken", "pooled", "global_pool", "features"):
            value = out.get(key)
            if torch.is_tensor(value):
                return value
        for value in out.values():
            if torch.is_tensor(value):
                return value
    if isinstance(out, (tuple, list)):
        for value in out:
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(out):
        return out
    raise TypeError(
        f"Foundation model returned {type(out)} instead of a tensor-like output."
    )


def extract_features(
    model: torch.nn.Module,
    images: torch.Tensor,
    foundation: str,
) -> torch.Tensor:
    """Run foundation-specific pooling and return raw pooled model features."""
    raw_out = model(images)

    if foundation == "virchow2":
        # Official `model(images)` output is [B,261,1280].  The dict branch
        # retains compatibility with timm wrappers that expose normalized class
        # and patch tokens separately (as handled by the prior BraTS script).
        if isinstance(raw_out, dict):
            cls = raw_out.get("x_norm_clstoken")
            patches = raw_out.get("x_norm_patchtokens")
            if torch.is_tensor(cls) and torch.is_tensor(patches):
                feats = torch.cat([cls, patches.mean(dim=1)], dim=1)
            else:
                out = _unwrap_model_output(raw_out)
                if out.ndim == 2 and out.shape[1] == 2560:
                    feats = out
                else:
                    raise RuntimeError(
                        "Virchow2 dict output did not provide both x_norm_clstoken and "
                        f"x_norm_patchtokens; got tensor shape {tuple(out.shape)}."
                    )
        else:
            out = _unwrap_model_output(raw_out)
            if out.ndim == 3:
                if out.shape[1] <= 5:
                    raise RuntimeError(
                        f"Virchow2 returned too few tokens: shape={tuple(out.shape)}"
                    )
                cls = out[:, 0]
                patch_tokens = out[:, 5:]  # 1-4 are Virchow2 register tokens
                feats = torch.cat([cls, patch_tokens.mean(dim=1)], dim=1)
            elif out.ndim == 2 and out.shape[1] == 2560:
                # Some wrappers may expose the already-pooled official embedding.
                feats = out
            else:
                raise RuntimeError(
                    f"Unexpected Virchow2 output shape {tuple(out.shape)}; expected [B,261,1280] "
                    "or [B,2560]."
                )
    elif foundation == "hoptimus1":
        out = _unwrap_model_output(raw_out)
        if out.ndim != 2:
            out = out.flatten(1)
        feats = out
    elif foundation == "genbiopathfm":
        # Official GenBio-PathFM forward call returns the tile-level CLS
        # embedding directly: model(images) -> [B, 4608]. Patch tokens are
        # available via model.forward_with_patches(x), but this extractor
        # stores only the CLS embedding, consistent with the other foundations.
        # Reference: https://github.com/genbio-ai/genbio-pathfm
        out = _unwrap_model_output(raw_out)
        if out.ndim != 2:
            out = out.flatten(1)
        feats = out
    else:
        raise KeyError(foundation)

    expected = int(FOUNDATIONS[foundation]["embedding_dim"])
    if feats.ndim != 2 or feats.shape[1] != expected:
        raise RuntimeError(
            f"{foundation} embedding dimension mismatch: got {tuple(feats.shape)}, expected [B,{expected}]."
        )
    # Store raw pooled features. No row-wise L2 normalization or other
    # post-pooling feature transformation is performed in this extractor.
    return feats.float()


class LivePatchDataset(Dataset):
    """Image dataset that records decode failures instead of crashing a whole run."""

    def __init__(self, records: Sequence[PatchRecord], transform):
        self.records = list(records)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        try:
            with Image.open(record.path) as im:
                im.load()
                image = im.convert("RGB")
            tensor = self.transform(image)
            return record, tensor, ""
        except Exception as exc:
            return record, None, f"{type(exc).__name__}: {exc}"


def collate_live(batch):
    records: List[PatchRecord] = []
    tensors: List[torch.Tensor] = []
    failures: List[Tuple[PatchRecord, str]] = []
    for record, tensor, err in batch:
        if tensor is None:
            failures.append((record, err))
        else:
            records.append(record)
            tensors.append(tensor)
    if not tensors:
        return records, None, failures
    return records, torch.stack(tensors, dim=0), failures


def make_loader(
    records: Sequence[PatchRecord],
    transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    workers = max(0, int(num_workers))
    kwargs: Dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collate_live,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(LivePatchDataset(records, transform), **kwargs)


def append_failures(
    model_root: Path, failures: Sequence[Tuple[PatchRecord, str]]
) -> None:
    if not failures:
        return
    path = model_root / "decode_failures.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        fields = [
            "utc",
            "patch_uid",
            "label",
            "filename",
            "source_relpath",
            "specimen",
            "image_id",
            "left",
            "top",
            "downsample",
            "error",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for rec, err in failures:
            writer.writerow(
                {
                    "utc": now_utc(),
                    "patch_uid": rec.patch_uid,
                    "label": rec.label,
                    "filename": rec.filename,
                    "source_relpath": rec.source_relpath,
                    "specimen": rec.specimen,
                    "image_id": rec.image_id,
                    "left": rec.left,
                    "top": rec.top,
                    "downsample": rec.downsample,
                    "error": err,
                }
            )


def write_embedding_block(
    *,
    model_root: Path,
    model_key: str,
    foundation: str,
    run_id: str,
    block_index: int,
    records: Sequence[PatchRecord],
    X: np.ndarray,
) -> str:
    if not records:
        raise ValueError("Cannot write an empty embedding block.")
    if X.ndim != 2 or X.shape[0] != len(records):
        raise ValueError(
            f"Block feature shape {X.shape} does not match {len(records)} records."
        )
    expected = int(FOUNDATIONS[foundation]["embedding_dim"])
    if X.shape[1] != expected:
        raise ValueError(f"Block has dim={X.shape[1]}, expected {expected}.")

    name = f"block-{run_id}-{block_index:06d}.npz"
    path = model_root / "blocks" / name
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite existing embedding block: {path}")
    atomic_npz(
        path,
        schema_version=np.asarray(SCHEMA_VERSION),
        script_version=np.asarray(SCRIPT_VERSION),
        foundation=np.asarray(foundation),
        model_key=np.asarray(model_key),
        X=X,
        label=np.asarray([r.label for r in records], dtype=np.str_),
        label_int=np.asarray([r.label_int for r in records], dtype=np.int16),
        patch_uid=np.asarray([r.patch_uid for r in records], dtype=np.str_),
        filename=np.asarray([r.filename for r in records], dtype=np.str_),
        source_relpath=np.asarray([r.source_relpath for r in records], dtype=np.str_),
        specimen=np.asarray([r.specimen for r in records], dtype=np.str_),
        image_id=np.asarray([r.image_id for r in records], dtype=np.int64),
        left=np.asarray([r.left for r in records], dtype=np.int64),
        top=np.asarray([r.top for r in records], dtype=np.int64),
        downsample=np.asarray([r.downsample for r in records], dtype=np.int16),
        source_mtime_ns=np.asarray([r.mtime_ns for r in records], dtype=np.int64),
        source_size_bytes=np.asarray([r.size_bytes for r in records], dtype=np.int64),
    )
    return str(path.relative_to(model_root))


def amp_enabled_for(device: torch.device, no_amp: bool, amp_on_mps: bool) -> bool:
    if no_amp:
        return False
    if device.type == "cuda":
        return True
    if device.type == "mps":
        return bool(amp_on_mps)
    return False


def flush_buffers(
    *,
    conn: sqlite3.Connection,
    model_root: Path,
    model_key: str,
    foundation: str,
    run_id: str,
    block_index: int,
    records_buffer: List[PatchRecord],
    feats_buffer: List[np.ndarray],
) -> Tuple[int, int]:
    if not records_buffer:
        return block_index, 0
    X = np.concatenate(feats_buffer, axis=0)
    rel = write_embedding_block(
        model_root=model_root,
        model_key=model_key,
        foundation=foundation,
        run_id=run_id,
        block_index=block_index,
        records=records_buffer,
        X=X,
    )
    # Block is already atomic on disk. Only now mark it committed in SQLite.
    db_insert_records(conn, model_key, rel, records_buffer)
    print(
        f"[block] {rel}: {len(records_buffer):,} patches, X={tuple(X.shape)}",
        flush=True,
    )
    return block_index + 1, int(len(records_buffer))


def extract_selected(
    *,
    selected: Sequence[PatchRecord],
    need_by_class: Dict[str, int],
    classes: Sequence[str],
    model: torch.nn.Module,
    transform,
    device: torch.device,
    foundation: str,
    model_root: Path,
    conn: sqlite3.Connection,
    model_key: str,
    batch_size: int,
    num_workers: int,
    storage_dtype: str,
    no_amp: bool,
    amp_on_mps: bool,
    flush_every_batches: int,
) -> Dict[str, Any]:
    loader = make_loader(
        selected,
        transform,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    target = {lab: int(need_by_class.get(lab, 0)) for lab in classes}
    added = {lab: 0 for lab in classes}
    decode_failures = 0
    buffers_records: List[PatchRecord] = []
    buffers_feats: List[np.ndarray] = []
    # A unique run ID makes collision impossible with prior runs; numeric index is
    # only for order within this run.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-p{os.getpid()}"
    block_index = 0

    output_dtype = np.float16 if storage_dtype == "float16" else np.float32
    use_amp = amp_enabled_for(device, no_amp=no_amp, amp_on_mps=amp_on_mps)
    amp_dtype = torch.float16
    n_batches_since_flush = 0
    t0 = time.time()
    n_loaded = 0
    n_skipped_after_quota = 0

    for records, images, failures in loader:
        if failures:
            decode_failures += len(failures)
            append_failures(model_root, failures)
        if images is None:
            continue

        # The decode stage may have returned buffer candidates beyond a class quota.
        keep_input = [
            i
            for i, rec in enumerate(records)
            if added.get(rec.label, 0) < target.get(rec.label, 0)
        ]
        if not keep_input:
            n_skipped_after_quota += len(records)
            if all(added[lab] >= target[lab] for lab in classes):
                break
            continue
        if len(keep_input) != len(records):
            indices = torch.as_tensor(keep_input, dtype=torch.long)
            images = images.index_select(0, indices)
            records = [records[i] for i in keep_input]
            n_skipped_after_quota += len(records) - len(keep_input)

        n_loaded += len(records)
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            with (
                torch.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                )
                if use_amp
                else contextlib.nullcontext()
            ):
                feats = extract_features(model, images, foundation)

        feats_np = feats.detach().cpu().numpy()
        finite_rows = np.isfinite(feats_np).all(axis=1)
        if not finite_rows.all():
            bad = int((~finite_rows).sum())
            raise RuntimeError(
                f"{bad}/{len(records)} non-finite embeddings occurred. Re-run with --no-amp "
                "and a smaller --batch-size; no block from this batch was committed."
            )
        feats_np = feats_np.astype(output_dtype, copy=False)

        # Enforce exact per-class additions even where a final batch crosses a quota.
        chosen_idx: List[int] = []
        for i, rec in enumerate(records):
            if added[rec.label] < target[rec.label]:
                chosen_idx.append(i)
                added[rec.label] += 1
        if chosen_idx:
            ia = np.asarray(chosen_idx, dtype=np.int64)
            buffers_records.extend(records[i] for i in chosen_idx)
            buffers_feats.append(feats_np[ia])
            n_batches_since_flush += 1

        if n_batches_since_flush >= max(1, int(flush_every_batches)):
            block_index, _ = flush_buffers(
                conn=conn,
                model_root=model_root,
                model_key=model_key,
                foundation=foundation,
                run_id=run_id,
                block_index=block_index,
                records_buffer=buffers_records,
                feats_buffer=buffers_feats,
            )
            buffers_records, buffers_feats = [], []
            n_batches_since_flush = 0
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if all(added[lab] >= target[lab] for lab in classes):
            break

    if buffers_records:
        block_index, _ = flush_buffers(
            conn=conn,
            model_root=model_root,
            model_key=model_key,
            foundation=foundation,
            run_id=run_id,
            block_index=block_index,
            records_buffer=buffers_records,
            feats_buffer=buffers_feats,
        )

    elapsed = time.time() - t0
    return {
        "added_by_class": added,
        "decode_failures": decode_failures,
        "n_selected_candidates": len(selected),
        "n_loaded": n_loaded,
        "n_skipped_after_quota": n_skipped_after_quota,
        "elapsed_s": round(elapsed, 3),
        "patches_per_s": round(sum(added.values()) / max(elapsed, 1e-9), 3),
        "amp_enabled": use_amp,
        "storage_dtype": storage_dtype,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live, resumable Ivy GAP patch embedding extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--patch-dir",
        type=Path,
        required=True,
        help="Directory containing the live Ivy patch PNG files, normally <ivygap_out>/patches.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Parent output directory. Each foundation writes to <out>/<foundation>/. Use a fresh output root for this raw-only extractor; normalized legacy embeddings cannot be mixed here.",
    )
    parser.add_argument("--foundation", choices=sorted(FOUNDATIONS), default="virchow2")
    sample_mode = parser.add_mutually_exclusive_group()
    sample_mode.add_argument(
        "--add-per-class",
        type=int,
        default=5000,
        help="Add this many NEW patches for every requested class on this run.",
    )
    sample_mode.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Grow each requested class only until its cumulative committed count reaches this target.",
    )
    parser.add_argument("--classes", default=",".join(DEFAULT_IVY_CLASSES))
    parser.add_argument(
        "--min-file-age-s",
        type=float,
        default=120.0,
        help="Ignore files modified more recently than this, protecting against the concurrent writer.",
    )
    parser.add_argument(
        "--candidate-buffer-ratio",
        type=float,
        default=0.03,
        help="Sample this fraction of extra candidates per class to tolerate rare image decode failures.",
    )
    parser.add_argument(
        "--reuse-index",
        type=Path,
        default=None,
        help="Prior embedded_index.csv. Use to align a second foundation to exactly the same patch IDs.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA/MPS automatic mixed precision.",
    )
    parser.add_argument(
        "--amp-on-mps",
        action="store_true",
        help="Enable fp16 AMP on Apple MPS. Off by default because MPS may produce non-finite features.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Attempt torch.compile after model loading.",
    )
    parser.add_argument("--flush-every-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report the chosen candidates without loading a model.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Reconcile existing blocks, export the index/counts, print totals, and exit.",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not rewrite embedded_index.csv/class_counts.csv after this run.",
    )
    args = parser.parse_args()

    if args.add_per_class is not None and args.add_per_class < 0:
        parser.error("--add-per-class must be >= 0")
    if args.target_per_class is not None and args.target_per_class < 0:
        parser.error("--target-per-class must be >= 0")
    if args.min_file_age_s < 0:
        parser.error("--min-file-age-s must be >= 0")
    if args.candidate_buffer_ratio < 0:
        parser.error("--candidate-buffer-ratio must be >= 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    return args


def main() -> None:
    args = parse_args()
    speed_flags()

    patch_dir = args.patch_dir.expanduser().resolve()
    if not patch_dir.is_dir():
        raise SystemExit(
            f"--patch-dir does not exist or is not a directory: {patch_dir}"
        )

    classes = parse_csv_list(args.classes)
    if not classes:
        raise SystemExit("--classes is empty")
    unsupported = [lab for lab in classes if lab not in DEFAULT_IVY_CLASSES]
    if unsupported:
        raise SystemExit(
            f"This Ivy GAP writer emits only {DEFAULT_IVY_CLASSES}; unsupported requested classes: {unsupported}"
        )

    config = static_embedding_config(args.foundation)
    model_key = str(config["model_key"])
    model_root = args.out.expanduser().resolve() / args.foundation
    model_root.mkdir(parents=True, exist_ok=True)

    with RunLock(model_root / ".writer.lock"):
        ensure_embedding_config(model_root, config)
        conn = connect_db(model_root / "registry.sqlite")
        try:
            recovered = reconcile_blocks(conn, model_root, model_key)
            counts_before = db_counts(conn, model_key, classes)

            if args.audit_only:
                counts_after = counts_before
                if not args.no_index:
                    counts_after = export_index_and_counts(
                        conn, model_root, model_key, classes
                    )
                print(
                    json.dumps(
                        {
                            "foundation": args.foundation,
                            "model_root": str(model_root),
                            "recovered_rows": recovered,
                            "embedded_by_class": counts_after,
                        },
                        indent=2,
                    )
                )
                return

            if args.target_per_class is not None:
                need_by_class = {
                    lab: max(
                        0, int(args.target_per_class) - int(counts_before.get(lab, 0))
                    )
                    for lab in classes
                }
                sampling_mode = "target_per_class"
            else:
                need_by_class = {lab: int(args.add_per_class) for lab in classes}
                sampling_mode = "add_per_class"

            # Draw a few extra candidates so rare failures do not short the requested quota.
            capacity_by_class = {
                lab: (
                    0
                    if need_by_class[lab] <= 0
                    else int(
                        min(
                            need_by_class[lab]
                            + max(
                                4,
                                math.ceil(
                                    need_by_class[lab] * args.candidate_buffer_ratio
                                ),
                            ),
                            need_by_class[lab] + 10_000,
                        )
                    )
                )
                for lab in classes
            }
            embedded_uids = db_existing_uids(conn, model_key)

            if args.reuse_index is None:
                chosen, available, scan_stats = scan_candidates(
                    patch_dir=patch_dir,
                    classes=classes,
                    embedded_uids=embedded_uids,
                    capacity_by_class=capacity_by_class,
                    min_age_s=args.min_file_age_s,
                    seed=args.seed,
                )
                source_mode = "live_patch_directory_scan"
            else:
                chosen, available, scan_stats = read_reuse_index(
                    index_path=args.reuse_index.expanduser().resolve(),
                    patch_dir=patch_dir,
                    classes=classes,
                    embedded_uids=embedded_uids,
                    capacity_by_class=capacity_by_class,
                    min_age_s=args.min_file_age_s,
                    seed=args.seed,
                )
                source_mode = "reuse_existing_index"

            selected = interleave_by_class(chosen, classes, args.seed)
            selected_by_class = {lab: len(chosen[lab]) for lab in classes}

            planning = {
                "utc": now_utc(),
                "script_version": SCRIPT_VERSION,
                "foundation": args.foundation,
                "model_key": model_key,
                "patch_dir": str(patch_dir),
                "model_root": str(model_root),
                "sampling_mode": sampling_mode,
                "source_mode": source_mode,
                "recovered_rows": recovered,
                "embedded_before_by_class": counts_before,
                "need_by_class": need_by_class,
                "stable_unembedded_available_by_class": available,
                "selected_candidate_by_class": selected_by_class,
                "scan_stats": scan_stats,
                "min_file_age_s": args.min_file_age_s,
            }
            print(json.dumps(planning, indent=2), flush=True)

            if args.dry_run:
                history = {**planning, "dry_run": True}
                append_run_history(model_root, history)
                if not args.no_index:
                    export_index_and_counts(conn, model_root, model_key, classes)
                return

            if not selected:
                print(
                    "[done] No candidates selected. The current target may already be met, or stable files are not yet available."
                )
                history = {**planning, "result": "no_candidates"}
                append_run_history(model_root, history)
                if not args.no_index:
                    export_index_and_counts(conn, model_root, model_key, classes)
                return

            device = (
                auto_device() if args.device == "auto" else torch.device(args.device)
            )
            if device.type == "cuda" and not torch.cuda.is_available():
                raise SystemExit(
                    "--device cuda requested, but torch.cuda.is_available() is false."
                )
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise SystemExit("--device mps requested, but MPS is unavailable.")
            print(
                f"[runtime] device={device} foundation={args.foundation} "
                f"candidates={len(selected):,} batch_size={args.batch_size} "
                f"dtype={args.dtype} feature_postprocessing=none(raw)",
                flush=True,
            )

            model, transform, runtime = load_foundation(
                args.foundation, device, args.compile
            )
            runtime_path = model_root / "runtime_last_loaded.json"
            atomic_json(
                runtime_path, {**runtime, "utc": now_utc(), "model_key": model_key}
            )

            result = extract_selected(
                selected=selected,
                need_by_class=need_by_class,
                classes=classes,
                model=model,
                transform=transform,
                device=device,
                foundation=args.foundation,
                model_root=model_root,
                conn=conn,
                model_key=model_key,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                storage_dtype=args.dtype,
                no_amp=args.no_amp,
                amp_on_mps=args.amp_on_mps,
                flush_every_batches=args.flush_every_batches,
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            counts_after = db_counts(conn, model_key, classes)
            if not args.no_index:
                counts_after = export_index_and_counts(
                    conn, model_root, model_key, classes
                )

            history = {
                **planning,
                "runtime": runtime,
                "result": result,
                "embedded_after_by_class": counts_after,
                "completed_utc": now_utc(),
            }
            append_run_history(model_root, history)

            print("\n[done] committed embedding counts by class:", flush=True)
            for lab in classes:
                print(
                    f"  {lab} (BraTS {BRATS_LABEL_TO_INT[lab]}): "
                    f"{counts_before[lab]:,} -> {counts_after[lab]:,} "
                    f"(+{result['added_by_class'][lab]:,})",
                    flush=True,
                )
            print(f"  blocks: {model_root / 'blocks'}")
            print(f"  registry: {model_root / 'registry.sqlite'}")
            print(f"  index: {model_root / 'embedded_index.csv'}")
        finally:
            conn.close()


if __name__ == "__main__":
    main()

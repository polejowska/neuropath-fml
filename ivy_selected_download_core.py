from __future__ import annotations

from pathlib import Path
import csv
import gzip
import hashlib
import io
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

SELECTION_MANIFEST = Path("reproducibility/ivy_selected_manifest.csv.gz")
OUTPUT_ROOT = Path("data/ivygap_selected")
PATCH_DIR = OUTPUT_ROOT / "patches"

ALLEN_API_BASE = "https://api.brain-map.org/api/v2"
TILE_SIZE = 512

DOWNLOAD_WORKERS = 8
REQUEST_TIMEOUT_S = 120.0
REQUEST_RETRIES = 4
OVERWRITE_EXISTING = False
VERIFY_RGB_HASH = True

LABEL_BY_INT = {
    0: "CT",
    2: "IC",
    4: "MP",
    5: "NC",
    7: "PN",
}

FOUNDATION_ORDER_COLUMNS = {
    "virchow2": "virchow2_order",
    "hoptimus1": "hoptimus1_order",
    "genbiopathfm": "genbiopathfm_order",
}

UID_RE = re.compile(
    r"^ivy-gap::(?P<specimen>.+)::(?P<image_id>\d+)::"
    r"(?P<left>\d+)::(?P<top>\d+)::d(?P<downsample>\d+)$"
)

REQUIRED_COLUMNS = {
    "patch_uid",
    "label_int",
    *FOUNDATION_ORDER_COLUMNS.values(),
}


def open_text_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def parse_patch_uid(uid: str):
    m = UID_RE.match(str(uid))
    if m is None:
        raise ValueError(f"Unexpected Ivy patch_uid format: {uid!r}")
    g = m.groupdict()
    return {
        "specimen": g["specimen"],
        "image_id": int(g["image_id"]),
        "left": int(g["left"]),
        "top": int(g["top"]),
        "downsample": int(g["downsample"]),
    }


def load_selection_manifest(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen Ivy selection manifest: {path}")

    with open_text_csv(path) as fh:
        reader = csv.DictReader(fh)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - fields

        if missing:
            raise RuntimeError(f"{path}: missing required columns {sorted(missing)}")

        rows = list(reader)

    if not rows:
        raise RuntimeError(f"{path}: manifest is empty")

    seen = set()

    for row in rows:
        uid = str(row["patch_uid"])

        if uid in seen:
            raise RuntimeError(f"{path}: duplicate patch_uid {uid}")

        seen.add(uid)

        label_int = int(row["label_int"])

        if label_int not in LABEL_BY_INT:
            raise RuntimeError(f"{path}: unsupported label_int={label_int} for {uid}")

        parse_patch_uid(uid)

    return rows


def canonical_filename(row) -> str:
    meta = parse_patch_uid(row["patch_uid"])
    label = LABEL_BY_INT[int(row["label_int"])]

    return (
        f"{meta['specimen']}_{meta['image_id']}_{meta['left']}_{meta['top']}_"
        f"d{meta['downsample']}_{label}.png"
    )


def rgb_pixel_sha256(image_or_path) -> str:
    if isinstance(image_or_path, Image.Image):
        image = image_or_path.convert("RGB")
    else:
        with Image.open(image_or_path) as src:
            image = src.convert("RGB")

    arr = np.asarray(image, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(f"RGB:{image.width}x{image.height}\n".encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def validate_manifest_orders(rows):
    for foundation, column in FOUNDATION_ORDER_COLUMNS.items():
        order = []

        for row in rows:
            value = str(row.get(column, "")).strip()

            if value != "":
                order.append(int(value))

        if not order:
            raise RuntimeError(f"No selected rows found for foundation={foundation!r}")

        if sorted(order) != list(range(len(order))):
            raise RuntimeError(f"{foundation}: {column} is not a dense 0..N-1 ordering")

        counts = {}

        for row in rows:
            value = str(row.get(column, "")).strip()

            if value == "":
                continue

            label_int = int(row["label_int"])
            counts[label_int] = counts.get(label_int, 0) + 1

        print(
            f"[manifest] {foundation}: {len(order):,} rows "
            f"by_class={dict(sorted(counts.items()))}"
        )


def verify_existing(path: Path, row):
    with Image.open(path) as src:
        if src.size != (TILE_SIZE, TILE_SIZE):
            raise RuntimeError(
                f"{path}: expected {TILE_SIZE}x{TILE_SIZE}, got {src.size}"
            )

    expected = str(row.get("rgb_sha256", "")).strip()

    if VERIFY_RGB_HASH:
        if not expected:
            raise RuntimeError(f"Missing rgb_sha256 for {row['patch_uid']}")

        actual = rgb_pixel_sha256(path)

        if actual != expected:
            raise RuntimeError(
                f"RGB hash mismatch for {path}: expected={expected} actual={actual}"
            )


def download_one(row):
    import requests

    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    dest = PATCH_DIR / canonical_filename(row)

    if dest.is_file() and not OVERWRITE_EXISTING:
        verify_existing(dest, row)
        return "existing"

    meta = parse_patch_uid(row["patch_uid"])
    params = {
        "left": meta["left"],
        "top": meta["top"],
        "width": TILE_SIZE,
        "height": TILE_SIZE,
        "downsample": meta["downsample"],
    }
    url = f"{ALLEN_API_BASE}/image_download/{meta['image_id']}"

    last_error = None

    for attempt in range(REQUEST_RETRIES):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_S,
            )
            response.raise_for_status()

            with Image.open(io.BytesIO(response.content)) as src:
                image = src.convert("RGB")

            if image.size != (TILE_SIZE, TILE_SIZE):
                raise RuntimeError(
                    f"{row['patch_uid']}: returned {image.size}, "
                    f"expected {(TILE_SIZE, TILE_SIZE)}"
                )

            expected = str(row.get("rgb_sha256", "")).strip()

            if VERIFY_RGB_HASH:
                if not expected:
                    raise RuntimeError(f"Missing rgb_sha256 for {row['patch_uid']}")

                actual = rgb_pixel_sha256(image)

                if actual != expected:
                    raise RuntimeError(
                        f"{row['patch_uid']}: RGB mismatch "
                        f"expected={expected} actual={actual}"
                    )

            tmp = dest.with_name(f".{dest.name}.tmp.{os.getpid()}")
            image.save(tmp, format="PNG")
            os.replace(tmp, dest)
            return "downloaded"

        except Exception as exc:
            last_error = exc

            if attempt + 1 < REQUEST_RETRIES:
                time.sleep(min(2**attempt, 8))

    raise RuntimeError(
        f"Failed to download {row['patch_uid']} after "
        f"{REQUEST_RETRIES} attempts: {last_error}"
    )


def download_all(rows):
    PATCH_DIR.mkdir(parents=True, exist_ok=True)

    status_counts = {
        "downloaded": 0,
        "existing": 0,
    }
    failures = []

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download_one, row): row for row in rows}

        for i, future in enumerate(as_completed(futures), start=1):
            row = futures[future]

            try:
                status = future.result()
                status_counts[status] += 1
            except Exception as exc:
                failures.append(f"{row['patch_uid']}: {exc}")

            if i % 250 == 0 or i == len(rows):
                print(
                    f"[download] {i:,}/{len(rows):,} "
                    f"downloaded={status_counts['downloaded']:,} "
                    f"existing={status_counts['existing']:,} "
                    f"failed={len(failures):,}"
                )

    if failures:
        raise RuntimeError(
            f"{len(failures):,} Ivy downloads failed\n" + "\n".join(failures[:10])
        )


def main():
    rows = load_selection_manifest(SELECTION_MANIFEST)
    validate_manifest_orders(rows)

    print(f"[Ivy] unique physical patches required: {len(rows):,}")
    print(f"[Ivy] output directory: {PATCH_DIR.resolve()}")

    download_all(rows)

    print("[Ivy] exact selected patch set is ready")


if __name__ == "__main__":
    main()

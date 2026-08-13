from __future__ import annotations

import argparse
import csv
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

BRATS_LABEL_TO_INT = {
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

PATCH_NAME_RE = re.compile(
    r"^(?P<specimen>.+)_(?P<image_id>\d+)_(?P<left>\d+)_(?P<top>\d+)"
    r"_d(?P<downsample>\d+)_(?P<label>[A-Za-z0-9]+)\.(?P<suffix>png|jpg|jpeg)$",
    flags=re.IGNORECASE,
)


def make_patch_uid(specimen, image_id, left, top, downsample):
    return f"ivy-gap::{specimen}::{image_id}::{left}::{top}::d{downsample}"


# ==== end copied section ====


def parse_one(path: Path):
    m = PATCH_NAME_RE.match(path.name)
    if m is None:
        return None
    g = m.groupdict()
    label = str(g["label"]).upper()
    if label not in BRATS_LABEL_TO_INT:
        return None
    specimen = str(g["specimen"])
    image_id = int(g["image_id"])
    left = int(g["left"])
    top = int(g["top"])
    downsample = int(g["downsample"])
    return {
        "path": path,
        "filename": path.name,
        "specimen": specimen,
        "image_id": image_id,
        "left": left,
        "top": top,
        "downsample": downsample,
        "label": label,
        "label_int": BRATS_LABEL_TO_INT[label],
        "patch_uid": make_patch_uid(specimen, image_id, left, top, downsample),
    }


def scan_all_sources(source_dirs):
    """Returns dict: label -> list of parsed-record dicts (across ALL sources)."""
    by_class = defaultdict(list)
    seen_uids = set()
    dupes = 0
    for src in source_dirs:
        src = Path(src).expanduser().resolve()
        if not src.is_dir():
            raise SystemExit(f"--patch-dir does not exist or is not a directory: {src}")
        for path in src.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {
                ".png",
                ".jpg",
                ".jpeg",
            }:
                continue
            rec = parse_one(path)
            if rec is None:
                continue
            if rec["patch_uid"] in seen_uids:
                dupes += 1
                continue  # same logical tile appears in more than one source dir -- keep first
            seen_uids.add(rec["patch_uid"])
            by_class[rec["label"]].append(rec)
    if dupes:
        print(
            f"[note] {dupes} duplicate patch_uid(s) found across your --patch-dir sources "
            f"(same specimen/tile appearing in more than one directory) -- kept the first "
            f"occurrence only, skipped the rest."
        )
    return by_class


def round_robin_select(records, quota, seed):
    """Diversity-first selection: group by specimen, shuffle specimen order and
    within-specimen order, then take one patch per specimen per round until
    quota is met or every record has been used."""
    by_specimen = defaultdict(list)
    for r in records:
        by_specimen[r["specimen"]].append(r)

    rng = random.Random(seed)
    specimens = list(by_specimen.keys())
    rng.shuffle(specimens)
    for sp in specimens:
        rng.shuffle(by_specimen[sp])

    selected = []
    pointers = {sp: 0 for sp in specimens}
    n_specimens_touched = set()
    while len(selected) < quota:
        made_progress = False
        for sp in specimens:
            if len(selected) >= quota:
                break
            i = pointers[sp]
            if i < len(by_specimen[sp]):
                selected.append(by_specimen[sp][i])
                pointers[sp] = i + 1
                n_specimens_touched.add(sp)
                made_progress = True
        if not made_progress:
            break  # every specimen's list exhausted
    return selected, len(specimens), len(n_specimens_touched)


def write_symlink_pool(records, merged_patches_dir: Path):
    merged_patches_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        target = merged_patches_dir / r["filename"]
        if target.exists() or target.is_symlink():
            if target.resolve() == r["path"].resolve():
                continue  # already linked correctly (e.g. reused across classes -- shouldn't
                # happen since labels are disjoint, but harmless if it does)
            raise RuntimeError(
                f"Filename collision in merged pool: {target} already exists and points "
                f"elsewhere. Two different source patches produced the same filename -- "
                f"this shouldn't happen given the naming scheme; investigate before continuing."
            )
        target.symlink_to(r["path"])


def write_reuse_index_csv(records, merged_patches_dir: Path, csv_path: Path):
    fields = [
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
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            rel = str(
                (merged_patches_dir / r["filename"]).relative_to(merged_patches_dir)
            )
            w.writerow(
                {
                    "patch_uid": r["patch_uid"],
                    "label": r["label"],
                    "label_int": r["label_int"],
                    "filename": r["filename"],
                    "source_relpath": rel,
                    "specimen": r["specimen"],
                    "image_id": r["image_id"],
                    "left": r["left"],
                    "top": r["top"],
                    "downsample": r["downsample"],
                }
            )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--patch-dir",
        action="append",
        required=True,
        help="A patches directory to include. Repeat for multiple sources, "
        "e.g. --patch-dir ./ivygap_patches_balanced/patches "
        "--patch-dir ./ivygap_patches_priority/patches",
    )
    ap.add_argument(
        "--merged-out",
        required=True,
        help="Directory to create the merged symlink pool + reuse-index CSVs in. "
        "Should be separate from any existing patch/embedding directories.",
    )
    ap.add_argument(
        "--priority-classes",
        default="MP,PN,CT,NC,IC",
        help="Comma-separated class order. Selection (and the printed run "
        "commands) follow this order left to right.",
    )
    ap.add_argument(
        "--quota",
        action="append",
        default=[],
        help="Per-class quota override as NAME=N, repeatable, e.g. --quota MP=999999. "
        "Use a very large number to mean 'take everything available.'",
    )
    ap.add_argument(
        "--default-quota",
        type=int,
        default=3000,
        help="Quota for any priority class not given an explicit --quota override.",
    )
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument(
        "--out-embeddings",
        default="./ivygap_embeddings_new_raw",
        help="The --out value to print in the suggested ivygap_live_embeddings.py "
        "commands (not created by this script -- just used for the printed "
        "command text).",
    )
    ap.add_argument(
        "--foundation",
        default="virchow2",
        help="The --foundation value to print in the suggested commands.",
    )
    a = ap.parse_args()

    priority_classes = [
        c.strip().upper() for c in a.priority_classes.split(",") if c.strip()
    ]
    unsupported = [c for c in priority_classes if c not in DEFAULT_IVY_CLASSES]
    if unsupported:
        raise SystemExit(
            f"Unsupported class(es) {unsupported}; the Ivy writer only emits "
            f"{DEFAULT_IVY_CLASSES}"
        )

    quota_overrides = {}
    for entry in a.quota:
        if "=" not in entry:
            raise SystemExit(f"--quota must be NAME=N, got: {entry!r}")
        name, n = entry.split("=", 1)
        quota_overrides[name.strip().upper()] = int(n)

    print(f"Scanning {len(a.patch_dir)} source director(ies)...")
    by_class = scan_all_sources(a.patch_dir)
    for c in priority_classes:
        print(f"  {c}: {len(by_class.get(c, []))} patches found across all sources")

    merged_out = Path(a.merged_out).expanduser().resolve()
    merged_patches_dir = merged_out / "patches"
    merged_patches_dir.mkdir(parents=True, exist_ok=True)

    print(
        "\nSelecting diversity-first (round-robin across specimens), in priority order:"
    )
    commands = []
    summary_rows = []
    for c in priority_classes:
        records = by_class.get(c, [])
        quota = quota_overrides.get(c, a.default_quota)
        selected, n_specimens_total, n_specimens_touched = round_robin_select(
            records, quota, seed=a.seed + BRATS_LABEL_TO_INT[c]
        )

        write_symlink_pool(selected, merged_patches_dir)
        csv_path = merged_out / f"reuse_index_{c}.csv"
        write_reuse_index_csv(selected, merged_patches_dir, csv_path)

        print(
            f"  {c}: selected {len(selected)}/{len(records)} available "
            f"(touched {n_specimens_touched}/{n_specimens_total} specimens) "
            f"-> {csv_path}"
        )
        summary_rows.append(
            (c, len(selected), len(records), n_specimens_touched, n_specimens_total)
        )

        if selected:
            commands.append(
                f"python ivygap_live_embeddings.py \\\n"
                f"    --patch-dir {merged_patches_dir} \\\n"
                f"    --out {a.out_embeddings} \\\n"
                f"    --foundation {a.foundation} \\\n"
                f"    --classes {c} \\\n"
                f"    --reuse-index {csv_path} \\\n"
                f"    --target-per-class {len(selected)}"
            )

    print("\n" + "=" * 70)
    print("Run these IN ORDER (MP first, etc.) -- if you run out of time partway,")
    print("everything up to that point is already safely committed:")
    print("=" * 70)
    for cmd in commands:
        print(f"\n{cmd}")

    summary_path = merged_out / "selection_summary.csv"
    with summary_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["class", "selected", "available", "specimens_touched", "specimens_total"]
        )
        w.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()

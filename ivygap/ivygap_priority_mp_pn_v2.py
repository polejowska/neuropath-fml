from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter

BASE = "https://api.brain-map.org/api/v2"
TILE = 512

BRATS_CLASS_TO_INT = {
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
MAPPABLE_CLASSES = ["CT", "MP", "NC", "PN", "IC"]
MANIFEST_HEADER = [
    "filepath",
    "label",
    "label_int",
    "specimen",
    "image_id",
    "left",
    "top",
    "downsample",
    "coverage",
    "purity_frac",
]

# =====================  HTTP layer (shared pooled session)  ================

_SESSION = requests.Session()
_ADAPTER = HTTPAdapter(pool_connections=128, pool_maxsize=128, max_retries=0)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)


def _get(url, params=None, binary=False, retries=4):
    last = None
    for i in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.content if binary else r.json()
        except Exception as e:
            last = e
            time.sleep(min(2**i, 8))
    raise RuntimeError(f"GET failed after {retries} tries: {url} :: {last}")


def rma(model, criteria="", include="", num_rows=5000, start_row=0):
    crit = f"model::{model}"
    if criteria:
        crit += f",rma::criteria,{criteria}"
    if include:
        crit += f",rma::include,{include}"
    data = _get(
        f"{BASE}/data/query.json",
        params={"criteria": crit, "num_rows": num_rows, "start_row": start_row},
    )
    if not data.get("success", False):
        raise RuntimeError(data.get("msg", "RMA query failed"))
    return data["msg"]


def list_specimens(product_abbrev):
    recs = rma("Specimen", criteria=f"products[abbreviation$eq'{product_abbrev}']")
    return sorted(
        {r["external_specimen_name"] for r in recs if r.get("external_specimen_name")}
    )


def section_images_for_specimen(specimen, treatment):
    recs = rma(
        "SectionDataSet",
        criteria=(
            f"specimen[external_specimen_name$eq'{specimen}'],"
            f"treatments[name$eq'{treatment}']"
        ),
        include="sub_images",
    )
    out = []
    for ds in recs:
        for si in ds.get("sub_images", []):
            out.append({k: si.get(k) for k in ("id", "width", "height", "x", "y")})
    return out


def download_tile(image_id, left, top, downsample, view=None):
    params = {
        "left": left,
        "top": top,
        "width": TILE,
        "height": TILE,
        "downsample": downsample,
    }
    if view:
        params["view"] = view
    raw = _get(f"{BASE}/image_download/{image_id}", params=params, binary=True)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def tile_origins(width, height, downsample, tile=TILE):
    step = tile * (2**downsample)
    xs = range(0, max(width - step + 1, 1), step) if width >= step else [0]
    ys = range(0, max(height - step + 1, 1), step) if height >= step else [0]
    return [(int(x), int(y)) for y in ys for x in xs]


def load_palette_from_file(path):
    resolved = json.load(open(path))
    pal = np.array([r["rgb"] for r in resolved], dtype=np.float32)
    labs = [r["brats"] for r in resolved]
    return pal, labs, resolved


def _nearest(rgb, palette, tol):
    d = np.linalg.norm(rgb[:, None, :] - palette[None, :, :], axis=2)
    idx = d.argmin(axis=1)
    idx[d.min(axis=1) > tol] = -1
    return idx


def label_from_annotation(
    ann, palette, labels, purity=0.70, min_coverage=0.50, color_tol=30.0
):
    if palette.shape[0] == 0:
        return None, {"reason": "empty_palette"}
    arr = np.asarray(ann, dtype=np.float32).reshape(-1, 3)
    idx = _nearest(arr, palette, color_tol)
    annotated = idx >= 0
    coverage = float(annotated.mean())
    if coverage < min_coverage:
        return None, {"reason": "low_coverage", "coverage": coverage}
    lab_arr = np.array([labels[i] if i >= 0 else "" for i in idx], dtype=object)
    vals, counts = np.unique(lab_arr[annotated].astype(str), return_counts=True)
    top = int(counts.argmax())
    frac = float(counts[top] / counts.sum())
    if frac < purity:
        return None, {"reason": "mixed", "coverage": coverage, "top_frac": frac}
    return str(vals[top]), {"coverage": coverage, "top_frac": frac}


def tissue_ok(
    he,
    bg_frac_max=0.60,
    lowsat_frac_max=0.95,
    eosin_low_frac_max=0.80,
    eosin_low_val=50,
):
    rgb = np.asarray(he, dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    white = np.all(flat > 230, axis=1)
    black = np.all(flat < 25, axis=1)
    if (white | black).mean() > bg_frac_max:
        return False
    try:
        from skimage.color import rgb2hed, rgb2hsv

        hsv = rgb2hsv(rgb)
        if (hsv[..., 1] < 0.05).mean() > lowsat_frac_max:
            return False
        eo = rgb2hed(rgb)[..., 1]
        eo = (eo - eo.min()) / (eo.ptp() + 1e-8) * 255.0
        if (eo < eosin_low_val).mean() > eosin_low_frac_max:
            return False
    except Exception:
        pass
    return True


# =====================  probe + full-scan worker functions  ================


def _probe_one_tile(
    iid, L, T, downsample, purity, min_coverage, color_tol, palette, labels
):
    """Annotation-only download + classify. No H&E fetch -- this is the cheap check."""
    try:
        ann = download_tile(iid, L, T, downsample, view="tumor_feature_annotation")
        label, st = label_from_annotation(
            ann, palette, labels, purity, min_coverage, color_tol
        )
        return label
    except Exception:
        return None  # treat probe errors as "no hit" -- full scan will retry/log properly if it matters


def _process_one_tile(
    iid,
    L,
    T,
    downsample,
    purity,
    min_coverage,
    color_tol,
    palette,
    labels,
    sp_counts,
    per_class_cap,
):
    """Same as ivygap_sample_balanced.py: full classify + conditional H&E fetch."""
    try:
        ann = download_tile(iid, L, T, downsample, view="tumor_feature_annotation")
        label, st = label_from_annotation(
            ann, palette, labels, purity, min_coverage, color_tol
        )
        if label is None:
            return (iid, L, T, None, st, None)
        if sp_counts.get(label, 0) >= per_class_cap:
            return (iid, L, T, None, {"reason": "class_already_capped"}, None)
        he = download_tile(iid, L, T, downsample)
        if not tissue_ok(he):
            return (iid, L, T, None, {"reason": "tissue_qc_failed"}, None)
        return (iid, L, T, label, st, he)
    except Exception as e:
        return (iid, L, T, "__ERROR__", f"{type(e).__name__}: {e}", None)


# =====================  skip-set / resume-state loaders  ===================


def load_manifest_specimens(path):
    if not path or not os.path.exists(path):
        return set(), None
    seen_order, seen_set = [], set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sp = row.get("specimen")
            if sp and sp not in seen_set:
                seen_set.add(sp)
                seen_order.append(sp)
    return seen_set, (seen_order[-1] if seen_order else None)


def load_completed_specimens(path):
    """
    specimens_completed.txt is the authoritative "fully attempted" record from a
    prior run of ivygap_sample_balanced.py -- unlike manifest.csv, it includes
    specimens that were fully scanned but yielded ZERO patches (e.g. no
    annotation data), which would otherwise be invisible and get re-queued.
    """
    if not path or not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def derive_completed_path(manifest_path, explicit_override):
    """Prefer an explicit --*-completed path; otherwise look for
    specimens_completed.txt alongside the given manifest.csv."""
    if explicit_override:
        return explicit_override
    if manifest_path:
        candidate = os.path.join(
            os.path.dirname(manifest_path) or ".", "specimens_completed.txt"
        )
        if os.path.exists(candidate):
            return candidate
    return None


def load_own_progress(manifest_path, completed_path):
    done_tiles, counts, completed = {}, {}, set()
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as fh:
            for row in csv.DictReader(fh):
                sp = row["specimen"]
                key = (row["image_id"], row["left"], row["top"])
                done_tiles.setdefault(sp, set()).add(key)
                counts.setdefault(sp, {}).setdefault(row["label"], 0)
                counts[sp][row["label"]] += 1
    if os.path.exists(completed_path):
        with open(completed_path) as fh:
            completed = {ln.strip() for ln in fh if ln.strip()}
    return done_tiles, counts, completed


# =====================  driver  =============================================


@dataclass
class Cfg:
    out_dir: str
    old_manifest: str | None
    balanced_manifest: str | None
    old_completed: str | None
    balanced_completed: str | None
    palette_file: str
    product: str = "Gbm"
    treatment: str = "H&E"
    downsample: int = 0
    purity: float = 0.70
    min_coverage: float = 0.50
    color_tol: float = 30.0
    per_class_cap: int = 1000
    priority_classes: tuple = ("MP", "PN")
    probe_tiles: int = 40
    workers: int = 12
    min_specimen: str | None = None
    skip_specimens: str | None = None
    force_include: str | None = None
    specimen_allowlist: str | None = (
        None  # path to file, one specimen/line: restrict queue to only these
    )
    max_tiles_per_specimen: int | None = None
    save_images: bool = True


def run(cfg: Cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    img_dir = os.path.join(cfg.out_dir, "patches")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.out_dir, "manifest.csv")
    completed_path = os.path.join(cfg.out_dir, "specimens_completed.txt")
    failed_path = os.path.join(cfg.out_dir, "failed_specimens.txt")
    no_priority_path = os.path.join(cfg.out_dir, "no_priority_found.txt")

    print(f"[palette] loading {cfg.palette_file}")
    palette, labels, resolved = load_palette_from_file(cfg.palette_file)
    json.dump(
        resolved,
        open(os.path.join(cfg.out_dir, "resolved_palette.json"), "w"),
        indent=2,
    )
    found_classes = sorted(set(labels))
    print(f"loaded {len(labels)} colors -> classes {found_classes}")
    missing = [c for c in MAPPABLE_CLASSES if c not in found_classes]
    if missing:
        raise SystemExit(
            f"Palette is missing expected class(es) {missing} -- wrong file?"
        )
    for pc in cfg.priority_classes:
        if pc not in found_classes:
            raise SystemExit(
                f"Priority class '{pc}' isn't in the loaded palette at all."
            )

    # ---- combined skip set from BOTH prior runs ----
    old_skip, old_last = load_manifest_specimens(cfg.old_manifest)
    bal_skip, bal_last = load_manifest_specimens(cfg.balanced_manifest)

    old_completed_path = derive_completed_path(cfg.old_manifest, cfg.old_completed)
    bal_completed_path = derive_completed_path(
        cfg.balanced_manifest, cfg.balanced_completed
    )
    old_completed_set = load_completed_specimens(old_completed_path)
    bal_completed_set = load_completed_specimens(bal_completed_path)

    combined_skip = old_skip | bal_skip | old_completed_set | bal_completed_set
    print(
        f"[skip set] manifest rows: {len(old_skip)} (old) + {len(bal_skip)} (balanced). "
        f"specimens_completed.txt: {len(old_completed_set)} (old, {old_completed_path or 'not found'}) + "
        f"{len(bal_completed_set)} (balanced, {bal_completed_path or 'not found'}). "
        f"{len(combined_skip)} unique combined -- this now correctly includes zero-yield "
        f"specimens that were fully attempted but never wrote a manifest row."
    )
    if old_last:
        print(f"  (old manifest's last-seen specimen, possibly incomplete: {old_last})")
    if bal_last:
        print(
            f"  (balanced manifest's last-seen specimen, possibly incomplete: {bal_last})"
        )

    if cfg.force_include:
        forced = {s.strip() for s in cfg.force_include.split(",") if s.strip()}
        combined_skip -= forced
        print(f"[--force-include] removed {len(forced)} specimen(s) from the skip set.")

    # ---- this script's own resume state ----
    done_tiles, counts, completed = load_own_progress(manifest_path, completed_path)
    if completed:
        print(
            f"[resume] {len(completed)} specimens already fully completed in this run's own output."
        )

    # ---- specimen queue ----
    all_specimens = list_specimens(cfg.product)
    queue = [
        sp for sp in all_specimens if sp not in combined_skip and sp not in completed
    ]

    if cfg.min_specimen:
        n0 = len(queue)
        queue = [sp for sp in queue if sp >= cfg.min_specimen]
        print(
            f"[--min-specimen {cfg.min_specimen}] dropped {n0 - len(queue)} earlier specimens."
        )
    if cfg.skip_specimens:
        skip_explicit = {s.strip() for s in cfg.skip_specimens.split(",") if s.strip()}
        n0 = len(queue)
        queue = [sp for sp in queue if sp not in skip_explicit]
        print(f"[--skip-specimens] excluded {n0 - len(queue)} named specimen(s).")
    if cfg.specimen_allowlist:
        with open(cfg.specimen_allowlist) as fh:
            allowlist = {ln.strip() for ln in fh if ln.strip()}
        n0 = len(queue)
        queue = [sp for sp in queue if sp in allowlist]
        print(
            f"[--specimen-allowlist {cfg.specimen_allowlist}] restricted queue from {n0} "
            f"to {len(queue)} specimens confirmed (via metadata) to contain a priority class."
        )

    print(
        f"{len(all_specimens)} total specimens for product {cfg.product}; "
        f"{len(combined_skip)} skipped (prior runs), {len(completed)} skipped (already done here); "
        f"{len(queue)} to process now."
    )

    manifest_is_new = not os.path.exists(manifest_path)
    fh = open(manifest_path, "a", newline="")
    w = csv.writer(fh)
    if manifest_is_new:
        w.writerow(MANIFEST_HEADER)
        fh.flush()

    completed_fh = open(completed_path, "a")
    failed_fh = open(failed_path, "a")
    no_priority_fh = open(no_priority_path, "a")
    probe_log_path = os.path.join(cfg.out_dir, "probe_hits_log.tsv")
    probe_log_fh = open(probe_log_path, "a")
    global_probe_hits = Counter({c: 0 for c in cfg.priority_classes})

    def cap_reached(sp_counts):
        return all(sp_counts.get(c, 0) >= cfg.per_class_cap for c in MAPPABLE_CLASSES)

    try:
        for sp in queue:
            sp_counts = counts.setdefault(sp, {})
            sp_done = done_tiles.setdefault(sp, set())

            try:
                images = section_images_for_specimen(sp, cfg.treatment)
                rng = random.Random(hash(("specimen", sp)) & 0xFFFFFFFF)
                rng.shuffle(images)
                print(
                    f"[{sp}] starting: {len(images)} section image(s) to probe",
                    flush=True,
                )

                n_scanned_total = 0
                t_start = time.time()
                any_image_qualified = False

                with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
                    for si in images:
                        if cap_reached(sp_counts):
                            break
                        if (
                            cfg.max_tiles_per_specimen
                            and n_scanned_total >= cfg.max_tiles_per_specimen
                        ):
                            print(
                                f"  [{sp}] hit --max-tiles-per-specimen "
                                f"({cfg.max_tiles_per_specimen}); stopping.",
                                flush=True,
                            )
                            break

                        iid, W, H = si["id"], int(si["width"]), int(si["height"])
                        img_rng = random.Random(hash(("image", sp, iid)) & 0xFFFFFFFF)
                        origins = tile_origins(W, H, cfg.downsample)
                        img_rng.shuffle(origins)

                        # ---- cheap probe: annotation-only, no H&E ----
                        probe_set = origins[: cfg.probe_tiles]
                        probe_futs = [
                            executor.submit(
                                _probe_one_tile,
                                iid,
                                L,
                                T,
                                cfg.downsample,
                                cfg.purity,
                                cfg.min_coverage,
                                cfg.color_tol,
                                palette,
                                labels,
                            )
                            for (L, T) in probe_set
                        ]
                        # Diagnostic: tally EVERY priority-class hit in the probe sample
                        # (not just "did we find one"), so we can tell definitively whether
                        # a specific class (e.g. MP) is ever showing up in probes at all,
                        # versus being found-then-lost downstream.
                        probe_hit_counts = {c: 0 for c in cfg.priority_classes}
                        for f in as_completed(probe_futs):
                            r = f.result()
                            if r in probe_hit_counts:
                                probe_hit_counts[r] += 1
                        n_scanned_total += len(probe_set)

                        for c, n in probe_hit_counts.items():
                            if n > 0:
                                global_probe_hits[c] += n
                        found_priority_in_probe = any(
                            n > 0 for n in probe_hit_counts.values()
                        )

                        if not found_priority_in_probe:
                            continue  # skip full scan of this image entirely

                        any_image_qualified = True
                        hit_summary = {
                            c: n for c, n in probe_hit_counts.items() if n > 0
                        }
                        print(
                            f"  [{sp}] image {iid}: probe hits {hit_summary} out of "
                            f"{len(probe_set)} sampled -> full scan "
                            f"({len(origins)} tile positions)",
                            flush=True,
                        )
                        probe_log_fh.write(f"{sp}\t{iid}\t{hit_summary}\n")
                        probe_log_fh.flush()

                        # ---- full multi-class scan of this qualifying image ----
                        remaining = [
                            (iid, L, T)
                            for (L, T) in origins
                            if (str(iid), str(L), str(T)) not in sp_done
                        ]
                        idx = 0
                        chunk_size = max(cfg.workers * 4, cfg.workers)
                        while idx < len(remaining) and not cap_reached(sp_counts):
                            if (
                                cfg.max_tiles_per_specimen
                                and n_scanned_total >= cfg.max_tiles_per_specimen
                            ):
                                break
                            chunk = remaining[idx : idx + chunk_size]
                            idx += chunk_size
                            futures = {
                                executor.submit(
                                    _process_one_tile,
                                    tiid,
                                    L,
                                    T,
                                    cfg.downsample,
                                    cfg.purity,
                                    cfg.min_coverage,
                                    cfg.color_tol,
                                    palette,
                                    labels,
                                    sp_counts,
                                    cfg.per_class_cap,
                                ): (tiid, L, T)
                                for (tiid, L, T) in chunk
                            }
                            for fut in as_completed(futures):
                                tiid, L, T = futures[fut]
                                n_scanned_total += 1
                                _, _, _, label, st, he = fut.result()
                                tile_key = (str(tiid), str(L), str(T))

                                if n_scanned_total % 200 == 0:
                                    elapsed = time.time() - t_start
                                    rate = (
                                        n_scanned_total / elapsed if elapsed > 0 else 0
                                    )
                                    print(
                                        f"  [{sp}] scanned_total={n_scanned_total} "
                                        f"kept={sp_counts} ({rate:.1f} tiles/s, "
                                        f"{elapsed:.0f}s elapsed)",
                                        flush=True,
                                    )

                                if label == "__ERROR__":
                                    print(
                                        f"  ! tile {tiid}@{L},{T} ({sp}): {st}",
                                        file=sys.stderr,
                                    )
                                    continue
                                if label is None:
                                    continue
                                if sp_counts.get(label, 0) >= cfg.per_class_cap:
                                    continue

                                fn = (
                                    f"{sp}_{tiid}_{L}_{T}_d{cfg.downsample}_{label}.png"
                                )
                                fp = os.path.join(img_dir, fn)
                                if cfg.save_images:
                                    he.save(fp)
                                w.writerow(
                                    [
                                        fp,
                                        label,
                                        BRATS_CLASS_TO_INT[label],
                                        sp,
                                        tiid,
                                        L,
                                        T,
                                        cfg.downsample,
                                        f"{st.get('coverage', 0):.3f}",
                                        f"{st.get('top_frac', 0):.3f}",
                                    ]
                                )
                                fh.flush()
                                sp_counts[label] = sp_counts.get(label, 0) + 1
                                sp_done.add(tile_key)

                if not any_image_qualified:
                    print(
                        f"[{sp}] done: no image passed the MP/PN probe -- 0 patches saved.",
                        flush=True,
                    )
                    no_priority_fh.write(sp + "\n")
                    no_priority_fh.flush()
                else:
                    got_priority = any(
                        sp_counts.get(c, 0) > 0 for c in cfg.priority_classes
                    )
                    print(
                        f"[{sp}] done. counts={sp_counts} "
                        f"(priority classes present: {got_priority})",
                        flush=True,
                    )
                    if not got_priority:
                        # Rare edge case: probe hit but full scan somehow never confirmed it
                        # (e.g. that exact tile got rejected on H&E tissue QC). Still keep
                        # whatever else was captured -- not worth discarding real patches
                        # over one unlucky tile -- but flag it so you can look closer if needed.
                        no_priority_fh.write(
                            sp + "\t(probe positive but 0 kept -- check)\n"
                        )
                        no_priority_fh.flush()

                completed_fh.write(sp + "\n")
                completed_fh.flush()

            except Exception as e:
                msg = f"{sp}\t{type(e).__name__}: {e}"
                print(f"  !! SPECIMEN FAILED, skipping for now: {msg}", file=sys.stderr)
                failed_fh.write(msg + "\n")
                failed_fh.flush()
                continue
    finally:
        fh.close()
        completed_fh.close()
        failed_fh.close()
        no_priority_fh.close()
        probe_log_fh.close()

    print(
        f"\n[diagnostic] total probe hits this run, by priority class: {dict(global_probe_hits)}"
    )
    for c in cfg.priority_classes:
        if global_probe_hits.get(c, 0) == 0:
            print(
                f"  --> '{c}' was NEVER hit by any probe this entire run. This points to "
                f"either genuine absence of '{c}' in the specimens processed so far, or "
                f"the probe sample ({cfg.probe_tiles} tiles/image) being too small to catch "
                f"it given how rare it is -- consider rerunning remaining specimens with a "
                f"larger --probe-tiles to distinguish between the two."
            )
    print(f"See {probe_log_path} for the full per-image hit log.")
    print(
        "\nDONE with this pass. Rerun the same command to pick up any "
        "specimens listed in failed_specimens.txt or left mid-way."
    )


# =====================  CLI  ==============================================


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--old-manifest",
        default=None,
        help="manifest.csv from the first (unbounded) run -- used only to build the skip set.",
    )
    ap.add_argument(
        "--balanced-manifest",
        default=None,
        help="manifest.csv from the balanced/capped run -- used only to build the skip set.",
    )
    ap.add_argument(
        "--old-completed",
        default=None,
        help="specimens_completed.txt from the old run, if you have one (the "
        "original ivygap_prepare.py didn't write one, so this is usually N/A). "
        "Auto-detected alongside --old-manifest if not given.",
    )
    ap.add_argument(
        "--balanced-completed",
        default=None,
        help="specimens_completed.txt from the balanced run -- this is the "
        "authoritative record of specimens fully attempted there, INCLUDING "
        "zero-yield ones invisible in manifest.csv. Auto-detected alongside "
        "--balanced-manifest if not given (looked for in the same directory).",
    )
    ap.add_argument(
        "--palette-file",
        required=True,
        help="Known-good resolved_palette.json (required -- this script always "
        "loads it directly, no graph_id discovery).",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--product", default="Gbm")
    ap.add_argument("--treatment", default="H&E")
    ap.add_argument("--downsample", type=int, default=0)
    ap.add_argument("--purity", type=float, default=0.70)
    ap.add_argument("--min-coverage", type=float, default=0.50)
    ap.add_argument("--color-tol", type=float, default=30.0)
    ap.add_argument("--per-class-cap", type=int, default=1000)
    ap.add_argument(
        "--priority-classes",
        default="MP,PN",
        help="Comma-separated classes that must appear (via probe) before an "
        "image gets a full scan. Default: MP,PN",
    )
    ap.add_argument(
        "--probe-tiles",
        type=int,
        default=150,
        help="Number of tiles to sample per image for the cheap priority-class "
        "probe before deciding whether to fully scan it. Given how rare MP "
        "(~0.8%%) and PN (~1.4%%) are, a small sample can miss a class that's "
        "genuinely present just by chance -- e.g. 40 tiles misses a truly-"
        "present MP region ~73%% of the time (Poisson approx). 150 brings "
        "that down to ~30%%; raise further (300+) if you have time to spare.",
    )
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-specimen", default=None)
    ap.add_argument("--skip-specimens", default=None)
    ap.add_argument(
        "--force-include",
        default=None,
        help="Comma-separated specimen names to pull back out of the skip set "
        "even if they appear in --old-manifest/--balanced-manifest.",
    )
    ap.add_argument(
        "--specimen-allowlist",
        default=None,
        help="Path to a text file (one specimen per line) restricting the queue "
        "to ONLY these specimens -- e.g. specimens_with_priority.txt from "
        "ivygap_metadata_prefilter.py, so metadata-negative specimens are "
        "skipped without any image API calls at all.",
    )
    ap.add_argument("--max-tiles-per-specimen", type=int, default=None)
    ap.add_argument("--no-save-images", action="store_true")
    a = ap.parse_args()
    run(
        Cfg(
            out_dir=a.out,
            old_manifest=a.old_manifest,
            balanced_manifest=a.balanced_manifest,
            old_completed=a.old_completed,
            balanced_completed=a.balanced_completed,
            palette_file=a.palette_file,
            product=a.product,
            treatment=a.treatment,
            downsample=a.downsample,
            purity=a.purity,
            min_coverage=a.min_coverage,
            color_tol=a.color_tol,
            per_class_cap=a.per_class_cap,
            priority_classes=tuple(
                s.strip() for s in a.priority_classes.split(",") if s.strip()
            ),
            probe_tiles=a.probe_tiles,
            workers=a.workers,
            min_specimen=a.min_specimen,
            skip_specimens=a.skip_specimens,
            force_include=a.force_include,
            specimen_allowlist=a.specimen_allowlist,
            max_tiles_per_specimen=a.max_tiles_per_specimen,
            save_images=not a.no_save_images,
        )
    )


if __name__ == "__main__":
    main()

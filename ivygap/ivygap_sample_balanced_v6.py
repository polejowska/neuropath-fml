from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

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

# =====================  same core logic as ivygap_prepare.py  ==============


def map_structure_to_brats(name: str, acronym: str) -> str | None:
    n = (name or "").lower()
    a = acronym or ""
    if "pseudopalis" in n or a in ("CTpan", "CTpnn"):
        return "PN"
    if "microvascular" in n or a == "CTmvp":
        return "MP"
    if "perinecrotic" in n or a == "CTpnz":
        return None
    if "necrosis" in n or "necrotic" in n or a == "CTne":
        return "NC"
    if "cellular tumor" in n or a == "CT":
        return "CT"
    if "infiltrating tumor" in n or a == "IT":
        return "IC"
    if "leading edge" in n or a == "LE":
        return None
    if "hyperplastic" in n or "blood vessel" in n:
        return None
    return None


import requests
from requests.adapters import HTTPAdapter

# Shared session across all worker threads: reuses TCP/TLS connections
# (keep-alive) instead of paying connection-setup cost on every single tile
# request. pool_maxsize should be >= --workers so concurrent requests don't
# queue up waiting for a free connection.
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


GBM_ACRONYMS = ["CT", "CTmvp", "CTpan", "CTne", "IT", "LE", "CTpnz", "CTpnn"]


def find_graph_id():
    """
    Vote across ALL GBM-specific acronyms rather than trusting the first hit
    for a single one. 'CT' alone is generic enough that it can nondeterministically
    match a record in an unrelated ontology graph (RMA result ordering is not
    guaranteed stable across calls) -- this bit us: one run resolved graph_id=10
    with only 3 (wrong) structures, another run correctly resolved the real Ivy
    GBM graph with dozens of structures. The real Ivy GBM graph is the only one
    that will contain acronyms like CTmvp/CTpan/CTne/IT/LE/CTpnz/CTpnn together,
    so it should win the vote by a landslide.
    """
    from collections import Counter

    votes = Counter()
    per_acronym_graphs = {}
    for ac in GBM_ACRONYMS:
        try:
            hits = rma("Structure", criteria=f"[acronym$eq'{ac}']")
            gids = {int(r["graph_id"]) for r in hits if r.get("graph_id")}
            per_acronym_graphs[ac] = gids
            votes.update(gids)
        except Exception:
            pass
    if not votes:
        return None
    winner, n_votes = votes.most_common(1)[0]
    if n_votes < len(GBM_ACRONYMS) // 2:
        print(
            f"[warn] find_graph_id: winning graph_id={winner} only matched "
            f"{n_votes}/{len(GBM_ACRONYMS)} acronyms -- result may be unreliable. "
            f"Per-acronym graph_ids seen: {per_acronym_graphs}",
            file=sys.stderr,
        )
    return winner


def fetch_structures(graph_id):
    out = []
    for s in rma("Structure", criteria=f"[graph_id$eq{graph_id}]"):
        hx = s.get("color_hex_triplet")
        if hx and len(hx) == 6:
            out.append(
                (
                    s.get("acronym"),
                    s.get("name"),
                    (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)),
                )
            )
    return out


def load_palette_from_file(path):
    """
    Load a previously-verified resolved_palette.json directly (same format
    build_palette_auto()/run() writes out) instead of re-deriving it via
    find_graph_id()/fetch_structures(). This sidesteps the graph_id
    nondeterminism entirely -- if you already have a known-good palette
    (e.g. ivygap_patches_all/resolved_palette.json, confirmed correct with
    the full CT/MP/PN/IC/NC structure set), there's no reason to re-risk
    landing on the wrong ontology graph on every run.
    """
    resolved = json.load(open(path))
    pal = np.array([r["rgb"] for r in resolved], dtype=np.float32)
    labs = [r["brats"] for r in resolved]
    return pal, labs, resolved


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
    """
    Runs in a worker thread. Touches sp_counts read-only (a plain dict; under
    the GIL a simple `.get()` read is safe enough for this best-effort check --
    we don't need perfect atomicity, just to avoid the common case of
    needlessly downloading a full H&E tile for a class that's already full).
    Returns (iid, L, T, label_or_None, stats_or_errmsg, he_image_or_None).
    label == "__ERROR__" signals an exception occurred; stats holds the message.
    """
    try:
        ann = download_tile(iid, L, T, downsample, view="tumor_feature_annotation")
        label, st = label_from_annotation(
            ann, palette, labels, purity, min_coverage, color_tol
        )
        if label is None:
            return (iid, L, T, None, st, None)
        if sp_counts.get(label, 0) >= per_class_cap:
            # Already full -- skip the (larger, slower) H&E download entirely.
            return (iid, L, T, None, {"reason": "class_already_capped"}, None)
        he = download_tile(iid, L, T, downsample)
        if not tissue_ok(he):
            return (iid, L, T, None, {"reason": "tissue_qc_failed"}, None)
        return (iid, L, T, label, st, he)
    except Exception as e:
        return (iid, L, T, "__ERROR__", f"{type(e).__name__}: {e}", None)


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


def build_palette_auto(structures):
    pal, labs, resolved = [], [], []
    for acr, name, rgb in structures:
        lab = map_structure_to_brats(name, acr)
        if lab is None:
            continue
        pal.append(rgb)
        labs.append(lab)
        resolved.append({"rgb": list(rgb), "acronym": acr, "name": name, "brats": lab})
    return np.array(pal, dtype=np.float32), labs, resolved


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


# =====================  NEW: old-manifest skip-set  =========================


def load_old_manifest_specimens(path):
    """Returns (skip_set, last_specimen_seen_in_file_order)."""
    if not path or not os.path.exists(path):
        return set(), None
    seen_order = []
    seen_set = set()
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            sp = row.get("specimen")
            if sp and sp not in seen_set:
                seen_set.add(sp)
                seen_order.append(sp)
    last = seen_order[-1] if seen_order else None
    return seen_set, last


# =====================  NEW: own-manifest resume state  =====================


def load_own_progress(manifest_path, completed_path):
    """
    Returns:
      done_tiles: dict specimen -> set of (image_id, left, top) already written
      counts:     dict specimen -> dict label -> count
      completed:  set of specimens already fully finished (skip outright)
    """
    done_tiles = {}
    counts = {}
    completed = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, newline="") as fh:
            r = csv.DictReader(fh)
            for row in r:
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
    old_manifest: str
    product: str = "Gbm"
    treatment: str = "H&E"
    downsample: int = 0
    purity: float = 0.70
    min_coverage: float = 0.50
    color_tol: float = 30.0
    per_class_cap: int = 1000
    seed: int = 0
    redo_last_specimen: bool = False
    max_tiles_per_specimen: int | None = None  # safety valve, optional
    save_images: bool = True
    palette_file: str | None = (
        None  # if set, load palette from this JSON instead of re-deriving it
    )
    workers: int = 12  # concurrent tile-download threads
    min_specimen: str | None = (
        None  # only queue specimens sorting >= this name (alphabetical)
    )
    skip_specimens: str | None = (
        None  # comma-separated explicit specimen names to exclude
    )


def run(cfg: Cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    img_dir = os.path.join(cfg.out_dir, "patches")
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.out_dir, "manifest.csv")
    completed_path = os.path.join(cfg.out_dir, "specimens_completed.txt")
    failed_path = os.path.join(cfg.out_dir, "failed_specimens.txt")

    # ---- palette: prefer a known-good file over re-deriving via the API ----
    if cfg.palette_file:
        print(
            f"[palette] loading known-good palette from {cfg.palette_file} "
            f"(skipping find_graph_id()/fetch_structures() entirely)"
        )
        palette, labels, resolved = load_palette_from_file(cfg.palette_file)
        gid = None
    else:
        gid = find_graph_id()
        if gid is None:
            raise SystemExit("Could not auto-locate GBM structure graph_id.")
        structures = fetch_structures(gid)
        palette, labels, resolved = build_palette_auto(structures)

    if palette.shape[0] == 0:
        raise SystemExit("Palette is empty (no mappable structures).")
    json.dump(
        resolved,
        open(os.path.join(cfg.out_dir, "resolved_palette.json"), "w"),
        indent=2,
    )
    found_classes = sorted(set(labels))
    print(
        f"graph_id={gid}: auto-mapped {len(labels)} colors -> classes {found_classes}"
    )
    missing = [c for c in MAPPABLE_CLASSES if c not in found_classes]
    if missing:
        raise SystemExit(
            f"Sanity check failed: resolved palette is missing expected class(es) "
            f"{missing} (only found {found_classes}, {len(labels)} colors total). "
            + (
                f"This almost certainly means find_graph_id() resolved the wrong "
                f"ontology graph (a known nondeterminism in the Allen RMA API for "
                f"generic acronyms). Check resolved_palette.json in --out, or rerun -- "
                f"the vote-based graph_id resolution should pick the correct graph, but "
                f"if it's still wrong, pass --palette-file pointing at a known-good "
                f"resolved_palette.json instead."
                if not cfg.palette_file
                else f"The --palette-file you passed itself does not cover all expected "
                f"classes -- double check it's the right file."
            )
        )

    # ---- old-run skip set ----
    old_skip, old_last = load_old_manifest_specimens(cfg.old_manifest)
    if old_last is not None:
        print(
            f"[old manifest] {len(old_skip)} specimens already attempted in "
            f"'{cfg.old_manifest}'. Last one seen (possibly incomplete due to "
            f"the crash): {old_last}"
        )
        if cfg.redo_last_specimen and old_last in old_skip:
            old_skip.discard(old_last)
            print(f"  --redo-last-specimen set: '{old_last}' will be reprocessed here.")
        else:
            print(
                f"  Default: '{old_last}' is still skipped. Pass --redo-last-specimen "
                f"to force it back into the queue."
            )

    # ---- this script's own resume state ----
    done_tiles, counts, completed = load_own_progress(manifest_path, completed_path)
    if completed:
        print(
            f"[resume] {len(completed)} specimens already fully completed in this run's own output."
        )

    # ---- specimen queue ----
    all_specimens = list_specimens(cfg.product)
    queue = [sp for sp in all_specimens if sp not in old_skip and sp not in completed]

    n_before_bound = len(queue)
    if cfg.min_specimen:
        queue = [sp for sp in queue if sp >= cfg.min_specimen]
        print(
            f"[--min-specimen {cfg.min_specimen}] dropped {n_before_bound - len(queue)} "
            f"specimens that sort before it; {len(queue)} remain."
        )

    if cfg.skip_specimens:
        skip_explicit = {s.strip() for s in cfg.skip_specimens.split(",") if s.strip()}
        n_before_explicit = len(queue)
        queue = [sp for sp in queue if sp not in skip_explicit]
        print(
            f"[--skip-specimens] excluded {n_before_explicit - len(queue)} of "
            f"{len(skip_explicit)} named specimen(s); {len(queue)} remain."
        )

    print(
        f"{len(all_specimens)} total specimens for product {cfg.product}; "
        f"{len(old_skip)} skipped (old run), {len(completed)} skipped (already done here); "
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

    try:
        for sp in queue:
            sp_counts = counts.setdefault(sp, {})
            sp_done = done_tiles.setdefault(sp, set())

            def cap_reached():
                return all(
                    sp_counts.get(c, 0) >= cfg.per_class_cap for c in MAPPABLE_CLASSES
                )

            try:
                images = section_images_for_specimen(sp, cfg.treatment)
                rng = random.Random(hash(("specimen", sp)) & 0xFFFFFFFF)
                rng.shuffle(images)

                # Flatten every (image_id, left, top) across the whole specimen into
                # one list, then shuffle it as a whole. This is what actually enables
                # concurrency: instead of finishing one image before starting the
                # next, we can have workers pulling from many different images/regions
                # at once, which also gives better spatial diversity than shuffling
                # within each image separately.
                flat = []
                for si in images:
                    iid, W, H = si["id"], int(si["width"]), int(si["height"])
                    for L, T in tile_origins(W, H, cfg.downsample):
                        flat.append((iid, L, T))
                rng.shuffle(flat)
                flat = [
                    (iid, L, T)
                    for (iid, L, T) in flat
                    if (str(iid), str(L), str(T)) not in sp_done
                ]

                print(
                    f"[{sp}] starting: {len(images)} section image(s), "
                    f"{len(flat)} candidate tile positions to scan "
                    f"(already have {sum(sp_counts.values())} from a prior partial run), "
                    f"{cfg.workers} concurrent workers",
                    flush=True,
                )

                n_scanned = 0
                t_start = time.time()
                idx = 0
                chunk_size = max(cfg.workers * 4, cfg.workers)

                with ThreadPoolExecutor(max_workers=cfg.workers) as executor:
                    while idx < len(flat) and not cap_reached():
                        if (
                            cfg.max_tiles_per_specimen
                            and n_scanned >= cfg.max_tiles_per_specimen
                        ):
                            print(
                                f"  [{sp}] hit --max-tiles-per-specimen safety cap "
                                f"({cfg.max_tiles_per_specimen}); moving on. counts={sp_counts}",
                                flush=True,
                            )
                            break

                        chunk = flat[idx : idx + chunk_size]
                        idx += chunk_size
                        futures = {
                            executor.submit(
                                _process_one_tile,
                                iid,
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
                            ): (iid, L, T)
                            for (iid, L, T) in chunk
                        }
                        for fut in as_completed(futures):
                            iid, L, T = futures[fut]
                            n_scanned += 1
                            iid_r, L_r, T_r, label, st, he = fut.result()
                            tile_key = (str(iid), str(L), str(T))

                            if label == "__ERROR__":
                                print(
                                    f"  ! tile {iid}@{L},{T} ({sp}): {st}",
                                    file=sys.stderr,
                                )
                                continue
                            if label is None:
                                continue
                            if sp_counts.get(label, 0) >= cfg.per_class_cap:
                                continue  # filled by the time this result came back (race
                                # under concurrency) -- a little wasted work near
                                # the cap boundary is the acceptable trade-off

                            fn = f"{sp}_{iid}_{L}_{T}_d{cfg.downsample}_{label}.png"
                            fp = os.path.join(img_dir, fn)
                            if cfg.save_images:
                                he.save(fp)
                            w.writerow(
                                [
                                    fp,
                                    label,
                                    BRATS_CLASS_TO_INT[label],
                                    sp,
                                    iid,
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

                            if n_scanned % 200 == 0:
                                elapsed = time.time() - t_start
                                rate = n_scanned / elapsed if elapsed > 0 else 0
                                print(
                                    f"  [{sp}] scanned={n_scanned}/{len(flat)} "
                                    f"kept={sp_counts} ({rate:.1f} tiles/s, "
                                    f"{elapsed:.0f}s elapsed)",
                                    flush=True,
                                )

                print(f"[{sp}] done. counts={sp_counts}", flush=True)
                completed_fh.write(sp + "\n")
                completed_fh.flush()

            except Exception as e:
                msg = f"{sp}\t{type(e).__name__}: {e}"
                print(f"  !! SPECIMEN FAILED, skipping for now: {msg}", file=sys.stderr)
                failed_fh.write(msg + "\n")
                failed_fh.flush()
                continue  # do NOT mark completed -> will retry next run
    finally:
        fh.close()
        completed_fh.close()
        failed_fh.close()

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
        required=True,
        help="Path to manifest.csv from the previous (crashed) run; "
        "specimens found in it are skipped.",
    )
    ap.add_argument("--out", required=True, help="New output directory for this pass.")
    ap.add_argument("--product", default="Gbm")
    ap.add_argument("--treatment", default="H&E")
    ap.add_argument("--downsample", type=int, default=0)
    ap.add_argument("--purity", type=float, default=0.70)
    ap.add_argument("--min-coverage", type=float, default=0.50)
    ap.add_argument("--color-tol", type=float, default=30.0)
    ap.add_argument(
        "--palette-file",
        default=None,
        help="Path to a known-good resolved_palette.json (e.g. "
        "ivygap_patches_all/resolved_palette.json) to load directly, "
        "skipping find_graph_id()/fetch_structures() entirely. "
        "Recommended -- sidesteps the graph_id nondeterminism bug.",
    )
    ap.add_argument("--per-class-cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--redo-last-specimen",
        action="store_true",
        help="Reprocess the last specimen seen in --old-manifest "
        "(it may have been mid-flight when that run crashed).",
    )
    ap.add_argument(
        "--max-tiles-per-specimen",
        type=int,
        default=None,
        help="Optional safety valve: stop scanning a specimen after "
        "this many tiles even if caps aren't all reached (useful "
        "if a specimen is missing a class and would otherwise be "
        "scanned exhaustively).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Concurrent tile-download threads. This is an I/O-bound "
        "workload (network latency dominates), so concurrency is "
        "the main speed lever. 8-16 is reasonable; raise it if the "
        "API tolerates it, lower it if you start seeing timeouts/429s.",
    )
    ap.add_argument(
        "--min-specimen",
        default=None,
        help="Only queue specimens that sort alphabetically at-or-after this "
        "name, e.g. --min-specimen W19-1-1-D.03 skips everything before "
        "it (useful to bypass earlier specimens you don't want to revisit, "
        "such as ones suspected of having no annotation data).",
    )
    ap.add_argument(
        "--skip-specimens",
        default=None,
        help="Comma-separated specimen names to explicitly exclude regardless "
        "of --min-specimen, e.g. --skip-specimens W10-1-1-J.2.03",
    )
    ap.add_argument("--no-save-images", action="store_true")
    a = ap.parse_args()
    run(
        Cfg(
            out_dir=a.out,
            old_manifest=a.old_manifest,
            product=a.product,
            treatment=a.treatment,
            downsample=a.downsample,
            purity=a.purity,
            min_coverage=a.min_coverage,
            color_tol=a.color_tol,
            per_class_cap=a.per_class_cap,
            seed=a.seed,
            redo_last_specimen=a.redo_last_specimen,
            max_tiles_per_specimen=a.max_tiles_per_specimen,
            save_images=not a.no_save_images,
            palette_file=a.palette_file,
            workers=a.workers,
            min_specimen=a.min_specimen,
            skip_specimens=a.skip_specimens,
        )
    )


if __name__ == "__main__":
    main()

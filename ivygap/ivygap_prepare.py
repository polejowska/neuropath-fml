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


def map_structure_to_brats(name, acronym):
    """Ivy GAP structure -> BraTS class, by NAME (acronyms like 'CT' collide
    with brain-atlas structures, e.g. nucleus conterminalis, so are not used
    for the ambiguous classes). Order matters (necrosis substring)."""
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
    if "cellular tumor" in n:
        return "CT"
    if "infiltrating tumor" in n:
        return "IC"
    if "leading edge" in n:
        return None
    if "hyperplastic" in n or "blood vessel" in n:
        return None
    return None


# =====================  API LAYER  ========================================


def _get(url, params=None, binary=False, retries=4):
    import requests

    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=180)
            r.raise_for_status()
            return r.content if binary else r.json()
        except Exception as e:
            last = e
            time.sleep(2**i)
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


def list_products():
    return rma("Product")


def list_treatments():
    return rma("Treatment")


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
            out.append({k: si.get(k) for k in ("id", "width", "height")})
    return out


# GBM ontology located by NAME (unique to Ivy GAP), not by acronym.
GBM_NAME_PROBES = ["pseudopalisading", "microvascular proliferation", "perinecrotic"]


def find_graph_id():
    c = Counter()
    for nm in GBM_NAME_PROBES:
        try:
            for r in rma("Structure", criteria=f"[name$il'*{nm}*']"):
                if r.get("graph_id"):
                    c[int(r["graph_id"])] += 1
        except Exception:
            pass
    return c.most_common(1)[0][0] if c else None


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


def download_whole(image_id, downsample, view=None):
    p = {"downsample": downsample}
    if view:
        p["view"] = view
    raw = _get(f"{BASE}/image_download/{image_id}", params=p, binary=True)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def download_tile(image_id, left, top, downsample, view=None):
    p = {
        "left": left,
        "top": top,
        "width": TILE,
        "height": TILE,
        "downsample": downsample,
    }
    if view:
        p["view"] = view
    raw = _get(f"{BASE}/image_download/{image_id}", params=p, binary=True)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


# =====================  PURE LOGIC (offline-tested)  ======================


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


def label_region(region, palette, labels, purity, min_coverage, color_tol):
    """region: HxWx3 uint8 annotation crop -> (label|None, stats)."""
    if palette.shape[0] == 0 or region.size == 0:
        return None, {"reason": "empty"}
    arr = region.reshape(-1, 3).astype(np.float32)
    idx = _nearest(arr, palette, color_tol)
    annotated = idx >= 0
    cov = float(annotated.mean())
    if cov < min_coverage:
        return None, {"reason": "low_coverage", "coverage": cov}
    lab = np.array([labels[i] if i >= 0 else "" for i in idx], dtype=object)
    vals, counts = np.unique(lab[annotated].astype(str), return_counts=True)
    top = int(counts.argmax())
    frac = float(counts[top] / counts.sum())
    if frac < purity:
        return None, {"reason": "mixed", "coverage": cov, "top_frac": frac}
    return str(vals[top]), {"coverage": cov, "top_frac": frac}


def choose_ad(W, H, d, maxdim):
    ad = d
    while max(W, H) // (2**ad) > maxdim:
        ad += 1
    return ad


def plan_tiles(ann, W, H, d):
    """Yield (full_res_left, full_res_top, ann_crop_box) for full tiles.
    ann is the whole-section annotation raster (HxWx3). Coordinates map the
    full-res tile grid onto the (downsampled) annotation raster."""
    step = TILE * (2**d)
    nx, ny = W // step, H // step
    Ah, Aw = ann.shape[:2]
    sx, sy = Aw / W, Ah / H
    for j in range(ny):
        for i in range(nx):
            x0 = int(i * step * sx)
            x1 = max(int((i + 1) * step * sx), x0 + 1)
            y0 = int(j * step * sy)
            y1 = max(int((j + 1) * step * sy), y0 + 1)
            yield i * step, j * step, (x0, y0, x1, y1)


def tissue_ok(
    rgb,
    bg_frac_max=0.60,
    lowsat_frac_max=0.95,
    eosin_low_frac_max=0.80,
    eosin_low_val=50,
):
    flat = rgb.reshape(-1, 3)
    white = np.all(flat > 230, axis=1)
    black = np.all(flat < 25, axis=1)
    if (white | black).mean() > bg_frac_max:
        return False
    try:
        from skimage.color import rgb2hed, rgb2hsv

        if (rgb2hsv(rgb)[..., 1] < 0.05).mean() > lowsat_frac_max:
            return False
        eo = rgb2hed(rgb)[..., 1]
        eo = (eo - eo.min()) / (eo.ptp() + 1e-8) * 255.0
        if (eo < eosin_low_val).mean() > eosin_low_frac_max:
            return False
    except Exception:
        pass
    return True


def crop_tile(whole, L, T, d):
    """Crop a 512 tile from a whole-section raster captured at downsample d."""
    x, y = L // (2**d), T // (2**d)
    c = whole[y : y + TILE, x : x + TILE]
    if c.shape[0] != TILE or c.shape[1] != TILE:
        pad = np.full((TILE, TILE, 3), 255, np.uint8)
        pad[: c.shape[0], : c.shape[1]] = c
        c = pad
    return c


# =====================  DRIVER  ===========================================


@dataclass
class Cfg:
    out_dir: str
    product: str = "Gbm"
    treatment: str = "H&E"
    downsample: int = 0
    purity: float = 0.70
    min_coverage: float = 0.50
    color_tol: float = 30.0
    roi_maxdim: int = 6000
    workers: int = 16
    max_tiles_per_section: int | None = 400
    max_specimens: int | None = None
    save_images: bool = True


def process_section(iid, W, H, cfg, palette, labels):
    """Return list of (L, T, label) tiles to keep, plus optional whole HE raster."""
    step = TILE * (2**cfg.downsample)
    if W < step or H < step:
        return [], None
    ad = choose_ad(W, H, cfg.downsample, cfg.roi_maxdim)
    ann = download_whole(iid, ad, view="tumor_feature_annotation")
    kept = []
    for L, T, (x0, y0, x1, y1) in plan_tiles(ann, W, H, cfg.downsample):
        lab, _ = label_region(
            ann[y0:y1, x0:x1],
            palette,
            labels,
            cfg.purity,
            cfg.min_coverage,
            cfg.color_tol,
        )
        if lab:
            kept.append((L, T, lab))
    if cfg.max_tiles_per_section and len(kept) > cfg.max_tiles_per_section:
        random.shuffle(kept)
        kept = kept[: cfg.max_tiles_per_section]
    whole_he = download_whole(iid, cfg.downsample) if ad == cfg.downsample else None
    return kept, whole_he


def run(cfg: Cfg) -> str:
    t0 = time.time()
    os.makedirs(cfg.out_dir, exist_ok=True)
    img_dir = os.path.join(cfg.out_dir, "patches")
    os.makedirs(img_dir, exist_ok=True)

    gid = find_graph_id()
    if gid is None:
        raise SystemExit(
            "Could not locate GBM structure graph by name. Run `discover` to inspect."
        )
    palette, labels, resolved = build_palette_auto(fetch_structures(gid))
    if palette.shape[0] == 0:
        raise SystemExit(
            f"No mappable GBM structures found in graph {gid}; run `discover`."
        )
    json.dump(
        resolved,
        open(os.path.join(cfg.out_dir, "resolved_palette.json"), "w"),
        indent=2,
    )
    print(f"graph_id={gid}: {len(labels)} colors -> {sorted(set(labels))}")
    for r in resolved:
        print(f"   rgb{tuple(r['rgb'])} -> {r['brats']:4s} ({r['name']})")

    specimens = list_specimens(cfg.product)
    if cfg.max_specimens:
        specimens = specimens[: cfg.max_specimens]
    print(
        f"{len(specimens)} specimens; workers={cfg.workers} "
        f"downsample={cfg.downsample} cap={cfg.max_tiles_per_section}/section\n"
    )

    manifest = os.path.join(cfg.out_dir, "manifest.csv")
    counts, n_written, n_sections = Counter(), 0, 0
    fh = open(manifest, "w", newline="")
    w = csv.writer(fh)
    w.writerow(
        [
            "filepath",
            "label",
            "label_int",
            "specimen",
            "image_id",
            "left",
            "top",
            "downsample",
            "purity_frac",
        ]
    )

    def fetch_and_save(iid, sp, L, T, lab, whole_he):
        he = (
            crop_tile(whole_he, L, T, cfg.downsample)
            if whole_he is not None
            else download_tile(iid, L, T, cfg.downsample)
        )
        if not tissue_ok(he):
            return None
        fn = f"{sp}_{iid}_{L}_{T}_d{cfg.downsample}_{lab}.png"
        fp = os.path.join(img_dir, fn)
        if cfg.save_images:
            Image.fromarray(he).save(fp)
        return (fp, lab, BRATS_CLASS_TO_INT[lab], sp, iid, L, T, cfg.downsample)

    for sp in specimens:
        for si in section_images_for_specimen(sp, cfg.treatment):
            iid, W, H = si["id"], int(si["width"]), int(si["height"])
            try:
                kept, whole_he = process_section(iid, W, H, cfg, palette, labels)
            except Exception as e:
                print(f"  ! section {iid}: {e}", file=sys.stderr)
                continue
            n_sections += 1
            with ThreadPoolExecutor(max_workers=cfg.workers) as ex:
                futs = [
                    ex.submit(fetch_and_save, iid, sp, L, T, lab, whole_he)
                    for (L, T, lab) in kept
                ]
                for f in as_completed(futs):
                    try:
                        row = f.result()
                    except Exception as e:
                        print(f"  ! tile: {e}", file=sys.stderr)
                        continue
                    if row:
                        w.writerow(row)
                        counts[row[1]] += 1
                        n_written += 1
            el = time.time() - t0
            rate = n_written / el if el else 0
            print(
                f"[{sp}/{iid}] kept {len(kept)} -> wrote {n_written} total "
                f"| {n_sections} sections | {el:5.0f}s | {rate:4.1f} patches/s "
                f"| dist={dict(counts)}"
            )
    fh.close()
    with open(
        os.path.join(cfg.out_dir, "class_distribution.csv"), "w", newline=""
    ) as g:
        cw = csv.writer(g)
        cw.writerow(["label", "label_int", "count"])
        for k in sorted(counts):
            cw.writerow([k, BRATS_CLASS_TO_INT[k], counts[k]])
    print(f"\nDONE in {time.time() - t0:.0f}s: {n_written} patches -> {manifest}")
    print(f"class distribution: {dict(counts)}")
    return manifest


# =====================  DISCOVER  =========================================


def discover(hint="GBM"):
    print("== PRODUCTS ==")
    for p in list_products():
        ab = p.get("abbreviation", "")
        if hint.lower() in ab.lower() or "gbm" in ab.lower():
            print(f"  {ab:18s} id={p.get('id')}  {p.get('name')}")
    print("\n== H&E-ish TREATMENTS ==")
    for t in list_treatments():
        if "h&e" in t.get("name", "").lower() or t.get("name") == "Annotated":
            print(f"  {t.get('name')}")
    gid = find_graph_id()
    print(f"\n== GBM STRUCTURE GRAPH (graph_id={gid}) ==")
    if gid:
        for acr, name, rgb in fetch_structures(gid):
            print(
                f"  rgb{rgb} {acr or '?':10s} -> "
                f"{str(map_structure_to_brats(name, acr)):4} ({name})"
            )


# =====================  OFFLINE SELF-TEST  ================================


def selftest():
    # collision fix: a brain structure named 'conterminalis' w/ acronym CT -> None
    assert map_structure_to_brats("conterminalis", "CT") is None
    assert map_structure_to_brats("Cellular Tumor", "CT") == "CT"
    assert map_structure_to_brats("Microvascular proliferation", "CTmvp") == "MP"
    assert (
        map_structure_to_brats("Pseudopalisading cells around necrosis", "CTpan")
        == "PN"
    )
    assert map_structure_to_brats("Perinecrotic zone", "CTpnz") is None
    assert map_structure_to_brats("Necrosis", "CTne") == "NC"
    print("[ok] name-based mapping (conterminalis no longer -> CT)")

    structs = [
        ("CT", "Cellular Tumor", (10, 10, 10)),
        ("CTmvp", "Microvascular proliferation", (0, 200, 0)),
        ("CTpan", "Pseudopalisading cells around necrosis", (220, 20, 60)),
        ("CTne", "Necrosis", (30, 30, 200)),
        ("LE", "Leading Edge", (240, 240, 0)),
    ]
    pal, labs, _ = build_palette_auto(structs)
    assert sorted(set(labs)) == ["CT", "MP", "NC", "PN"], labs
    print(f"[ok] palette -> {sorted(set(labs))}")

    # ROI planning: a 4096x2048 full-res section, downsample 0, annotation raster
    # downsampled by choose_ad. Paint MP(green) only in top-left tile region.
    W, H, d = 4096, 2048, 0
    ad = choose_ad(W, H, d, maxdim=1024)  # -> 2 (4096/4=1024)
    Ah, Aw = H // (2**ad), W // (2**ad)
    ann = np.full((Ah, Aw, 3), 255, np.uint8)  # white = unannotated
    # tile grid is 8x4 (step=512). Fill tile (0,0)'s annotation region green.
    sx, sy = Aw / W, Ah / H
    x1 = int(512 * sx)
    y1 = int(512 * sy)
    ann[0:y1, 0:x1] = (0, 200, 0)
    kept = [
        (L, T, lab)
        for (L, T, (a, b, c, dd)) in [
            (L, T, box) for (L, T, box) in plan_tiles(ann, W, H, d)
        ]
        for lab, _ in [label_region(ann[b:dd, a:c], pal, labs, 0.7, 0.5, 30)]
        if lab
    ]
    assert kept == [(0, 0, "MP")], kept
    print(f"[ok] ROI planning keeps only annotated tile -> {kept}")

    # crop_tile local cropping + padding
    whole = np.random.randint(80, 180, (700, 700, 3), np.uint8)
    t = crop_tile(whole, 0, 0, 0)
    assert t.shape == (512, 512, 3)
    t2 = crop_tile(whole, 512, 512, 0)
    assert t2.shape == (512, 512, 3)  # padded
    print("[ok] local crop + pad")

    # concurrency assembly with a stubbed network fetch
    import ivygap_prepare as M

    orig = M.download_tile
    M.download_tile = lambda iid, L, T, d, view=None: np.random.randint(
        80, 180, (512, 512, 3), np.uint8
    )
    try:
        cfg = Cfg(out_dir="/tmp/_st", save_images=False, workers=4)
        rows = []
        with ThreadPoolExecutor(max_workers=4) as ex:

            def job(L, T, lab):
                he = M.download_tile(1, L, T, 0)
                return (f"x_{L}_{T}.png", lab) if tissue_ok(he) else None

            for f in as_completed(
                [ex.submit(job, L, T, "MP") for (L, T) in [(0, 0), (512, 0), (0, 512)]]
            ):
                r = f.result()
                if r:
                    rows.append(r)
        assert len(rows) == 3
    finally:
        M.download_tile = orig
    print("[ok] concurrent fetch/assemble (stubbed network)")
    assert BRATS_CLASS_TO_INT["PN"] == 7 and BRATS_CLASS_TO_INT["CT"] == 0
    print("\nALL OFFLINE TESTS PASSED")


# =====================  CLI  ==============================================


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    dp = sub.add_parser("discover")
    dp.add_argument("--hint", default="GBM")
    rp = sub.add_parser("run")
    rp.add_argument("--out", required=True)
    rp.add_argument("--product", default="Gbm")
    rp.add_argument("--treatment", default="H&E")
    rp.add_argument("--downsample", type=int, default=0)
    rp.add_argument("--purity", type=float, default=0.70)
    rp.add_argument("--min-coverage", type=float, default=0.50)
    rp.add_argument("--color-tol", type=float, default=30.0)
    rp.add_argument("--roi-maxdim", type=int, default=6000)
    rp.add_argument("--workers", type=int, default=16)
    rp.add_argument("--max-tiles-per-section", type=int, default=40000)
    rp.add_argument("--max-specimens", type=int, default=None)
    rp.add_argument("--no-save-images", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "discover":
        discover(a.hint)
    elif a.cmd == "run":
        run(
            Cfg(
                out_dir=a.out,
                product=a.product,
                treatment=a.treatment,
                downsample=a.downsample,
                purity=a.purity,
                min_coverage=a.min_coverage,
                color_tol=a.color_tol,
                roi_maxdim=a.roi_maxdim,
                workers=a.workers,
                max_tiles_per_section=a.max_tiles_per_section,
                max_specimens=a.max_specimens,
                save_images=not a.no_save_images,
            )
        )


if __name__ == "__main__":
    main()

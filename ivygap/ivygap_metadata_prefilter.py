from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import requests

BASE = "https://api.brain-map.org/api/v2"
MAPPABLE_CLASSES = ["CT", "MP", "NC", "PN", "IC"]

_SESSION = requests.Session()


def _get(url, params=None, retries=4):
    last = None
    for i in range(retries):
        try:
            r = _SESSION.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
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


def list_specimens(product_abbrev="Gbm"):
    recs = rma("Specimen", criteria=f"products[abbreviation$eq'{product_abbrev}']")
    return sorted(
        {r["external_specimen_name"] for r in recs if r.get("external_specimen_name")}
    )


def load_manifest_specimens(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path, newline="") as fh:
        return {row["specimen"] for row in csv.DictReader(fh) if row.get("specimen")}


def load_completed_specimens(path):
    if not path or not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def derive_completed_path(manifest_path, explicit_override):
    if explicit_override:
        return explicit_override
    if manifest_path:
        candidate = os.path.join(
            os.path.dirname(manifest_path) or ".", "specimens_completed.txt"
        )
        if os.path.exists(candidate):
            return candidate
    return None


def fetch_tumor_features(specimen_name):
    """Returns the raw list of TumorFeature records for a specimen, each with
    an embedded 'structure' association (per rma::include,structure)."""
    return rma(
        "TumorFeature",
        criteria=f"data_set(specimen[external_specimen_name$eq'{specimen_name}'])",
        include="structure",
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Print raw TumorFeature records for --check-specimens and exit. "
        "ALWAYS run this first to confirm field names before trusting the "
        "full run below.",
    )
    ap.add_argument(
        "--check-specimens",
        default=None,
        help="Comma-separated specimen names to inspect in --debug mode.",
    )
    ap.add_argument("--old-manifest", default=None)
    ap.add_argument("--balanced-manifest", default=None)
    ap.add_argument("--priority-manifest", default=None)
    ap.add_argument("--balanced-completed", default=None)
    ap.add_argument("--priority-completed", default=None)
    ap.add_argument(
        "--palette-file",
        default=None,
        help="resolved_palette.json, used to map structure acronym -> BraTS class. "
        "Required unless --debug.",
    )
    ap.add_argument("--out", default=".")
    ap.add_argument("--product", default="Gbm")
    ap.add_argument(
        "--priority-classes",
        default="MP",
        help="Comma-separated classes to check for (default: MP, since that's "
        "the one your audit shows as most critically underrepresented).",
    )
    ap.add_argument(
        "--area-field",
        default="area",
        help="Field name on the TumorFeature record holding the area value. "
        "UNVERIFIED against a live response -- check with --debug first "
        "and override this if the real field name differs (candidates per "
        "the docs: area, normalized_area, nuclei_count, nuclei_fraction_coverage).",
    )
    ap.add_argument("--min-specimen", default=None)
    ap.add_argument("--skip-specimens", default=None)
    a = ap.parse_args()

    if a.debug:
        if not a.check_specimens:
            raise SystemExit("--debug requires --check-specimens NAME1,NAME2,...")
        for sp in a.check_specimens.split(","):
            sp = sp.strip()
            print(f"\n=== raw TumorFeature records for '{sp}' ===")
            try:
                recs = fetch_tumor_features(sp)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            print(f"  {len(recs)} records returned")
            for r in recs[:5]:
                print(f"  {json.dumps(r, indent=2)[:800]}")
            if len(recs) > 5:
                print(f"  ... and {len(recs) - 5} more (truncated)")
            if recs:
                print(f"  Top-level keys on first record: {sorted(recs[0].keys())}")
                struct = recs[0].get("structure")
                if struct:
                    print(f"  Keys on embedded 'structure': {sorted(struct.keys())}")
        print(
            "\nCompare these field names against --area-field (default 'area'). "
            "If 'area' isn't among the printed keys, rerun the full pass with "
            "--area-field set to whatever the real key is."
        )
        return

    if not a.palette_file:
        raise SystemExit("--palette-file is required for a real (non-debug) run.")

    resolved = json.load(open(a.palette_file))
    acronym_to_brats = {r["acronym"]: r["brats"] for r in resolved}
    priority_classes = [c.strip() for c in a.priority_classes.split(",") if c.strip()]

    old_skip = load_manifest_specimens(a.old_manifest)
    bal_skip = load_manifest_specimens(a.balanced_manifest)
    pri_skip = load_manifest_specimens(a.priority_manifest)
    bal_completed_path = derive_completed_path(
        a.balanced_manifest, a.balanced_completed
    )
    pri_completed_path = derive_completed_path(
        a.priority_manifest, a.priority_completed
    )
    combined_skip = (
        old_skip
        | bal_skip
        | pri_skip
        | load_completed_specimens(bal_completed_path)
        | load_completed_specimens(pri_completed_path)
    )
    print(
        f"[skip set] {len(combined_skip)} specimens already processed across all prior runs."
    )

    all_specimens = list_specimens(a.product)
    queue = [sp for sp in all_specimens if sp not in combined_skip]
    if a.min_specimen:
        queue = [sp for sp in queue if sp >= a.min_specimen]
    if a.skip_specimens:
        explicit = {s.strip() for s in a.skip_specimens.split(",") if s.strip()}
        queue = [sp for sp in queue if sp not in explicit]
    print(f"{len(queue)} specimens to check via metadata (no image downloads at all).")

    os.makedirs(a.out, exist_ok=True)
    report_path = os.path.join(a.out, "specimen_tumor_feature_areas.csv")
    positive_path = os.path.join(a.out, "specimens_with_priority.txt")
    negative_path = os.path.join(a.out, "specimens_metadata_negative.txt")

    zero_everything_warnings = 0
    with (
        open(report_path, "w", newline="") as report_fh,
        open(positive_path, "w") as pos_fh,
        open(negative_path, "w") as neg_fh,
    ):
        writer = csv.writer(report_fh)
        writer.writerow(["specimen"] + MAPPABLE_CLASSES + ["unmapped_structures_seen"])

        for i, sp in enumerate(queue, 1):
            try:
                recs = fetch_tumor_features(sp)
            except Exception as e:
                print(
                    f"  ! {sp}: fetch failed ({e}), skipping for now", file=sys.stderr
                )
                continue

            class_area = {c: 0.0 for c in MAPPABLE_CLASSES}
            unmapped = set()
            any_nonzero_anywhere = False
            for rec in recs:
                struct = rec.get("structure") or {}
                acr = struct.get("acronym")
                brats = acronym_to_brats.get(acr)
                val = rec.get(a.area_field)
                if val is None:
                    continue
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    any_nonzero_anywhere = True
                if brats is None:
                    if acr:
                        unmapped.add(acr)
                    continue
                class_area[brats] += val

            if recs and not any_nonzero_anywhere:
                zero_everything_warnings += 1

            writer.writerow(
                [sp]
                + [f"{class_area[c]:.4f}" for c in MAPPABLE_CLASSES]
                + [";".join(sorted(unmapped))]
            )
            report_fh.flush()

            has_priority = any(class_area.get(c, 0) > 0 for c in priority_classes)
            if has_priority:
                pos_fh.write(sp + "\n")
                pos_fh.flush()
            else:
                neg_fh.write(sp + "\n")
                neg_fh.flush()

            if i % 100 == 0:
                print(f"  checked {i}/{len(queue)} specimens...", flush=True)

    if zero_everything_warnings > len(queue) * 0.5:
        print(
            f"\n[!] WARNING: {zero_everything_warnings}/{len(queue)} specimens had "
            f"records but EVERY single area value (using field '{a.area_field}') was "
            f"zero or unparseable. That's suspicious -- it likely means '{a.area_field}' "
            f"is the WRONG field name. Rerun with --debug first to find the correct one "
            f"before trusting {positive_path}/{negative_path}."
        )

    print(f"\nWrote:\n  {report_path}\n  {positive_path}\n  {negative_path}")
    print(
        "Feed specimens_with_priority.txt into ivygap_priority_mp_pn.py's "
        "--specimen-allowlist to skip all metadata-negative specimens entirely "
        "(zero image API calls for them)."
    )


if __name__ == "__main__":
    main()

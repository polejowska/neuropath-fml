#!/usr/bin/env bash
# check_predictions.sh -- sanity-check a BraTS-Path 2026 container run.
#
# Verifies, given the /input tar shard(s) you fed the container and the
# /output directory it wrote to:
#   1. predictions.csv exists with the exact header "SubjectID,Prediction"
#   2. every SubjectID in the tar(s) got exactly one prediction row
#      (no dropped / duplicated / extra subjects)
#   3. every Prediction value is an integer in [0, 9]
#   4. (if present) reports inference_summary.json's runtime + whether the
#      time-budget TTA fallback ever triggered
#
# Usage:
#   ./check_predictions.sh /path/to/input /path/to/output
#
# Exit code: 0 if all checks pass, 1 if any check fails.

set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 /path/to/input /path/to/output" >&2
  exit 2
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CSV_PATH="$OUTPUT_DIR/predictions.csv"
SUMMARY_PATH="$OUTPUT_DIR/inference_summary.json"

fail=0
EXPECTED_IDS_FILE=""
ACTUAL_IDS_FILE=""
PRED_VALUES_FILE=""
trap 'rm -f "${EXPECTED_IDS_FILE:-}" "${ACTUAL_IDS_FILE:-}" "${PRED_VALUES_FILE:-}"' EXIT
note() { echo "  -> $*"; }
ok()   { echo "[OK]   $*"; }
bad()  { echo "[FAIL] $*"; fail=1; }

echo "== 1. Locating input tar shard(s) =="
mapfile -t TAR_FILES < <(find "$INPUT_DIR" -type f -name "*.tar" | sort)
if [[ ${#TAR_FILES[@]} -eq 0 ]]; then
  bad "No .tar files found under $INPUT_DIR"
  exit 1
fi
for t in "${TAR_FILES[@]}"; do note "$t"; done
ok "Found ${#TAR_FILES[@]} shard(s)."

echo
echo "== 2. Extracting expected SubjectIDs from tar(s) =="
EXPECTED_IDS_FILE="$(mktemp)"
: > "$EXPECTED_IDS_FILE"
for t in "${TAR_FILES[@]}"; do
  tar tf "$t" | grep '\.jpg$' | sed -e 's#^\./##' -e 's/\.jpg$//' >> "$EXPECTED_IDS_FILE"
done
n_expected=$(wc -l < "$EXPECTED_IDS_FILE")
n_expected_unique=$(sort -u "$EXPECTED_IDS_FILE" | wc -l)
note "expected samples (from .jpg entries): $n_expected"
if [[ "$n_expected" -ne "$n_expected_unique" ]]; then
  bad "Duplicate __key__ values found across shard(s) -- $n_expected total vs $n_expected_unique unique. Check the shards themselves."
else
  ok "All expected SubjectIDs are unique across shard(s)."
fi

echo
echo "== 3. Checking predictions.csv exists and has the right header =="
if [[ ! -f "$CSV_PATH" ]]; then
  bad "Missing $CSV_PATH"
  exit 1
fi
ok "$CSV_PATH exists."

header="$(head -n1 "$CSV_PATH" | tr -d '\r')"
if [[ "$header" == "SubjectID,Prediction" ]]; then
  ok "Header is exactly 'SubjectID,Prediction'."
else
  bad "Header is '$header', expected 'SubjectID,Prediction'."
fi

echo
echo "== 4. Checking row count and SubjectID coverage =="
ACTUAL_IDS_FILE="$(mktemp)"
tail -n +2 "$CSV_PATH" | awk -F',' '{print $1}' > "$ACTUAL_IDS_FILE"
n_actual=$(wc -l < "$ACTUAL_IDS_FILE")
n_actual_unique=$(sort -u "$ACTUAL_IDS_FILE" | wc -l)
note "predictions.csv rows (excl. header): $n_actual"

if [[ "$n_actual" -ne "$n_expected" ]]; then
  bad "Row count mismatch: $n_actual predictions vs $n_expected expected samples."
else
  ok "Row count matches expected sample count ($n_actual)."
fi

if [[ "$n_actual" -ne "$n_actual_unique" ]]; then
  bad "predictions.csv contains duplicate SubjectID rows ($n_actual rows, $n_actual_unique unique)."
else
  ok "No duplicate SubjectID rows in predictions.csv."
fi

missing=$(comm -23 <(sort -u "$EXPECTED_IDS_FILE") <(sort -u "$ACTUAL_IDS_FILE") | head -5)
extra=$(comm -13 <(sort -u "$EXPECTED_IDS_FILE") <(sort -u "$ACTUAL_IDS_FILE") | head -5)
if [[ -n "$missing" ]]; then
  bad "SubjectIDs present in the tar but MISSING from predictions.csv (showing up to 5):"
  echo "$missing" | sed 's/^/       /'
fi
if [[ -n "$extra" ]]; then
  bad "SubjectIDs present in predictions.csv but NOT in the tar (showing up to 5):"
  echo "$extra" | sed 's/^/       /'
fi
if [[ -z "$missing" && -z "$extra" ]]; then
  ok "Every expected SubjectID has exactly one prediction, and no extras."
fi

echo
echo "== 5. Checking Prediction values are integers in [0, 9] =="
PRED_VALUES_FILE="$(mktemp)"
tail -n +2 "$CSV_PATH" | awk -F',' '{print $2}' | tr -d '\r' > "$PRED_VALUES_FILE"
bad_values=$(grep -vE '^[0-9]$' "$PRED_VALUES_FILE" | sort -u | head -5 || true)
if [[ -n "$bad_values" ]]; then
  bad "Found Prediction values outside [0-9] (showing up to 5 distinct bad values):"
  echo "$bad_values" | sed 's/^/       /'
else
  ok "All Prediction values are single-digit integers in [0, 9]."
fi

echo
echo "== 6. inference_summary.json (if present) =="
if [[ -f "$SUMMARY_PATH" ]]; then
  ok "$SUMMARY_PATH exists."
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$SUMMARY_PATH" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    s = json.load(f)
print(f"  -> pipeline used for predictions.csv : {s.get('pipeline_used_for_predictions_csv')}")
print(f"  -> tta_aug                            : {s.get('tta_aug')}")
print(f"  -> total_runtime_hours                : {s.get('total_runtime_hours')}")
print(f"  -> n_rows                             : {s.get('n_rows')}")
print(f"  -> n_shards                           : {s.get('n_shards')}")
PYEOF
  else
    cat "$SUMMARY_PATH"
  fi
else
  note "Not found (not written, or you pointed OUTPUT_DIR somewhere else) -- skipping."
fi

echo
if [[ "$fail" -eq 0 ]]; then
  echo "ALL CHECKS PASSED."
else
  echo "ONE OR MORE CHECKS FAILED -- see [FAIL] lines above."
fi
exit "$fail"

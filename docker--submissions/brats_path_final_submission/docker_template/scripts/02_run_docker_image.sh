#!/usr/bin/env bash
set -euo pipefail

# Directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Template project root, one level above scripts/.
TEMPLATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Docker image repository/name to run. Must match scripts/01_build_image.sh.
IMAGE_NAME="${IMAGE_NAME:-template_docker}"
# Docker image tag to run. Must match scripts/01_build_image.sh.
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Host folder containing challenge WebDataset .tar files. Required.
INPUT_DIR="${INPUT_DIR:-}"
# Host folder where predictions.csv and other outputs will be written.
OUTPUT_DIR="${OUTPUT_DIR:-$TEMPLATE_DIR/output}"

# Docker shared memory size. Increase this for large batches or DataLoader use.
SHM_SIZE="${SHM_SIZE:-16g}"

# GPU selection passed to docker run. Default uses all visible GPUs.
# Example: GPUS=0 uses only GPU 0. The challenge requires GPU execution.
GPU_ARGS=(--gpus "${GPUS:-all}")

if ! docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1; then
  echo "ERROR: Missing local Docker image: ${IMAGE_NAME}:${IMAGE_TAG}" >&2
  echo "Build it first with:" >&2
  echo "  IMAGE_NAME=$IMAGE_NAME IMAGE_TAG=$IMAGE_TAG ./scripts/01_build_image.sh" >&2
  exit 2
fi

if [[ -z "$INPUT_DIR" ]]; then
  echo "ERROR: INPUT_DIR is required." >&2
  echo "Example:" >&2
  echo "  INPUT_DIR=/path/to/challenge_input OUTPUT_DIR=/path/to/challenge_output ./scripts/02_run_docker_image.sh" >&2
  exit 2
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: Missing input directory: $INPUT_DIR" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "Running image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Input:  $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

docker run --rm \
  "${GPU_ARGS[@]}" \
  --shm-size "$SHM_SIZE" \
  -v "$(realpath "$INPUT_DIR"):/input:ro" \
  -v "$(realpath "$OUTPUT_DIR"):/output" \
  "${IMAGE_NAME}:${IMAGE_TAG}"

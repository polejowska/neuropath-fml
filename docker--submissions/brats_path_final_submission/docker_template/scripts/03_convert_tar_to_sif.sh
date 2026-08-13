#!/usr/bin/env bash
set -euo pipefail

# Directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Template project root, one level above scripts/.
TEMPLATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Docker image repository/name used to derive the default tar and SIF names.
# Keep this aligned with scripts/01_build_image.sh unless TAR_PATH is set directly.
IMAGE_NAME="${IMAGE_NAME:-template_docker}"
# Docker image tag used to derive the default tar and SIF names.
IMAGE_TAG="${IMAGE_TAG:-latest}"
# Filesystem-safe name derived from IMAGE_NAME and IMAGE_TAG.
# Example: template_docker:latest -> template_docker_latest
IMAGE_NAME_SAFE="$(echo "${IMAGE_NAME}_${IMAGE_TAG}" | tr '/:' '__')"

# Folder containing the Docker archive tar produced by scripts/01_build_image.sh.
ARCHIVE_DIR="${ARCHIVE_DIR:-$TEMPLATE_DIR/docker_archives}"
# Full path to the Docker archive tar used as Apptainer input.
TAR_PATH="${TAR_PATH:-$ARCHIVE_DIR/${IMAGE_NAME_SAFE}.tar}"

# Folder where the converted Apptainer SIF will be written.
SIF_DIR="${SIF_DIR:-$TEMPLATE_DIR/sif}"
# Full output SIF path.
SIF_PATH="${SIF_PATH:-$SIF_DIR/${IMAGE_NAME_SAFE}.sif}"
# Temporary SIF path used during conversion; renamed to SIF_PATH only after success.
TMP_SIF="${SIF_PATH}.tmp.$$"

# Apptainer executable. Set APPTAINER_BIN=/path/to/apptainer if it is not on PATH.
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"

# Root folder for Apptainer cache and temporary files used during conversion.
# Put this on large local/scratch storage if the default location is slow or quota-limited.
APPTAINER_CACHE_ROOT="${APPTAINER_CACHE_ROOT:-$TEMPLATE_DIR/.apptainer}"
# Apptainer cache directory for pulled layers and conversion metadata.
APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-$APPTAINER_CACHE_ROOT/cache}"
# Apptainer temporary build directory. Needs enough free space for image conversion.
APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-$APPTAINER_CACHE_ROOT/tmp}"

# Rebuild behavior. FORCE=1 overwrites an existing SIF; FORCE=0 skips if it exists.
FORCE="${FORCE:-0}"

cleanup() {
  rm -f "$TMP_SIF"
}
trap cleanup EXIT

if [[ ! -f "$TAR_PATH" ]]; then
  echo "ERROR: Missing Docker archive: $TAR_PATH" >&2
  echo "Build/save it first with:" >&2
  echo "  IMAGE_NAME=$IMAGE_NAME IMAGE_TAG=$IMAGE_TAG ./scripts/01_build_image.sh" >&2
  exit 2
fi

if [[ -f "$SIF_PATH" && "$FORCE" != "1" ]]; then
  echo "SIF already exists: $SIF_PATH"
  echo "Set FORCE=1 to rebuild it."
  exit 0
fi

if [[ -x "$APPTAINER_BIN" ]]; then
  :
elif command -v "$APPTAINER_BIN" >/dev/null 2>&1; then
  APPTAINER_BIN="$(command -v "$APPTAINER_BIN")"
else
  echo "ERROR: apptainer is not on PATH." >&2
  echo "Set APPTAINER_BIN=/path/to/apptainer if needed." >&2
  exit 2
fi

mkdir -p "$SIF_DIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
export APPTAINER_CACHEDIR
export APPTAINER_TMPDIR

echo "Docker archive: $TAR_PATH"
echo "SIF path:       $SIF_PATH"
echo "Apptainer:      $($APPTAINER_BIN --version)"
echo "Cache dir:      $APPTAINER_CACHEDIR"
echo "Tmp dir:        $APPTAINER_TMPDIR"

"$APPTAINER_BIN" build "$TMP_SIF" "docker-archive://$TAR_PATH"
mv "$TMP_SIF" "$SIF_PATH"
chmod 0644 "$SIF_PATH"

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$SIF_PATH" > "${SIF_PATH}.sha256"
  echo "Wrote checksum: ${SIF_PATH}.sha256"
fi

echo "Finished SIF: $SIF_PATH"

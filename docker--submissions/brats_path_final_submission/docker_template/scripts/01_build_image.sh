#!/usr/bin/env bash
set -euo pipefail

# Directory containing this script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Template project root, one level above scripts/.
TEMPLATE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Docker image repository/name to build. Change this for your algorithm or team.
IMAGE_NAME="${IMAGE_NAME:-template_docker}"
# Docker image tag. Use latest for local testing or a version string for releases.
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Folder where the saved Docker archive tar will be written.
ARCHIVE_DIR="${ARCHIVE_DIR:-$TEMPLATE_DIR/docker_archives}"
# Full output tar path. The default converts IMAGE_NAME:IMAGE_TAG into a safe filename.
# Example: template_docker:latest -> docker_archives/template_docker_latest.tar
ARCHIVE="${ARCHIVE:-$ARCHIVE_DIR/$(echo "${IMAGE_NAME}_${IMAGE_TAG}" | tr '/:' '__').tar}"

command -v docker >/dev/null || { echo "docker is not on PATH" >&2; exit 1; }

docker --version
echo "Template image: ${IMAGE_NAME}:${IMAGE_TAG}"
docker info --format 'Default runtime: {{.DefaultRuntime}}' || true

cd "$TEMPLATE_DIR"
# Build context is the template folder. If you move files, update Dockerfile
# COPY paths rather than this command.
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"

mkdir -p "$ARCHIVE_DIR"
docker save -o "$ARCHIVE" "${IMAGE_NAME}:${IMAGE_TAG}"
echo "Saved ${IMAGE_NAME}:${IMAGE_TAG} to $ARCHIVE"

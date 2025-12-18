#!/usr/bin/env bash
set -euo pipefail

# Pull a default base image into template/base.sif.
# You can override the source image by setting BASE_IMAGE (e.g., docker://nvidia/cuda:12.4.1-runtime-ubuntu22.04).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMPLATE_DIR="${ROOT_DIR}/template"
TARGET="${TEMPLATE_DIR}/base.sif"
SRC_IMAGE="${BASE_IMAGE:-docker://ubuntu:22.04}"

mkdir -p "${TEMPLATE_DIR}"
echo "Pulling ${SRC_IMAGE} into ${TARGET}"
singularity pull --force "${TARGET}" "${SRC_IMAGE}"
echo "Done."

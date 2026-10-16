#!/usr/bin/env bash
# Explicit CLI operations; dry plans never imply permission for science.
set -euo pipefail
export PROJECT_DIR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec python -u -s "${PROJECT_DIR}/scripts/luka_fullstat.py" "$@"

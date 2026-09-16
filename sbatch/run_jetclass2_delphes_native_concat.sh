#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?source-pinned project directory required}"
export JC2_SITE=sporc_a100
SPEC="${2:?experiment spec required}"
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_native_concat.py" run --spec "${SPEC}"

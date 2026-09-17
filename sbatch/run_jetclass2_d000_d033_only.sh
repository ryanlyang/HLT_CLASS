#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${1:?source-pinned project directory required}"
SPEC="${2:?experiment spec required}"
export JC2_SITE="${3:?execution site required}"

source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"

python -s "${PROJECT_DIR}/scripts/jetclass2_d000_d033_only.py" run \
  --spec "${SPEC}"

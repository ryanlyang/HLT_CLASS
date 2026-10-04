#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?absolute pinned project}"
SPEC="${2:?immutable campaign spec}"
TASK="${3:?registered task}"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
if [[ "${TASK}" == preflight ]]; then
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
exec python -s "${PROJECT_DIR}/scripts/noise_k2.py" run --spec "${SPEC}" --task "${TASK}"

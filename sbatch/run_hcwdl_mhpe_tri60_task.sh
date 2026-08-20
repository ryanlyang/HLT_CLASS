#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_TRI60_SPEC:?HCWDL_TRI60_SPEC is required}"
: "${HCWDL_TRI60_TASK:?HCWDL_TRI60_TASK is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_mhpe_tri60_task.py" \
  --spec "${HCWDL_TRI60_SPEC}" --task "${HCWDL_TRI60_TASK}" --device cuda

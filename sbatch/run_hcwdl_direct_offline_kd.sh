#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${HCWDL_DIRECT_SPEC:?HCWDL_DIRECT_SPEC is required}"
: "${HCWDL_DIRECT_TASK:?HCWDL_DIRECT_TASK is required}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_direct_offline_kd_task.py" \
  --campaign-spec "${HCWDL_DIRECT_SPEC}" \
  --task "${HCWDL_DIRECT_TASK}"

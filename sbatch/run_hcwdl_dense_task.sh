#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${HCWDL_DENSE_SPEC:?HCWDL_DENSE_SPEC is required}"
: "${HCWDL_DENSE_TASK:?HCWDL_DENSE_TASK is required}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_dense_task.py" \
  --campaign-spec "${HCWDL_DENSE_SPEC}" \
  --task "${HCWDL_DENSE_TASK}"

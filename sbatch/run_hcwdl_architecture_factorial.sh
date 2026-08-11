#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${HCWDL_AI_SPEC:?HCWDL_AI_SPEC is required}"
: "${HCWDL_AI_TASK:?HCWDL_AI_TASK is required}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_architecture_factorial_task.py" \
  --campaign-spec "${HCWDL_AI_SPEC}" \
  --task "${HCWDL_AI_TASK}"

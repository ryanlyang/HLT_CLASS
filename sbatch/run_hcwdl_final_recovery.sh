#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${HCWDL_FINAL_RECOVERY_SPEC:?HCWDL_FINAL_RECOVERY_SPEC is required}"
: "${HCWDL_FINAL_RECOVERY_TASK:?HCWDL_FINAL_RECOVERY_TASK is required}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_final_recovery_task.py" \
  --recovery-spec "${HCWDL_FINAL_RECOVERY_SPEC}" \
  --task "${HCWDL_FINAL_RECOVERY_TASK}"

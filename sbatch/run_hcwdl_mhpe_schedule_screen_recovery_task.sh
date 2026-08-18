#!/usr/bin/env bash
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_SCHEDULE_SCREEN_RECOVERY_SPEC:?HCWDL_SCHEDULE_SCREEN_RECOVERY_SPEC is required}"
: "${HCWDL_SCHEDULE_SCREEN_TASK:?HCWDL_SCHEDULE_SCREEN_TASK is required}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_mhpe_schedule_screen_recovery_task.py" \
  --recovery-spec "${HCWDL_SCHEDULE_SCREEN_RECOVERY_SPEC}" \
  --task "${HCWDL_SCHEDULE_SCREEN_TASK}" --device cuda

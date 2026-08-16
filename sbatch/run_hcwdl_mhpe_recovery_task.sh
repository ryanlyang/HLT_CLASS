#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_mhpe_recovery_task.py" \
  --recovery-spec "${HCWDL_MHPE_RECOVERY_SPEC}" --task "${HCWDL_MHPE_TASK}"

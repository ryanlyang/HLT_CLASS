#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR must be set}"
: "${HCWDL_CONCAT_RECOVERY:?HCWDL_CONCAT_RECOVERY must be set}"
: "${HCWDL_CONCAT_TASK:?HCWDL_CONCAT_TASK must be set}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_offline_hlt_concat_recovery_task.py" \
  --recovery "${HCWDL_CONCAT_RECOVERY}" --task "${HCWDL_CONCAT_TASK}" --device cuda

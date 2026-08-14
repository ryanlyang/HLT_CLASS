#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ -n "${HCWDL_UB_FOUNDATION_SPEC:-}" ]]; then
  exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_unified_balanced_task.py" --foundation-spec "${HCWDL_UB_FOUNDATION_SPEC}" --task "${HCWDL_UB_TASK}"
fi
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_unified_balanced_task.py" --arm-spec "${HCWDL_UB_ARM_SPEC}" --task "${HCWDL_UB_TASK}"

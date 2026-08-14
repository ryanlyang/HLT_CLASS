#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_unified_balanced_final.py" --foundation-spec "${HCWDL_UB_FOUNDATION_SPEC}" --finalist-lock "${HCWDL_UB_FINALIST_LOCK}" --execution-lock "${HCWDL_UB_EXECUTION_LOCK}" --output "${HCWDL_UB_FINAL_OUTPUT}" --device cuda

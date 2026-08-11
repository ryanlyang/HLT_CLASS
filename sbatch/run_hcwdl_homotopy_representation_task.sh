#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_U_RKD_SPEC:?HCWDL_U_RKD_SPEC is required}"
: "${HCWDL_U_RKD_TASK:?HCWDL_U_RKD_TASK is required}"

source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_homotopy_representation_task.py" \
  --campaign-spec "${HCWDL_U_RKD_SPEC}" \
  --task "${HCWDL_U_RKD_TASK}"

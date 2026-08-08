#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/HLT_Classification}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${HCWDL_SPEC:?HCWDL_SPEC is required}"
: "${HCWDL_TASK:?HCWDL_TASK is required}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_task.py" \
  --campaign-spec "${HCWDL_SPEC}" \
  --task "${HCWDL_TASK}"

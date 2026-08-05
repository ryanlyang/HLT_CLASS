#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/HLT_Classification}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${PMARD_SPEC:?PMARD_SPEC is required}"
: "${PMARD_TASK:?PMARD_TASK is required}"
python -s "${PROJECT_DIR}/scripts/run_pmard_task.py" --campaign-spec "${PMARD_SPEC}" --task "${PMARD_TASK}"

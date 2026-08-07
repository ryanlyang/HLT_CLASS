#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/HLT_Classification}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${PMARD_KD_FOLLOWUP_SPEC:?PMARD_KD_FOLLOWUP_SPEC is required}"
: "${PMARD_KD_FOLLOWUP_TASK:?PMARD_KD_FOLLOWUP_TASK is required}"
exec python -s "${PROJECT_DIR}/scripts/run_pmard_kd_followup_task.py" \
  --followup-spec "${PMARD_KD_FOLLOWUP_SPEC}" \
  --task "${PMARD_KD_FOLLOWUP_TASK}"

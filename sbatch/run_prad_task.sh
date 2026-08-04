#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/HLT_Classification}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${CAMPAIGN_SPEC:?CAMPAIGN_SPEC is required}"
: "${CAMPAIGN_TASK:?CAMPAIGN_TASK is required}"
python -s "${PROJECT_DIR}/scripts/run_prad_task.py" \
  --campaign-spec "${CAMPAIGN_SPEC}" \
  --task "${CAMPAIGN_TASK}"

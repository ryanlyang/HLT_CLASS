#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/HLT_Classification}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${PMARD_T100_SWEEP_SPEC:?PMARD_T100_SWEEP_SPEC is required}"
: "${PMARD_T100_SWEEP_TASK:?PMARD_T100_SWEEP_TASK is required}"
exec python -s "${PROJECT_DIR}/scripts/run_pmard_t100_kd_sweep_task.py" \
  --sweep-spec "${PMARD_T100_SWEEP_SPEC}" \
  --task "${PMARD_T100_SWEEP_TASK}"

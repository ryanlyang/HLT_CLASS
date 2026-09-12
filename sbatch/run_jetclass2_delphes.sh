#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${1:?absolute project directory required}"
SPEC="${2:?canonical campaign spec required}"
TASK="${3:?exact task required}"
export JC2_SITE="${4:?explicit execution site required}"
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" run \
  --spec "${SPEC}" --task "${TASK}" \
  --attempt "${SLURM_JOB_ID:?Slurm job ID required}_${SLURM_RESTART_COUNT:-0}"

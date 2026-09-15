#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${1:?project directory required}"
SPEC="${2:?campaign spec required}"
TASK="${3:?task required}"
export JC2_SITE="${4:?execution site required}"

source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"

python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_salience_mt20.py" run \
  --spec "${SPEC}" \
  --task "${TASK}" \
  --attempt "${SLURM_JOB_ID:?Slurm job required}_${SLURM_RESTART_COUNT:-0}"

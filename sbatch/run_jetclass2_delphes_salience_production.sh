#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?project required}"
export JC2_SITE=sporc_a100
SPEC="${2:?campaign spec required}"
TASK="${3:?task required}"
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_salience_production.py" run \
 --spec "${SPEC}" --task "${TASK}" \
 --attempt "${SLURM_JOB_ID:?job required}_${SLURM_RESTART_COUNT:-0}"

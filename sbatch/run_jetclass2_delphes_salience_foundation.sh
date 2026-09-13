#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?absolute project directory required}"
export JC2_SITE=sporc_a100
SPEC="${2:?readiness spec required}"
TASK="${3:?task required}"
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
ARGS=(run --spec "${SPEC}" --task "${TASK}")
if [[ "${TASK}" == "assign" ]]; then
  ARGS+=(--array-index "${SLURM_ARRAY_TASK_ID:?array index required}")
fi
python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_salience_foundation.py" "${ARGS[@]}"

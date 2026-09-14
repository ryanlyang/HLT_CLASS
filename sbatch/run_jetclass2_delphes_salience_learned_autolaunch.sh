#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${1:?project directory required}"
SPEC="${2:?autolaunch spec required}"
PHASE="${3:?autolaunch phase required}"
export JC2_SITE=sporc_a100

source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"

exec python -s \
  "${PROJECT_DIR}/scripts/jetclass2_delphes_salience_learned_autolaunch.py" run \
  --spec "${SPEC}" \
  --phase "${PHASE}"

#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="${1:?project directory required}"
SPEC="${2:?continuation specification required}"
PHASE="${3:?continuation phase required}"
export JC2_SITE=sporc_a100

source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"

exec python -s \
  "${PROJECT_DIR}/scripts/jetclass2_delphes_dzfix_salience_continuation.py" run \
  --spec "${SPEC}" \
  --phase "${PHASE}"

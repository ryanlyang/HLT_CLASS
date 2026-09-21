#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?absolute project}"
SPEC="${2:?immutable spec}"
MODE="${3:?run or launch-run}"
TASK="${4:?task or phase}"
export JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
case "${MODE}" in
  run) exec python -s "${PROJECT_DIR}/scripts/jetclass2_dzfix_fusion_chain.py" run --spec "${SPEC}" --task "${TASK}" ;;
  launch-run) exec python -s "${PROJECT_DIR}/scripts/jetclass2_dzfix_fusion_chain.py" launch-run --spec "${SPEC}" --phase "${TASK}" ;;
  *) echo "Unknown fusion-chain mode: ${MODE}" >&2; exit 2 ;;
esac

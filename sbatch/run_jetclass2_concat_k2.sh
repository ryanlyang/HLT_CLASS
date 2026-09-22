#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?absolute project}"
SPEC="${2:?immutable spec}"
MODE="${3:?run or launch-run}"
TASK="${4:?task or phase}"
# Resolve only the actual allowed partition; all scientific settings come from
# the immutable spec. The Python worker independently authenticates scontrol.
case "${SLURM_JOB_PARTITION:?Slurm execution partition required}" in
  tier3) export JC2_SITE=sporc_a100 ;;
  debug) export JC2_SITE=sporc_a100_debug ;;
  *) echo "K2 permits only SPORC tier3/debug" >&2; exit 2 ;;
esac
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
case "${MODE}" in
  run) exec python -s "${PROJECT_DIR}/scripts/jetclass2_concat_k2.py" run --spec "${SPEC}" --task "${TASK}" ;;
  launch-run) exec python -s "${PROJECT_DIR}/scripts/jetclass2_concat_k2.py" launch-run --spec "${SPEC}" --phase "${TASK}" ;;
  *) echo "Unknown K2 mode: ${MODE}" >&2; exit 2 ;;
esac

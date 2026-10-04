#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?executor checkout}"
SPEC="${2:?segmented campaign spec}"
TASK="${3:?exact task}"
case "${SLURM_JOB_PARTITION:?}" in
  debug) export JC2_SITE=sporc_a100_debug ;;
  tier3) export JC2_SITE=sporc_a100 ;;
  *) echo 'Segmented K2 permits only debug/tier3' >&2; exit 2 ;;
esac
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
if [[ "${TASK}" == preflight ]]; then
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
fi
exec python -s "${PROJECT_DIR}/scripts/jetclass2_k2_segmented.py" run --spec "${SPEC}" --task "${TASK}"

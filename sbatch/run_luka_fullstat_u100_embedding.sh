#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == /* && "$2" == /* ]]
export PROJECT_DIR="$1"
SPEC="$2"
TASK="$3"
export JC2_SITE=oscar_l40s
unset PYTHONHOME PYTHONPATH LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_u100_embedding.py" run --spec "${SPEC}" --task "${TASK}"

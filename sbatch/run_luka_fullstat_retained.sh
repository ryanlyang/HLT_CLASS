#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == /* && "$2" == /* ]]
export PROJECT_DIR="$1"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_retained.py" run --spec "$2" --task "$3"


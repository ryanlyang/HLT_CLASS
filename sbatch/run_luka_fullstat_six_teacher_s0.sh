#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 && "$1" == /* && "$2" == /* ]]
export PROJECT_DIR="$1"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
export PYTHONFAULTHANDLER=1
exec python -X faulthandler -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_six_teacher_s0.py" run --spec "$2"

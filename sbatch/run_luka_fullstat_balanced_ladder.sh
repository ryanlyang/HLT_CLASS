#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == /* && "$2" == /* && "$3" == train_* ]]
export PROJECT_DIR="$1"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
export PYTHONFAULTHANDLER=1
exec python -X faulthandler -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_balanced_ladder.py" run --spec "$2" --task "$3"

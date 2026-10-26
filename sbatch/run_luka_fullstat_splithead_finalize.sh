#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == /* && "$2" == /* ]]
export PROJECT_DIR="$1"
SPEC="$2"
TASK="$3"
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
exec python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_splithead_finalize.py" run --spec "${SPEC}" --task "${TASK}"

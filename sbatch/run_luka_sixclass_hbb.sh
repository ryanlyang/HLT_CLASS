#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == /* && "$2" == /* && "$3" == /* ]]
AUDIT_PROJECT="$1"; SPEC="$2"; OUT="$3"
export PROJECT_DIR=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_sixclass_balanced_oscar_8a25f126
export ENV_DIR=/oscar/home/rlyang/hlt_classification/environments/new_balanced_oscar_v1
source "${PROJECT_DIR}/sbatch/luka_sixclass_balanced_oscar_common.sh"
exec "$ENV_DIR/bin/python" -X faulthandler -u -B -s \
    "$AUDIT_PROJECT/scripts/reselect_luka_sixclass_hbb.py" --spec "$SPEC" --output "$OUT"

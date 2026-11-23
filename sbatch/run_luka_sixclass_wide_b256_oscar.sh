#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == /* && "$2" == /* ]]
export PROJECT_DIR="$1"
SPEC="$2"; TASK="$3"
ENV_DIR="$(/usr/bin/python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export ENV_DIR
source "${PROJECT_DIR}/sbatch/luka_sixclass_wide_b256_oscar_common.sh"
exec "$ENV_DIR/bin/python" -X faulthandler -u -B -s "${PROJECT_DIR}/scripts/luka_sixclass_wide_b256_oscar.py" run --spec "$SPEC" --task "$TASK"

#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 && "$1" == /* && "$2" == /* ]]
export PROJECT_DIR="$1"
SPEC="$2"; TASK="$3"
ENV_DIR="$(/usr/bin/python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export ENV_DIR
source "$PROJECT_DIR/sbatch/luka_balanced_reconstruction_hydra_common.sh"
exec "$ENV_DIR/bin/python" -X faulthandler -u -B -s "$PROJECT_DIR/scripts/luka_balanced_reconstruction_hydra.py" run --spec "$SPEC" --task "$TASK"

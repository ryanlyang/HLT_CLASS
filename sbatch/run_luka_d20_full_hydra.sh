#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 ]]
export PROJECT_DIR="$1"
SPEC="$2"; TASK="$3"
export ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
source "${PROJECT_DIR}/sbatch/luka_d20_full_hydra_common.sh"
ulimit -c 0
exec "$ENV_DIR/bin/python" -X faulthandler -I -u -B \
    "$PROJECT_DIR/scripts/luka_d20_full_hydra.py" run --spec "$SPEC" --task "$TASK"

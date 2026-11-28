#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 ]]
REPAIR_PROJECT="$1"; REPAIR_SPEC="$2"; TASK="$3"
[[ "$REPAIR_PROJECT" == /* && "$REPAIR_SPEC" == /* ]]
export PROJECT_DIR=/cs/home/rlyang/hlt_u100_pairs/code_d20_full_5af66449
export ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
source "$PROJECT_DIR/sbatch/luka_d20_full_hydra_common.sh"
CLI="$REPAIR_PROJECT/scripts/recover_luka_d20_inputs.py"
case "$TASK" in
    probe|release) exec "$ENV_DIR/bin/python" -I -u -B "$CLI" "$TASK" --repair-spec "$REPAIR_SPEC" ;;
    *) exec "$ENV_DIR/bin/python" -I -u -B "$CLI" run --repair-spec "$REPAIR_SPEC" --task "$TASK" ;;
esac

#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 || $# -eq 4 ]]
REPAIR_PROJECT="$1"; REPAIR_SPEC="$2"; PHASE="$3"
export PROJECT_DIR=/cs/home/rlyang/hlt_u100_pairs/code_balanced_reco_dc5920f0
export ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
source "$PROJECT_DIR/sbatch/luka_balanced_reconstruction_hydra_common.sh"
ARGS=("$PHASE" --repair-spec "$REPAIR_SPEC")
if [[ $# -eq 4 ]]; then ARGS+=(--task "$4"); fi
exec "$ENV_DIR/bin/python" -X faulthandler -I -u -B \
    "$REPAIR_PROJECT/scripts/recover_luka_reconstruction_normalization.py" "${ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 || $# -eq 2 ]]
COMMIT="$1"; [[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="$BASE/code_65952ed3"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
REPAIR_PROJECT="$BASE/code_balanced_norm_${COMMIT:0:8}"
if [[ ! -e "$REPAIR_PROJECT" && ! -L "$REPAIR_PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$REPAIR_PROJECT" "$COMMIT"
fi
test "$(git -C "$REPAIR_PROJECT" rev-parse HEAD)" = "$COMMIT"
ROOT="$BASE/runs/balanced_reco_normalization_${COMMIT:0:8}_r1"
[[ ! -e "$ROOT" && ! -L "$ROOT" && ! -e "$ROOT.launch" && ! -L "$ROOT.launch" ]]
mkdir "$ROOT.launch"
CANDIDATE="$REPAIR_PROJECT/configs/operations/balanced_reco_dc5920f0_normalization.json"
if [[ $# -eq 2 ]]; then CANDIDATE="$2"; fi
test -f "$CANDIDATE"
ARGS=(--commit "$COMMIT" --normalization-json "$CANDIDATE")
nohup bash -s -- "$REPAIR_PROJECT" "$ROOT" "${ARGS[@]}" >"$ROOT.launch/launcher.log" 2>&1 </dev/null <<'WORK' &
set -euo pipefail
PROJECT="$1"; ROOT="$2"; shift 2
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export PROJECT_DIR=/cs/home/rlyang/hlt_u100_pairs/code_balanced_reco_dc5920f0
export ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
source "$PROJECT_DIR/sbatch/luka_balanced_reconstruction_hydra_common.sh"
CLI="$PROJECT/scripts/recover_luka_reconstruction_normalization.py"
"$ENV_DIR/bin/python" -I -u -B "$CLI" create "$@"
"$ENV_DIR/bin/python" -I -u -B "$CLI" queue --repair-spec "$ROOT/repair_spec.json"
"$ENV_DIR/bin/python" -I -u -B "$CLI" queue --repair-spec "$ROOT/repair_spec.json" \
    --execute --authorize AUTHORIZE_BALANCED_RECO_NORMALIZATION_RECOVERY_V1
WORK
printf 'DETACHED normalization recovery PID %s\nLog: %s.launch/launcher.log\n' "$!" "$ROOT"
printf 'Pending original reconstruction/report jobs are retired by exact IDs; running/completed fits are preserved.\n'
exec tail -n 60 -f "$ROOT.launch/launcher.log"

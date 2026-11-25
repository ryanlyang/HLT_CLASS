#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 ]]
COMMIT="$1"; ROOT="$2"; AVAILABLE="$3"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$AVAILABLE" =~ ^[0-9]+$ ]]
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="$BASE/code_65952ed3"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_d20_full_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$COMMIT" ]]
[[ ! -e "$ROOT" && ! -L "$ROOT" && ! -e "$ROOT.launch" && ! -L "$ROOT.launch" ]]
mkdir "$ROOT.launch"
nohup bash -s -- "$PROJECT" "$ROOT" "$COMMIT" "$AVAILABLE" >"$ROOT.launch/launcher.log" 2>&1 </dev/null <<'WORK' &
set -euo pipefail
export PROJECT_DIR="$1"; ROOT="$2"; COMMIT="$3"; AVAILABLE="$4"
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
source "$PROJECT_DIR/sbatch/luka_d20_full_hydra_common.sh"
CLI="$PROJECT_DIR/scripts/luka_d20_full_hydra.py"
"$ENV_DIR/bin/python" -I -u -B "$CLI" create --root "$ROOT" --commit "$COMMIT" --available-gib "$AVAILABLE"
bash "$PROJECT_DIR/scripts/queue_luka_d20_full_hydra.sh" "$ROOT/campaign_spec.json"
WORK
printf 'DETACHED full D20 launcher PID %s\nLog: %s.launch/launcher.log\n' "$!" "$ROOT"
printf 'Run once. Bootstrap qualifies four GPUs before releasing 23 fits in 92 continuation segments.\n'
exec tail -n 60 -f "$ROOT.launch/launcher.log"

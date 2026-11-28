#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 ]]
COMMIT="$1"; ROOT="$2"; shift 2
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ $# -gt 0 ]] || set -- M0HLT_s22_p01
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="$BASE/code_65952ed3"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_d20_input_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
[[ "$(git -C "$PROJECT" rev-parse HEAD)" == "$COMMIT" ]]
[[ ! -e "$ROOT" && ! -L "$ROOT" && ! -e "$ROOT.launch" && ! -L "$ROOT.launch" ]]
mkdir "$ROOT.launch"
nohup bash -s -- "$PROJECT" "$ROOT" "$COMMIT" "$@" >"$ROOT.launch/launcher.log" 2>&1 </dev/null <<'WORK' &
set -euo pipefail
REPAIR_PROJECT="$1"; ROOT="$2"; COMMIT="$3"; shift 3
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export PROJECT_DIR=/cs/home/rlyang/hlt_u100_pairs/code_d20_full_5af66449
export ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
source "$PROJECT_DIR/sbatch/luka_d20_full_hydra_common.sh"
CLI="$REPAIR_PROJECT/scripts/recover_luka_d20_inputs.py"
"$ENV_DIR/bin/python" -I -u -B "$CLI" create --commit "$COMMIT" --root "$ROOT" --tasks "$@"
"$ENV_DIR/bin/python" -I -u -B "$CLI" queue --repair-spec "$ROOT/repair_spec.json"
"$ENV_DIR/bin/python" -I -u -B "$CLI" queue --repair-spec "$ROOT/repair_spec.json" \
    --execute --authorize AUTHORIZE_D20_TARGETED_INPUT_RECOVERY_V1
WORK
printf 'DETACHED targeted D20 input recovery PID %s\nLog: %s.launch/launcher.log\n' "$!" "$ROOT"
printf 'Running science jobs are preserved. CPU reader parity precedes targeted replacements. Run once.\n'
exec tail -n 60 -f "$ROOT.launch/launcher.log"

#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 8 ]] || { echo 'Expected MODE PROJECT ROOT COMMIT FREE_GIB ENVIRONMENT --authorize TOKEN'; exit 2; }
MODE="$1"; shift
PROJECT_DIR="$1"; ROOT="$2"; COMMIT="$3"; QUOTA="$4"; ENV_DIR="$5"
[[ "$6" == --authorize && "$7" == AUTHORIZE_BALANCED_RECONSTRUCTION_18_FOUR_1080TI_V1 ]]
[[ "$PROJECT_DIR" == /* && "$ROOT" == /* && "$ENV_DIR" == /* ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
LAUNCH="${ROOT}.launch"
if [[ "$MODE" == detach || "$MODE" == resume-detach ]]; then
    test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
    test -x "$ENV_DIR/bin/python"
    if [[ "$MODE" == detach ]]; then
        [[ ! -e "$ROOT" && ! -L "$ROOT" && ! -e "$LAUNCH" && ! -L "$LAUNCH" ]]
        mkdir -p "$(dirname "$ROOT")"
        mkdir "$LAUNCH"; NEXT=run
    else
        test -f "$ROOT/campaign_spec.json"
        LAUNCH="$(mktemp -d "${ROOT}.resume-launch.XXXXXX")"; NEXT=resume-run
    fi
    nohup bash "$PROJECT_DIR/scripts/launch_luka_balanced_reconstruction_hydra.sh" "$NEXT" "$@" >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$LAUNCH/pid.txt"
    printf 'DETACHED balanced reconstruction launcher PID %s\nLog: %s/launcher.log\nYou may disconnect now; do not launch twice. Tail is only a view.\n' "$!" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
[[ "$MODE" == run || "$MODE" == resume-run ]]
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export PROJECT_DIR ENV_DIR
source "$PROJECT_DIR/sbatch/luka_balanced_reconstruction_hydra_common.sh"
if [[ "$MODE" == run ]]; then
    "$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_balanced_reconstruction_hydra.py" register \
        --root "$ROOT" --commit "$COMMIT" --environment "$ENV_DIR" --available-gib "$QUOTA"
    "$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_balanced_reconstruction_hydra.py" queue \
        --spec "$ROOT/campaign_spec.json" --execute --authorize AUTHORIZE_BALANCED_RECONSTRUCTION_18_FOUR_1080TI_V1
else
    "$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_balanced_reconstruction_hydra.py" resume \
        --spec "$ROOT/campaign_spec.json" --available-gib "$QUOTA" --authorize AUTHORIZE_BALANCED_RECONSTRUCTION_18_FOUR_1080TI_V1
fi

#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 10 ]] || { echo 'Expected MODE PROJECT ROOT COMMIT AVAILABLE_GIB ENVIRONMENT IMPORT_ROOT BUNDLE_HASH --authorize TOKEN'; exit 2; }
MODE="$1"; shift
PROJECT_DIR="$1"; ROOT="$2"; COMMIT="$3"; QUOTA="$4"; ENV_DIR="$5"; IMPORT="$6"; BUNDLE="$7"
[[ "$8" == --authorize && "$9" == AUTHORIZE_SIXCLASS_TAIL_HYDRA_V1 ]]
[[ "$PROJECT_DIR" == /* && "$ROOT" == /* && "$ENV_DIR" == /* && "$IMPORT" == /* ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ && "$BUNDLE" =~ ^[0-9a-f]{64}$ ]]
LAUNCH="${ROOT}.launch"
if [[ "$MODE" == detach || "$MODE" == resume-detach ]]; then
    test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
    test -f "$IMPORT/transfer_complete.json"
    test -x "$ENV_DIR/bin/python"
    if [[ "$MODE" == detach ]]; then
        [[ ! -e "$ROOT" && ! -L "$ROOT" && ! -e "$LAUNCH" && ! -L "$LAUNCH" ]]
        mkdir -p "$(dirname "$ROOT")"
        mkdir "$LAUNCH"
        NEXT=run
    else
        test -f "$ROOT/campaign_spec.json"
        LAUNCH="$(mktemp -d "${ROOT}.resume-launch.XXXXXX")"
        NEXT=resume-run
    fi
    nohup bash "$PROJECT_DIR/scripts/launch_luka_sixclass_tail_hydra.sh" "$NEXT" "$@" >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    PID=$!
    printf '%s\n' "$PID" >"$LAUNCH/pid.txt"
    printf 'DETACHED Hydra tail launcher PID %s\nStudy: %s\nLog: %s/launcher.log\nYou may disconnect now. Do not launch twice; tail is only a view.\n' "$PID" "$ROOT" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
[[ "$MODE" == run || "$MODE" == resume-run ]]
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export PROJECT_DIR ENV_DIR
source "$PROJECT_DIR/sbatch/luka_sixclass_tail_hydra_common.sh"
if [[ "$MODE" == run ]]; then
    "$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_tail_hydra.py" register \
        --root "$ROOT" --commit "$COMMIT" --environment "$ENV_DIR" --available-gib "$QUOTA" \
        --import-root "$IMPORT" --bundle-hash "$BUNDLE"
    RESUME=()
else
    RESUME=(--resume)
fi
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_tail_hydra.py" queue \
    --spec "$ROOT/campaign_spec.json" --execute --available-gib "$QUOTA" "${RESUME[@]}" \
    --authorize AUTHORIZE_SIXCLASS_TAIL_HYDRA_V1

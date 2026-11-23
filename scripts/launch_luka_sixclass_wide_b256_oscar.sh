#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 9 ]] || { echo 'Expected MODE PROJECT DATASET ROOT COMMIT AVAILABLE_GIB ENVIRONMENT --authorize TOKEN'; exit 2; }
MODE="$1"; shift
PROJECT_DIR="$1"; DATASET="$2"; ROOT="$3"; COMMIT="$4"; QUOTA="$5"; ENV_DIR="$6"
[[ "$7" == --authorize && "$8" == AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1 ]]
[[ "$PROJECT_DIR" == /* && "$DATASET" == /* && "$ROOT" == /* && "$ENV_DIR" == /* ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
LAUNCH="${ROOT}.launch"
if [[ "$MODE" == detach || "$MODE" == resume-detach ]]; then
    test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
    test -f "$DATASET/transfer_complete.json"
    test -x "$ENV_DIR/bin/python"
    if [[ "$MODE" == detach ]]; then
        [[ ! -e "$ROOT" && ! -L "$ROOT" && ! -e "$LAUNCH" && ! -L "$LAUNCH" ]]
        NEXT=run
    else
        test -f "$ROOT/campaign_spec.json"
        # Unique diagnostic folder, not a new scientific identity. Python checks
        # all prior IDs and journals before an explicit continuation.
        LAUNCH="$(mktemp -d "${ROOT}.resume-launch.XXXXXX")"
        NEXT=resume-run
    fi
    if [[ "$MODE" == detach ]]; then mkdir "$LAUNCH"; fi
    nohup bash "$PROJECT_DIR/scripts/launch_luka_sixclass_wide_b256_oscar.sh" "$NEXT" "$@" >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    PID=$!
    printf '%s\n' "$PID" >"$LAUNCH/pid.txt"
    printf 'DETACHED launcher PID %s\nStudy: %s\nLog: %s/launcher.log\nYou may disconnect now. Do not launch twice; tail is only a view.\n' "$PID" "$ROOT" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
[[ "$MODE" == run || "$MODE" == resume-run ]]
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export PROJECT_DIR ENV_DIR
source "$PROJECT_DIR/sbatch/luka_sixclass_wide_b256_oscar_common.sh"
if [[ "$MODE" == run ]]; then
    "$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_wide_b256_oscar.py" register \
        --dataset "$DATASET" --root "$ROOT" --commit "$COMMIT" --environment "$ENV_DIR" --available-gib "$QUOTA"
    RESUME=()
else
    RESUME=(--resume)
fi
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_wide_b256_oscar.py" queue \
    --spec "$ROOT/campaign_spec.json" --execute --available-gib "$QUOTA" "${RESUME[@]}" \
    --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1

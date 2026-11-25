#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 7 ]] || { echo 'Expected MODE PROJECT ROOT COMMIT FREE_HOME_GIB --authorize TOKEN'; exit 2; }
MODE="$1"; PROJECT_DIR="$2"; ROOT="$3"; COMMIT="$4"; QUOTA="$5"
[[ "$6" == --authorize && "$7" == AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1 ]]
[[ "$MODE" == detach || "$MODE" == run || "$MODE" == gradient-detach || "$MODE" == gradient-run ]]
[[ "$PROJECT_DIR" == /* && "$ROOT" == /* && "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
BASE=/oscar/home/rlyang/hlt_classification
ENV_DIR="$BASE/environments/new_balanced_oscar_v1"
DATASET=/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1
PARENT="$BASE/checkpoints/luka_sixclass_wide_b256_oscar_d534fcfe_r1/campaign_spec.json"
CHILD_MODE=run
if [[ "$MODE" == gradient-* ]]; then
    PARENT="$BASE/checkpoints/luka_sixclass_wide_memory_44aa1522_r1/campaign_spec.json"
    CHILD_MODE=gradient-run
fi
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
test -f "$PARENT"
test -x "$ENV_DIR/bin/python"
LAUNCH="${ROOT}.launch"
if [[ "$MODE" == detach || "$MODE" == gradient-detach ]]; then
    [[ ! -e "$ROOT" && ! -L "$ROOT" && ! -e "$LAUNCH" && ! -L "$LAUNCH" ]]
    mkdir "$LAUNCH"
    nohup bash "$PROJECT_DIR/scripts/recover_luka_sixclass_wide_b256_oscar.sh" "$CHILD_MODE" \
        "$PROJECT_DIR" "$ROOT" "$COMMIT" "$QUOTA" --authorize "$7" \
        >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    PID=$!
    printf '%s\n' "$PID" >"$LAUNCH/pid.txt"
    printf 'DETACHED recovery launcher PID %s\nStudy: %s\nLog: %s/launcher.log\nFour unfinished stages only. Do not launch twice; tail is only a view.\n' "$PID" "$ROOT" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export PROJECT_DIR ENV_DIR
source "$PROJECT_DIR/sbatch/luka_sixclass_wide_b256_oscar_common.sh"
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_wide_b256_oscar.py" register \
    --dataset "$DATASET" --root "$ROOT" --commit "$COMMIT" --environment "$ENV_DIR" \
    --available-gib "$QUOTA" --recover "$PARENT"
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_wide_b256_oscar.py" queue \
    --spec "$ROOT/campaign_spec.json" --execute --available-gib "$QUOTA" \
    --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1

#!/usr/bin/env bash
set -euo pipefail
MODE="$1"; shift
[[ "$#" == 7 ]] || { echo 'Expected PROJECT DATASET ROOT COMMIT GPU AVAILABLE_GIB BASELINE_SPEC'; exit 2; }
PROJECT="$1"; DATASET="$2"; ROOT="$3"; COMMIT="$4"; GPU="$5"; QUOTA="$6"; BASELINE="$7"
ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
LAUNCH="${ROOT}.launch"
if [[ "$MODE" == detach || "$MODE" == resume-detach ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -f "$DATASET/transfer_complete.json"
    test -f "$BASELINE"
    if [[ "$MODE" == detach ]]; then
        [[ ! -e "$ROOT" && ! -e "$LAUNCH" ]]
        NEXT=run
    else
        test -f "$ROOT/campaign_spec.json"
        LAUNCH="${ROOT}.resume-launch"
        [[ ! -e "$LAUNCH" ]]
        NEXT=resume-run
    fi
    mkdir "$LAUNCH"
    nohup bash "$PROJECT/scripts/launch_luka_new_balanced_hydra.sh" "$NEXT" "$@" >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    PID=$!
    printf '%s\n' "$PID" >"$LAUNCH/pid.txt"
    printf 'DETACHED launcher PID %s\nLog: %s/launcher.log\nYou may disconnect; registration and submission continue. Do not launch twice.\n' "$PID" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
[[ "$MODE" == run || "$MODE" == resume-run ]]
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH PYTHONHOME
if [[ "$MODE" == run ]]; then
    "$ENV_DIR/bin/python" -u -B -s "$PROJECT/scripts/luka_new_balanced_hydra.py" register \
        --dataset "$DATASET" --root "$ROOT" --commit "$COMMIT" --environment "$ENV_DIR" \
        --gpu "$GPU" --available-gib "$QUOTA" --baseline-spec "$BASELINE"
    RESUME=()
else
    RESUME=(--resume)
fi
"$ENV_DIR/bin/python" -u -B -s "$PROJECT/scripts/luka_new_balanced_hydra.py" queue \
    --spec "$ROOT/campaign_spec.json" --execute --available-gib "$QUOTA" "${RESUME[@]}" \
    --authorize AUTHORIZE_NEW_BALANCED_HYDRA_SIX_V1

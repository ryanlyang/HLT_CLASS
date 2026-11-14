#!/usr/bin/env bash
set -euo pipefail
MODE="$1"; shift
PROJECT="$1"; DATASET="$2"; ROOT="$3"; CACHE="$4"; COMMIT="$5"; GPU="$6"; QUOTA="$7"
ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
LAUNCH="${ROOT}.launch"
if [[ "$MODE" == detach ]]; then
    [[ ! -e "$ROOT" && ! -e "$CACHE" && ! -e "$LAUNCH" ]]
    test -f "$DATASET/transfer_complete.json"
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    mkdir "$LAUNCH"
    nohup bash "$PROJECT/scripts/launch_luka_new_fullstat_hydra.sh" run "$@" >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    PID=$!
    printf '%s\n' "$PID" >"$LAUNCH/pid.txt"
    printf 'DETACHED launcher PID %s\nLog: %s/launcher.log\nYou may disconnect; registration and submission continue. Do not launch twice.\n' "$PID" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
[[ "$MODE" == run ]]
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH PYTHONHOME
"$ENV_DIR/bin/python" -u -B -s "$PROJECT/scripts/luka_new_fullstat_hydra.py" register \
    --dataset "$DATASET" --root "$ROOT" --cache "$CACHE" --commit "$COMMIT" \
    --environment "$ENV_DIR" --gpu "$GPU" --available-gib "$QUOTA"
"$ENV_DIR/bin/python" -u -B -s "$PROJECT/scripts/luka_new_fullstat_hydra.py" queue \
    --spec "$ROOT/campaign_spec.json" --execute --authorize AUTHORIZE_NEW_FULLSTAT_HYDRA_OFFLINE_V1

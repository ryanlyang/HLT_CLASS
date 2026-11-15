#!/usr/bin/env bash
set -euo pipefail
[[ $# == 4 ]] || { echo 'Usage: MODE PROJECT COMMIT AVAILABLE_GIB'; exit 2; }
MODE="$1"; PROJECT="$2"; COMMIT="$3"; QUOTA="$4"
ROOT=/cs/home/rlyang/hlt_u100_pairs/runs/new_balanced_4a61a34c_r1
LAUNCH="${ROOT}.cache-recovery-launch"
RUN_MODE=run
case "$MODE" in
    detach-cleanup|cleanup|continue-cleanup)
        LAUNCH="${ROOT}.cache-cleanup-launch"
        RUN_MODE=cleanup
        ;;
    detach-resume|resume|continue-resume)
        LAUNCH="${ROOT}.post-cache-resume-launch"
        RUN_MODE=resume
        ;;
esac
if [[ "$MODE" == detach || "$MODE" == detach-cleanup || "$MODE" == detach-resume ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    [[ ! -e "$LAUNCH" ]]
    mkdir "$LAUNCH"
    nohup bash "$PROJECT/scripts/launch_luka_hydra_cache_recovery.sh" "$RUN_MODE" "$PROJECT" "$COMMIT" "$QUOTA" \
        >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    PID=$!
    printf '%s\n' "$PID" >"$LAUNCH/pid.txt"
    printf 'DETACHED %s launcher PID %s\nLog: %s/launcher.log\nYou may disconnect now; do not launch twice.\n' "$RUN_MODE" "$PID" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
[[ "$MODE" == run || "$MODE" == continue-run || "$MODE" == cleanup || "$MODE" == continue-cleanup || "$MODE" == resume || "$MODE" == continue-resume ]]
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
ENV_DIR="$(python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$ROOT/campaign_spec.json")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH PYTHONHOME
EXTRA=()
if [[ "$MODE" == continue-run || "$MODE" == continue-cleanup || "$MODE" == continue-resume ]]; then EXTRA=(--continue-existing); fi
ACTION=recover
AUTHORIZATION=AUTHORIZE_HYDRA_BASELINE_CACHE_RETIREMENT_AND_BALANCED_RESUME_V1
if [[ "$RUN_MODE" == cleanup ]]; then
    ACTION=cleanup
    AUTHORIZATION=AUTHORIZE_HYDRA_COMPLETED_BASELINE_CACHE_RETIREMENT_ONLY_V1
elif [[ "$RUN_MODE" == resume ]]; then
    ACTION=resume
    AUTHORIZATION=AUTHORIZE_HYDRA_BALANCED_POST_CLEANUP_RESUME_V1
fi
"$ENV_DIR/bin/python" -u -B -s "$PROJECT/scripts/luka_hydra_cache_retirement.py" "$ACTION" \
    --commit "$COMMIT" --available-gib "$QUOTA" "${EXTRA[@]}" \
    --authorize "$AUTHORIZATION"

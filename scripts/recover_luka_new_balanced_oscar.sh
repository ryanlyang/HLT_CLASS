#!/usr/bin/env bash
# This specific startup-only recovery rebuilds HOME runtime, then registers a fresh execution.
set -euo pipefail
[[ $# -eq 6 ]] || { echo 'Expected MODE COMMIT AVAILABLE_HOME_GIB CONDA_EXE --authorize TOKEN'; exit 2; }
MODE="$1"; COMMIT="$2"; QUOTA="$3"; CONDA_EXE="$4"
[[ "$5" == --authorize && "$6" == AUTHORIZE_NEW_BALANCED_OSCAR_SIX_V1 ]]
[[ "$MODE" == detach || "$MODE" == run ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
(( QUOTA >= 28 ))
BASE=/oscar/home/rlyang/hlt_classification
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
[[ "$PROJECT_DIR" == "$BASE/source/HLT_Classification_new_balanced_oscar_${COMMIT:0:8}" ]]
[[ "$CONDA_EXE" == /* && -x "$CONDA_EXE" ]]
# Use healthy base Python, not the missing scratch stdlib. Resolve before detaching.
BASE_PYTHON="$(dirname "$CONDA_EXE")/python"
test -x "$BASE_PYTHON"
SPEC="$BASE/checkpoints/luka_new_balanced_oscar_f9dfc80b_r1/campaign_spec.json"
ROOT="$BASE/checkpoints/luka_new_balanced_oscar_${COMMIT:0:8}_r1"
LAUNCH="${ROOT}.runtime-launch"
DATASET=/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1
ENV_DIR="$BASE/environments/new_balanced_oscar_v1"
[[ ! -e "$ROOT" && ! -L "$ROOT" ]]
test -f "$SPEC"
test -f "$DATASET/transfer_complete.json"
if [[ "$MODE" == detach ]]; then
    [[ ! -e "$ENV_DIR" && ! -L "$ENV_DIR" && ! -e "${ENV_DIR}.restore" ]]
    [[ ! -e "$LAUNCH" && ! -L "$LAUNCH" ]]
    mkdir "$LAUNCH"
    nohup bash "$PROJECT_DIR/scripts/recover_luka_new_balanced_oscar.sh" run "$COMMIT" "$QUOTA" "$CONDA_EXE" \
        --authorize AUTHORIZE_NEW_BALANCED_OSCAR_SIX_V1 >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    PID=$!
    printf '%s\n' "$PID" >"$LAUNCH/pid.txt"
    printf 'DETACHED runtime recovery PID %s\nLog: %s/launcher.log\nYou may disconnect now; rebuild, verification and submission continue. Do not launch again.\n' "$PID" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
trap 'code=$?; printf "RECOVERY_LAUNCHER_EXIT=%s\n" "$code"' EXIT
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$BASE_PYTHON" -I -u -B "$PROJECT_DIR/scripts/restore_luka_oscar_runtime.py" \
    --spec "$SPEC" --conda "$CONDA_EXE" --available-gib "$QUOTA"
export PROJECT_DIR ENV_DIR
source "$PROJECT_DIR/sbatch/luka_new_balanced_oscar_common.sh"
# New prefix/source means a new immutable spec. Do not rewrite the failed execution.
"$ENV_DIR/bin/python" -I -u -B - "$PROJECT_DIR" "$SPEC" "$LAUNCH" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / 'src'))
from hlt_classification.operations.oscar_runtime_restore import startup_retry_evidence, publish
publish(Path(sys.argv[3]) / 'parent_terminal.json', startup_retry_evidence(Path(sys.argv[2])))
PY
REMAINING="$("$BASE_PYTHON" -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["declared_remaining_result_gib"])' "${ENV_DIR}.restore/complete.json")"
bash "$PROJECT_DIR/scripts/launch_luka_new_balanced_oscar.sh" run \
    "$PROJECT_DIR" "$DATASET" "$ROOT" "$COMMIT" "$REMAINING" "$ENV_DIR" \
    --authorize AUTHORIZE_NEW_BALANCED_OSCAR_SIX_V1

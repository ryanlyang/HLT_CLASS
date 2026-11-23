#!/usr/bin/env bash
# Bootstrap and submit only the new six-class campaign, detached from SSH.
set -euo pipefail
[[ $# -eq 6 ]] || { echo 'Expected MODE COMMIT AVAILABLE_HOME_GIB CONDA_EXE --authorize TOKEN'; exit 2; }
MODE="$1"; COMMIT="$2"; QUOTA="$3"; RUNTIME_CONDA="$4"
[[ "$5" == --authorize && "$6" == AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1 ]]
[[ "$MODE" == detach || "$MODE" == run ]]
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
BASE=/oscar/home/rlyang/hlt_classification
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$(git -C "$PROJECT_DIR" rev-parse HEAD)" = "$COMMIT"
[[ "$PROJECT_DIR" == "$BASE/source/HLT_Classification_sixclass_wide_b256_oscar_${COMMIT:0:8}" ]]
git -C "$PROJECT_DIR" merge-base --is-ancestor "$COMMIT" origin/main
git -C "$PROJECT_DIR" diff --quiet HEAD -- src scripts sbatch tests docs
[[ -z "$(git -C "$PROJECT_DIR" ls-files --others --exclude-standard -- src scripts sbatch tests)" ]]
ROOT="$BASE/checkpoints/luka_sixclass_wide_b256_oscar_${COMMIT:0:8}_r1"
LAUNCH="${ROOT}.bootstrap"
SPEC="$BASE/checkpoints/luka_new_balanced_oscar_f9dfc80b_r1/campaign_spec.json"
ENV_DIR="$BASE/environments/new_balanced_oscar_v1"
DATASET=/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1
test -f "$SPEC"
test -f "$DATASET/transfer_complete.json"
[[ ! -e "$ROOT" && ! -L "$ROOT" && ! -e "${ROOT}.launch" && ! -L "${ROOT}.launch" ]]
if [[ -e "$ENV_DIR" || -L "$ENV_DIR" || -e "${ENV_DIR}.restore" ]]; then
    test -f "${ENV_DIR}.restore/complete.json" || { echo 'STOP: incomplete runtime restoration; preserve and inspect its records.'; exit 1; }
    BOOTSTRAP_PYTHON="$ENV_DIR/bin/python"
    (( QUOTA >= 18 ))
else
    [[ "$RUNTIME_CONDA" == /* && -x "$RUNTIME_CONDA" ]] || { echo 'STOP: supply the healthy base Conda executable as argument 3 to the queue helper.'; exit 1; }
    BOOTSTRAP_PYTHON="$(dirname "$RUNTIME_CONDA")/python"
    (( QUOTA >= 32 ))
fi
test -x "$BOOTSTRAP_PYTHON"
if [[ "$MODE" == detach ]]; then
    [[ ! -e "$LAUNCH" && ! -L "$LAUNCH" ]]
    mkdir "$LAUNCH"
    nohup bash "$PROJECT_DIR/scripts/bootstrap_luka_sixclass_wide_b256_oscar.sh" run "$COMMIT" "$QUOTA" "$RUNTIME_CONDA" \
        --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1 >"$LAUNCH/launcher.log" 2>&1 </dev/null &
    PID=$!
    printf '%s\n' "$PID" >"$LAUNCH/pid.txt"
    printf 'DETACHED six-class launcher PID %s\nStudy: %s\nLog: %s/launcher.log\nYou may disconnect now. Runtime verification, population registration and six-job submission continue. Do not launch twice.\n' "$PID" "$ROOT" "$LAUNCH"
    exec tail -n 60 -f "$LAUNCH/launcher.log"
fi
trap 'code=$?; printf "BOOTSTRAP_EXIT=%s\n" "$code"' EXIT
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
"$BOOTSTRAP_PYTHON" -I -u -B - "$PROJECT_DIR" "$SPEC" "$RUNTIME_CONDA" "$QUOTA" "$LAUNCH" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from hlt_classification.luka_sixclass_wide_b256_oscar.runtime import ensure
from hlt_classification.operations.oscar_runtime_restore import publish
remaining=ensure(Path(sys.argv[2]),Path(sys.argv[1]),sys.argv[3],int(sys.argv[4]))
publish(Path(sys.argv[5])/'runtime_ready.json',dict(contract='LUKA_SIXCLASS_RUNTIME_READY/v1',
    schema_version=1,available_result_gib=remaining,jobs_submitted=0))
PY
export PROJECT_DIR ENV_DIR
source "$PROJECT_DIR/sbatch/luka_sixclass_wide_b256_oscar_common.sh"
"$ENV_DIR/bin/python" -I -u -B - "$PROJECT_DIR" "$SPEC" "$LAUNCH" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1])/'src'))
from hlt_classification.luka_sixclass_wide_b256_oscar.runtime import previous_terminal
from hlt_classification.operations.oscar_runtime_restore import publish
publish(Path(sys.argv[3])/'previous_terminal.json',previous_terminal(Path(sys.argv[2])))
PY
REMAINING="$("$ENV_DIR/bin/python" -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["available_result_gib"])' "$LAUNCH/runtime_ready.json")"
bash "$PROJECT_DIR/scripts/launch_luka_sixclass_wide_b256_oscar.sh" run \
    "$PROJECT_DIR" "$DATASET" "$ROOT" "$COMMIT" "$REMAINING" "$ENV_DIR" \
    --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1

#!/usr/bin/env bash
# Run from a clean, pushed detached worktree on SPORC. No destructive operations.
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: bash queue_luka_embedding_sweep.sh PROJECT ROOT auto|gate|science' >&2; exit 2; }
PROJECT_DIR="$(readlink -f -- "$1")"
ROOT="$2"
MODE="$3"
[[ "${MODE}" == auto || "${MODE}" == gate || "${MODE}" == science ]]
[[ "${ROOT}" = /* ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_embedding_sweep.py"
if [[ "${MODE}" == auto ]]; then
    test -f "${ROOT}/gate_spec.json"
    python -u -B -s "${CLI}" followup --gate "${ROOT}/gate_spec.json"
    read -r -p 'Paste the printed FOLLOWUP_PLAN_HASH (64 hex characters): ' PLAN_HASH </dev/tty
    [[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
    read -r -p 'Queue preflight and authorize all 8 tier3 fits after verified success? Type YES: ' CONFIRM </dev/tty
    [[ "${CONFIRM}" == YES ]] || { echo 'Stopped; no live submission.'; exit 0; }
    LOG="$(mktemp "${ROOT}/auto-followup.XXXXXX.log")"
    nohup python -u -B -s "${CLI}" followup --gate "${ROOT}/gate_spec.json" --execute \
        --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_EMBEDDING_SWEEP_FOLLOWUP_V1 \
        >"${LOG}" 2>&1 </dev/null &
    printf 'Controller PID: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${LOG}" "${LOG}"
    printf 'Wait for ARMED, not just a PID. Ctrl-C stops tail only; controller survives SSH disconnects.\n'
    printf 'Do not also submit science manually or launch a second controller.\n'
    exit 0
fi
if [[ "${MODE}" == gate ]]; then
    SPEC="${ROOT}/gate_spec.json"
else
    test -f "${ROOT}/gate_spec.json"
    SPEC="${ROOT}/science/campaign_spec.json"
    if [[ ! -e "${SPEC}" ]]; then
        python -u -B -s "${CLI}" science --gate "${ROOT}/gate_spec.json"
    fi
fi
test -f "${SPEC}"
python -u -B -s "${CLI}" submit --spec "${SPEC}"
read -r -p 'Paste the printed PLAN_HASH (64 hex characters): ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
COUNT=1
[[ "${MODE}" != science ]] || COUNT=8
read -r -p "Submit these ${COUNT} ${MODE} job(s)? Type YES: " CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || { echo 'Stopped; no live submission.'; exit 0; }
LOG="$(mktemp "${ROOT}/${MODE}-submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${SPEC}" --execute \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_EMBEDDING_SWEEP_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${LOG}" "${LOG}"
printf 'A PID alone is not submission confirmation. Check the log and exact job ID.\n'

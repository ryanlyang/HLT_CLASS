#!/usr/bin/env bash
# Exact dry review, separately confirmed cancellation, then separately confirmed submission.
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: queue_luka_embedding_sweep_debug.sh PROJECT ORIGINAL_GATE_ROOT ALTERNATING_SPEC' >&2; exit 2; }
PROJECT_DIR="$(readlink -f -- "$1")"
ROOT="$(readlink -f -- "$2")"
ALTERNATING="$(readlink -f -- "$3")"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_embedding_sweep_debug.py"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
python -u -B -s "${CLI}" create --original "${ROOT}/science/campaign_spec.json" --expected-commit "${COMMIT}"
SPEC="${ROOT}/science_debug/campaign_spec.json"
python -u -B -s "${CLI}" submit --spec "${SPEC}"
read -r -p 'Paste printed debug PLAN_HASH (64 hex characters): ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
python -B -s -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p["content_hash"]==sys.argv[2] else "STOP: debug PLAN_HASH differs; nothing cancelled")' \
    "${ROOT}/science_debug/submission/command_plan.json" "${PLAN_HASH}"
python -u -B -s "${CLI}" retire --spec "${SPEC}" --alternating "${ALTERNATING}"
read -r -p 'Paste printed RETIRE_PLAN_HASH (64 hex characters): ' RETIRE_HASH </dev/tty
[[ "${RETIRE_HASH}" =~ ^[0-9a-f]{64}$ ]]
printf '\nCancellation preserves files but loses unfinished alternating-gate work.\n'
read -r -p 'Cancel ONLY these exact old sweep/alternating jobs? Type CANCEL REVIEWED JOBS: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == 'CANCEL REVIEWED JOBS' ]] || { echo 'Stopped; no cancellation or submission.'; exit 0; }
python -u -B -s "${CLI}" retire --spec "${SPEC}" --alternating "${ALTERNATING}" --execute \
    --plan-hash "${RETIRE_HASH}" --authorization CANCEL_REVIEWED_LUKA_DEBUG_TRANSFER_JOBS_V1
# Accounting can lag cancellation. This read-only confirmation never retries scancel.
if ! python -u -B -s "${CLI}" confirm --spec "${SPEC}"; then
    echo 'Cancellation requested; accounting not confirmed. DO NOT rerun this helper.'
    echo 'Use the runbook post-cancellation submit block after inspecting accounting.'
    exit 1
fi
read -r -p 'Submit all EIGHT unchanged embedding fits on debug? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || { echo 'Stopped; old jobs cancelled, no replacement submitted.'; exit 0; }
LOG="$(mktemp "${ROOT}/science_debug/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${SPEC}" --execute \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_EMBEDDING_SWEEP_DEBUG_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${LOG}" "${LOG}"
printf 'Confirm eight actual Slurm IDs in the log; the helper PID is not a job ID.\n'

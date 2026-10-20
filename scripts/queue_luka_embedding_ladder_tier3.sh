#!/usr/bin/env bash
# Reuse accepted evidence; no repeated preflight or changes to old artifacts.
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: queue_luka_embedding_ladder_tier3.sh PROJECT GATE_ROOT EMBED|LOGIT' >&2; exit 2; }
PROJECT_DIR="$(readlink -f -- "$1")"
ROOT="$(readlink -f -- "$2")"
ARM="$3"
[[ "${ARM}" == EMBED || "${ARM}" == LOGIT ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_embedding_ladder_tier3.py"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
python -u -B -s "${CLI}" create --gate "${ROOT}/gate_spec.json" --arm "${ARM}" --expected-commit "${COMMIT}"
SPEC="${ROOT}/science_tier3/${ARM}/campaign_spec.json"
python -u -B -s "${CLI}" submit --spec "${SPEC}"
read -r -p 'Paste the printed PLAN_HASH value (64 hex characters): ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p "Submit ONE tier3 ${ARM} five-rung ladder? Type YES: " CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || { echo 'Stopped; no live submission.'; exit 0; }
LOG="$(mktemp "${ROOT}/science_tier3/${ARM}/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${SPEC}" --execute \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_EMBEDDING_LADDER_TIER3_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${LOG}" "${LOG}"
printf 'Confirm the actual Slurm job ID in the log, not just this helper PID.\n'

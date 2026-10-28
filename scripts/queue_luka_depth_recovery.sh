#!/usr/bin/env bash
# Reviewed stop / optional interrupted-attempt archive / queue. No automatic retry.
set -euo pipefail
trap 'echo "Stopped; preserve outputs, claims and journals. Do not blindly retry." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
PARENT="${2:-${MAIN}/checkpoints/luka_depth_9e90b7de_debug_r1/campaign_spec.json}"
ROOT="${MAIN}/checkpoints/luka_depth_recovery_${COMMIT:0:8}_debug_r1"
MODE="${1:-}"
[[ "${MODE}" == stop || "${MODE}" == queue || "${MODE}" == status || "${MODE}" == restart-set8-p3 ]]
test -s "${PARENT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_depth_recovery.py"
if [[ "${MODE}" == status ]]; then
    python -u -B -s "${CLI}" status --parent "${PARENT}"
    exit 0
fi
if [[ "${MODE}" == restart-set8-p3 ]]; then
    mkdir -p -- "${ROOT}"
    python -u -B -s "${CLI}" restart-plan --parent "${PARENT}" \
        --expected-commit "${COMMIT}" --output "${ROOT}/restart_plan.json"
    read -r -p 'Paste the RESTART PLAN_HASH (64 hexadecimal characters): ' HASH </dev/tty
    [[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
    read -r -p 'Archive ONLY the cancelled set8_p3 attempt; authorize an epoch-1 restart later? Type YES: ' CONFIRM </dev/tty
    [[ "${CONFIRM}" == YES ]]
    python -u -B -s "${CLI}" restart --plan "${ROOT}/restart_plan.json" --plan-hash "${HASH}" \
        --authorization ARCHIVE_CANCELLED_SET8_P3_RESTART_ZERO_V2
    printf '\nPartial outputs archived, not deleted; old log retained. No jobs submitted.\n'
    printf 'Now run this helper with queue to review ALL remaining jobs, including the deferred restart.\n'
    exit 0
fi
if [[ "${MODE}" == stop ]]; then
    mkdir -p -- "${ROOT}"
    python -u -B -s "${CLI}" stop-plan --parent "${PARENT}" --output "${ROOT}/stop_plan.json"
    read -r -p 'Paste the STOP PLAN_HASH: ' HASH </dev/tty
    [[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
    read -r -p 'Cancel ONLY listed jobs that are still pending, preserving running jobs? Type YES: ' CONFIRM </dev/tty
    [[ "${CONFIRM}" == YES ]]
    python -u -B -s "${CLI}" stop --plan "${ROOT}/stop_plan.json" --plan-hash "${HASH}"
    printf '\nRunning jobs were preserved. Once they finish, run this helper with queue.\n'
    exit 0
fi
df -h /home/ryreu
read -r -p 'Personally available GiB reserved for remaining fits (at least 8): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
if [[ ! -f "${ROOT}/campaign_spec.json" ]]; then
    python -u -B -s "${CLI}" create --root "${ROOT}" --parent "${PARENT}" \
        --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
fi
python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json"
read -r -p 'Paste the replacement PLAN_HASH: ' HASH </dev/tty
[[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue ONLY remaining fits in two debug lanes; no target rerun/preflight? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json" --execute \
    --plan-hash "${HASH}" --authorization AUTHORIZE_DEPTH_REVIEWED_TWO_LANES_DEBUG_V2 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nRecovery: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"

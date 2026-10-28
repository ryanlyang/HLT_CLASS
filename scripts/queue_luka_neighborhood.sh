#!/usr/bin/env bash
# Exact sixteen-job debug campaign; no old-job mutation or standalone preflight.
set -euo pipefail
trap 'echo "Stopped; preserve outputs and journals. Do not blindly retry." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_neighborhood_${COMMIT:0:8}_debug_r1"
PARENT="${1:-${MAIN}/checkpoints/luka_depth_9e90b7de_debug_r1/campaign_spec.json}"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
test -s "${PARENT}"
test ! -e "${ROOT}"
df -h /home/ryreu
read -r -p 'Personally available persistent GiB reserved for this study (at least 4; actual target count will be checked): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_neighborhood.py"
python -u -B -s "${CLI}" create --root "${ROOT}" --parent "${PARENT}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json"
read -r -p 'Paste the printed 64-character PLAN_HASH value: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue 16 debug jobs: shared preparation, 14 students in two lanes, report; NO separate preflight? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json" --execute \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_NEIGHBORHOOD_14_DEBUG_NO_PREFLIGHT_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nStudy: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
printf 'Helper survives SSH disconnect, not login-node reboot/site termination. Wait for the final 16-job ledger to confirm acceptance. Never blindly resubmit.\n'

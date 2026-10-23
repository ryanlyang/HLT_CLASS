#!/usr/bin/env bash
# Execute from the NEW exact pushed clean worktree, not an old parent checkout.
set -euo pipefail
trap 'echo "Stopped; preserve logs and journals. Do not blindly retry." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_particle_prediction_${COMMIT:0:8}_debug_r1"
REFERENCE="${MAIN}/checkpoints/luka_u100_direct_ed6d0af4_tier3_r1/campaign_spec.json"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
test -s "${REFERENCE}"
test ! -e "${ROOT}"
df -h /home/ryreu
read -r -p 'Personally available persistent GiB reserved for this study (at least 2): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_particle_prediction.py"
python -u -B -s "${CLI}" create --root "${ROOT}" --reference "${REFERENCE}" \
    --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json"
read -r -p 'Paste the printed 64-character PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue debug preflight and BOTH fixed30 fits after verified success? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json" \
    --execute --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_PARTICLE_PREDICTION_DEBUG_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nStudy: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}" "${LOG}"
printf 'Expect three exact job IDs. Fits use afterok; do not submit them again manually.\n'

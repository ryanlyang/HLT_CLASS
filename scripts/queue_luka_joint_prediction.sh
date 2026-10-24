#!/usr/bin/env bash
# Run from the new pushed clean worktree, never from the historical donor.
set -euo pipefail
trap 'echo "Stopped; preserve logs/journals. Do not blindly retry." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_joint_prediction_${COMMIT:0:8}_debug_r1"
TOKEN_STUDY="${MAIN}/checkpoints/luka_particle_prediction_d8dba242_debug_r1/campaign_spec.json"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
test -s "${TOKEN_STUDY}"
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
CLI="${PROJECT_DIR}/scripts/luka_joint_prediction.py"
python -u -B -s "${CLI}" create --root "${ROOT}" --token-study "${TOKEN_STUDY}" \
    --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json"
read -r -p 'Paste the printed 64-character PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue TWO debug fits: CE+4 jet+3 tokens; NO new preflight? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json" \
    --execute --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_JOINT_PREDICTION_DEBUG_NO_PREFLIGHT_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nStudy: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}" "${LOG}"
printf 'Expect lukajp_aligned and lukajp_set. No preflight, controller or dependent follow-up.\n'

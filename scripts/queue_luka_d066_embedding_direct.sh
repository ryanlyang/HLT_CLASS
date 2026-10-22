#!/usr/bin/env bash
# One tier3 allocation: target setup + actual fixed30 training. No new preflight.
set -euo pipefail
trap 'echo "Stopped; preserve logs and journals. Do not blindly retry." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_d066_direct_${COMMIT:0:8}_tier3_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
TEACHER="${MAIN}/checkpoints/luka_attention_94f56025_r1/science/campaign_spec.json"
REFERENCE="${MAIN}/checkpoints/luka_embedding_sweep_007885f7_tier3_r1/science_debug/campaign_spec.json"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
test ! -e "${ROOT}"
test -s "${TEACHER%/campaign_spec.json}/train_LOGIT_D066/selected.pt"
test -s "${REFERENCE%/campaign_spec.json}/w5/receipt.json"
df -h /home/ryreu
read -r -p 'Personally available persistent GiB reserved for this run (at least 2): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_d066_embedding.py"
python -u -B -s "${CLI}" create-direct --root "${ROOT}" --parent "${PARENT}" --teacher "${TEACHER}" \
    --reference "${REFERENCE}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${CLI}" submit-direct --spec "${ROOT}/campaign_spec.json"
read -r -p 'Paste the printed 64-character DIRECT_PLAN_HASH value: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit ONE tier3 target-export + fixed30 training job, with NO new preflight? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/direct-submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit-direct --spec "${ROOT}/campaign_spec.json" \
    --execute --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_D066_EMBEDDING_DIRECT_TIER3_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nStudy: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}" "${LOG}"
printf 'Look for the single direct job ID. No follow-up/preflight job; old holds remain untouched.\n'

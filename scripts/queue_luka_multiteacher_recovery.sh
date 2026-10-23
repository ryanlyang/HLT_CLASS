#!/usr/bin/env bash
# Queue all dependencies together; no login controller or second submission.
set -euo pipefail
trap 'echo "Stopped; preserve logs and journals. Do not blindly retry." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_multi5_recovery_${COMMIT:0:8}_tier3_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
ATTENTION="${MAIN}/checkpoints/luka_attention_94f56025_r1/science/campaign_spec.json"
REFERENCE="${MAIN}/checkpoints/luka_embedding_sweep_007885f7_tier3_r1/science_debug/campaign_spec.json"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
test ! -e "${ROOT}"
for FILE in "${PARENT}" "${ATTENTION}" "${REFERENCE}" \
    "${PARENT%/campaign_spec.json}/train_OFFLINE/selected.pt" \
    "${PARENT%/campaign_spec.json}/train_COARSE_U100/selected.pt"; do
    test -s "${FILE}"
done
df -h /home/ryreu
read -r -p 'Personally available persistent GiB reserved for ALL eight jobs (at least5): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_multiteacher_recovery.py"
python -u -B -s "${CLI}" create --root "${ROOT}" --parent "${PARENT}" --attention "${ATTENTION}" \
    --reference "${REFERENCE}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json"
read -r -p 'Paste the printed64-character PLAN_HASH value: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue TWO replacement teachers, export and FIVE tier3 fits, with NO new preflight? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json" --execute \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_MULTITEACHER_RECOVERY_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nStudy: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}" "${LOG}"
printf 'Wait for all EIGHT accepted IDs. No existing jobs/holds, teachers or Fresh_check are changed.\n'

#!/usr/bin/env bash
# Direct debug; exact dry review, no standalone preflight, no existing-job mutation.
set -euo pipefail
trap 'echo "Stopped; preserve logs/journals. Do not blindly retry." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_depth_${COMMIT:0:8}_debug_r1"
PARENT="${1:-}"
if [[ -z "${PARENT}" ]]; then
    read -r -p 'Full campaign_spec.json path of the COMPLETED twenty-fit auxiliary grid: ' PARENT </dev/tty
fi
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
test -s "${PARENT}"
test ! -e "${ROOT}"
df -h /home/ryreu
read -r -p 'Personally available persistent GiB reserved for this study (at least 8): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_supervision_depth.py"
python -u -B -s "${CLI}" create --root "${ROOT}" --parent "${PARENT}" \
    --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json"
read -r -p 'Paste the printed 64-character PLAN_HASH value: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue 25 debug jobs: 20 students, 2 teachers, targets, report and owned cleanup; no standalone preflight? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json" \
    --execute --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_SUPERVISION_DEPTH_20_STUDENTS_2_TEACHERS_DEBUG_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nStudy: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}" "${LOG}"
printf 'Expect 25 lukasd_* jobs, ordered afterok. No automatic retries or cancellation. Do not resubmit after disconnect.\n'

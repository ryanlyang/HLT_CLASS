#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve outputs, claims and journals. Do not blindly retry." >&2' ERR
[[ $# == 2 ]]
REFERENCE="$1"
QUOTA="$2"
[[ "${REFERENCE}" == /* && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_supcon5_${COMMIT:0:8}_debug_r1"
SPEC="${ROOT}/campaign_spec.json"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" fetch origin main
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
test ! -e "${ROOT}"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT}/scripts/luka_supcon5.py" create --root "${ROOT}" \
    --reference "${REFERENCE}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${PROJECT}/scripts/luka_supcon5.py" review --spec "${SPEC}"
PLAN_HASH="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/review.json")"
printf 'Launching reviewed SupCon-5 campaign: one setup, one teacher, one student, one report.\n'
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${PROJECT}/scripts/luka_supcon5.py" dispatch --spec "${SPEC}" \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_SUPCON5_DEBUG_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Dispatcher PID: %s\nCampaign: %s\nLog: %s\n' "$!" "${ROOT}" "${LOG}"
printf 'The SPORC submit-host dispatcher survives logout, but not a host/process failure.\n'
printf 'Watch the log until it confirms the setup job ID. Do not run this command twice.\n'

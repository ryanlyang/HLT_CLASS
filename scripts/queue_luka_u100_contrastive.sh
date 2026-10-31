#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve outputs, claims and journals. Do not blindly retry." >&2' ERR
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_u100_contrastive_${COMMIT:0:8}_debug_r1"
SPEC="${ROOT}/campaign_spec.json"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" fetch origin main
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
test ! -e "${ROOT}"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
REFERENCE="${MAIN}/checkpoints/luka_u100_direct_ed6d0af4_tier3_r1/campaign_spec.json"
test -s "${PARENT}"; test -s "${REFERENCE}"
df -h /home/ryreu
read -r -p 'Personally available persistent GiB (at least 4; campaign budget stays 4): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
python -u -B -s "${PROJECT}/scripts/luka_u100_contrastive.py" create --root "${ROOT}" \
    --parent "${PARENT}" --reference "${REFERENCE}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${PROJECT}/scripts/luka_u100_contrastive.py" review --spec "${SPEC}"
read -r -p 'Paste the printed 64-character PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
printf 'One shared setup automatically followed by 20 measured fits in two debug lanes and a report. No teacher retraining or separate preflight.\n'
read -r -p 'Submit this reviewed automatic campaign? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${PROJECT}/scripts/luka_u100_contrastive.py" dispatch --spec "${SPEC}" \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_U100_CONTRASTIVE_DEBUG_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Dispatcher PID: %s\nCampaign: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
printf 'The detached dispatcher survives logout. It submits the fits only after setup passes and within the reviewed resource envelope.\n'

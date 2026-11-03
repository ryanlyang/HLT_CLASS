#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve original and new outputs/journals. Do not blindly retry." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
PARENT=/home/ryreu/atlas/HLT_Classification/checkpoints/luka_u100_contrastive_431341db_debug_r1/campaign_spec.json
if [[ $# == 2 && "$1" == --parent ]]; then PARENT="$2"; elif [[ $# != 0 ]]; then exit 2; fi
[[ "${PARENT}" == /* ]]
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
git -C "${PROJECT_DIR}" fetch origin main
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
[[ -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)" ]]
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_u100_contrastive_serial_${COMMIT:0:8}_debug_r1"
[[ ! -e "${ROOT}" ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
df -h /home/ryreu
read -r -p 'Personally available persistent GiB (at least 8; campaign budget stays 8): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[0-9]+$ && "${QUOTA}" -ge 8 ]]
python -u -B -s "${PROJECT_DIR}/scripts/luka_u100_contrastive_serial.py" create \
    --root "${ROOT}" --parent "${PARENT}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
python -u -B -s "${PROJECT_DIR}/scripts/luka_u100_contrastive_serial.py" review --request "${ROOT}/request.json"
EXPECTED="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/review.json")"
read -r -p 'Paste the reviewed PLAN_HASH: ' REVIEWED </dev/tty
[[ "${REVIEWED}" == "${EXPECTED}" ]]
read -r -p 'Replace ONLY pending old campaign jobs, preserve running/completed fits, then queue remaining fits in ONE debug lane? Type YES: ' ANSWER </dev/tty
[[ "${ANSWER}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup "${CONDA_PREFIX}/bin/python" -u -B -s "${PROJECT_DIR}/scripts/luka_u100_contrastive_serial.py" dispatch \
    --request "${ROOT}/request.json" --plan-hash "${REVIEWED}" \
    --authorization AUTHORIZE_LUKA_CONTRASTIVE_SERIAL_DEBUG_V1 </dev/null >"${LOG}" 2>&1 &
printf 'Detached transition PID: %s\nCampaign: %s\nWatch with: tail -f "%s"\n' "$!" "${ROOT}" "${LOG}"

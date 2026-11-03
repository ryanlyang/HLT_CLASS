#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve outputs, claims and journals. Do not blindly retry." >&2' ERR
[[ $# == 1 ]]
BASE="$(cd -- "$1" && pwd -P)"
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
PYTHON="${ENV_DIR}/bin/python"
BUNDLE="${BASE}/bundle_65952ed3"
ROOT="${BASE}/runs/pair_hydra_${COMMIT:0:8}_r1"
SPEC="${ROOT}/campaign_spec.json"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" fetch origin main
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
test -s "${BASE}/environment_pair_hydra_v1/environment-check.json"
test -s "${BUNDLE}/extraction_receipt.json"
test ! -e "${ROOT}"
mkdir -p -- "${BASE}/runs"
export VIRTUAL_ENV="${ENV_DIR}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
df -h "${BASE}"
read -r -p 'Personally available GiB for outputs AFTER environment and extraction (at least 8): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" create --root "${ROOT}" \
    --bundle "${BUNDLE}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" review --spec "${SPEC}"
read -r -p 'Paste the printed PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
printf 'Hydra gpus/default/normal: one shared setup, four fresh fits C/A/D/B in two L40 lanes, report.\n'
read -r -p 'Submit this reviewed automatic campaign? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup "${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" dispatch --spec "${SPEC}" \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_U100_PAIR_HYDRA_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Dispatcher PID: %s\nCampaign: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
printf 'The detached dispatcher survives logout; successful setup automatically releases the four fits.\n'

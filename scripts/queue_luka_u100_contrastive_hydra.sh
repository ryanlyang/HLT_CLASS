#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve campaign outputs and exact submission journals." >&2' ERR
[[ $# == 3 ]]
BASE="$(cd -- "$1" && pwd -P)"
DELTA="$(cd -- "$2" && pwd -P)"
DELTA_HASH="$3"
[[ "${DELTA_HASH}" =~ ^[0-9a-f]{64}$ ]]
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
PYTHON="${ENV_DIR}/bin/python"
ROOT="${BASE}/runs/contrast_hydra_${COMMIT:0:8}_${DELTA_HASH:0:8}_r1"
SPEC="${ROOT}/campaign_spec.json"
PRECISION_SPEC="${BASE}/runs/pair_precision_ad55d3cd_r1/campaign_spec.json"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" fetch origin main
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
test -s "${BASE}/environment_pair_hydra_v1/environment-check.json"
test -s "${PRECISION_SPEC}"
test ! -e "${ROOT}"
mkdir -p -- "${BASE}/runs"
export VIRTUAL_ENV="${ENV_DIR}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
df -h "${BASE}"
read -r -p 'Personally available additional output GiB (at least 8, separate from other campaigns): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_contrastive_hydra.py" create \
    --root "${ROOT}" --bundle "${BASE}/bundle_65952ed3" --delta "${DELTA}" --delta-hash "${DELTA_HASH}" \
    --precision-spec "${PRECISION_SPEC}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_contrastive_hydra.py" review --spec "${SPEC}"
PLAN_HASH="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/review.json")"
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup "${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_contrastive_hydra.py" dispatch --spec "${SPEC}" \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_CONTRASTIVE_HYDRA_INDEPENDENT_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Dispatcher PID: %s\nCampaign: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
printf 'Successful setup submits all remaining fits together, with no student dependencies or concurrency cap.\n'

#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve campaign outputs and exact submission journals." >&2' ERR
[[ $# -ge 3 && $# -le 6 ]]
GPU_PROFILE="${5:-l40}"
[[ "${GPU_PROFILE}" == l40 || "${GPU_PROFILE}" == a6000 ]]
MODE="${6:-all}"
[[ "${MODE}" == all || "${MODE}" == setup-only ]]
DISPATCH_OPTIONS=()
if [[ "${MODE}" == setup-only ]]; then DISPATCH_OPTIONS+=(--setup-only); fi
BASE="$(cd -- "$1" && pwd -P)"
CONTROL_SPEC="$2"
ENCODER="$3"
ENCODER_HASH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ENCODER}/manifest.json")"
[[ "${ENCODER_HASH}" =~ ^[0-9a-f]{64}$ ]]
[[ "${CONTROL_SPEC}" == /* ]]
DELTA="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["delta_root"])' "${CONTROL_SPEC}")"
DELTA_HASH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["delta_hash"])' "${CONTROL_SPEC}")"
[[ "${DELTA_HASH}" =~ ^[0-9a-f]{64}$ ]]
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
PYTHON="${ENV_DIR}/bin/python"
ROOT="${BASE}/runs/cross_message_${COMMIT:0:8}_${ENCODER_HASH:0:8}_r1"
if [[ "${GPU_PROFILE}" == a6000 ]]; then
    ROOT="${BASE}/runs/cross_message_${COMMIT:0:8}_${ENCODER_HASH:0:8}_a6000_r1"
fi
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
if [[ $# -ge 4 ]]; then
    QUOTA="$4"
else
    read -r -p 'Personally available additional output GiB (at least 8, separate from other campaigns): ' QUOTA </dev/tty
fi
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" create \
    --root "${ROOT}" --bundle "${BASE}/bundle_65952ed3" --delta "${DELTA}" --delta-hash "${DELTA_HASH}" \
    --encoder "${ENCODER}" --encoder-hash "${ENCODER_HASH}" --precision-spec "${PRECISION_SPEC}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}" --gpu-profile "${GPU_PROFILE}"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" review --spec "${SPEC}"
PLAN_HASH="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/review.json")"
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
command -v setsid >/dev/null
nohup setsid "${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_cross_message_hydra.py" dispatch --spec "${SPEC}" \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_CROSS_MESSAGE_HYDRA_V1 "${DISPATCH_OPTIONS[@]}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Dispatcher PID: %s\nCampaign: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
printf 'Resume submission if the watcher stops: use the finish command in docs/LUKA_CROSS_MESSAGE_HYDRA_RUNBOOK.md.\n'
if [[ "${MODE}" == setup-only ]]; then
    printf 'Setup only. After success, finish releases the 2 teachers + 12 students; see the runbook.\n'
else
    printf 'Successful shared setup queues 2 oracles + 12 students; no lanes or concurrency cap.\n'
fi

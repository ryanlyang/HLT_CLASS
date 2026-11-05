#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve CE-only outputs and submission journals." >&2' ERR
[[ $# -ge 1 && $# -le 2 ]]
PARENT_SPEC="$1"
QUOTA="${2:-2}"
[[ "${PARENT_SPEC}" == /* && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
PARENT_ROOT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' "${PARENT_SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${PARENT_SPEC}")"
PARENT_HASH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${PARENT_SPEC}")"
ROOT="$(dirname -- "${PARENT_ROOT}")/cross_message_ce_${COMMIT:0:8}_${PARENT_HASH:0:8}_r1"
SPEC="${ROOT}/campaign_spec.json"
PYTHON="${ENV_DIR}/bin/python"
[[ "${ENV_DIR}" == /* && "${PARENT_HASH}" =~ ^[0-9a-f]{64}$ ]]
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" fetch origin main
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
test ! -e "${ROOT}"
export VIRTUAL_ENV="${ENV_DIR}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_cross_message_ce_hydra.py" create \
    --root "${ROOT}" --parent-spec "${PARENT_SPEC}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_cross_message_ce_hydra.py" review --spec "${SPEC}"
PLAN_HASH="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/command_plan.json")"
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
command -v setsid >/dev/null
nohup setsid "${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_cross_message_ce_hydra.py" submit --spec "${SPEC}" \
    --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_CROSS_MESSAGE_CE_HYDRA_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nCampaign: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
printf 'After both exact job IDs are recorded, Slurm owns both independent runs. Parent jobs are unchanged.\n'

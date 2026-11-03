#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve diagnostic outputs and submission journal. Do not blindly retry." >&2' ERR
[[ $# == 1 ]]
BASE="$(cd -- "$1" && pwd -P)"
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
PYTHON="${ENV_DIR}/bin/python"
BUNDLE="${BASE}/bundle_65952ed3"
ROOT="${BASE}/runs/pair_precision_${COMMIT:0:8}_r1"
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
printf 'One Hydra inference diagnostic: 1 L40, 6 CPUs, 120000 MiB, one hour. No training or automatic science release.\n'
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" precision-create \
    --root "${ROOT}" --bundle "${BUNDLE}" --expected-commit "${COMMIT}"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" precision-review --spec "${SPEC}"
PLAN_HASH="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/command_plan.json")"
"${PYTHON}" -u -B -s "${PROJECT}/scripts/luka_u100_pair_hydra.py" precision-submit \
    --spec "${SPEC}" --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_HYDRA_PRECISION_DIAGNOSTIC_V1
JOB="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["jobs"]["precision"])' "${ROOT}/submission/submission_ledger.json")"
printf 'Diagnostic accepted as %s; safe to log off.\nLog: %s/precision-%s.out\n' "${JOB}" "${ROOT}" "${JOB}"
printf 'After completion, display saved results:\n'
printf 'env PYTHONNOUSERSITE=1 LD_LIBRARY_PATH="%s/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "%s" -B -s "%s/scripts/luka_u100_pair_hydra.py" precision-results --spec "%s"\n' "${ENV_DIR}" "${PYTHON}" "${PROJECT}" "${SPEC}"

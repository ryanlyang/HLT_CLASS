#!/usr/bin/env bash
# Dry preparation by default; reviewed execution detaches automatically.
set -euo pipefail
if [[ $# -ne 2 && $# -ne 4 ]]; then
  echo 'Usage: queue_cms2jc2_proxy_recovery.sh STUDY_SPEC COMMIT [--execute REVIEWED_RECOVERY_PLAN_HASH]' >&2
  exit 2
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
STUDY="$1"
COMMIT="$2"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" == "${COMMIT}" ]]
test -f "${STUDY}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
CLI="${PROJECT_DIR}/scripts/cms2jc2_proxy_recovery.py"
if [[ $# -eq 2 ]]; then
  python -u -s "${CLI}" prepare --study "${STUDY}" --source-commit "${COMMIT}"
else
  [[ "$3" == '--execute' && "$4" =~ ^[0-9a-f]{64}$ ]]
  ROOT="$(dirname -- "${STUDY}")"
  test -f "${ROOT}/recovery/recovery_plan.json"
  LOG="$(mktemp "${ROOT}/recovery/controller.XXXXXX.log")"
  nohup bash -c '
    result=0
    "$@" || result=$?
    printf "\nRECOVERY_HELPER_EXIT=%s\n" "${result}"
    exit "${result}"
  ' _ "${CONDA_PREFIX}/bin/python" -u -s "${CLI}" run --study "${STUDY}" \
      --plan-hash "$4" --authorize-sealed-test-materialization \
      >"${LOG}" 2>&1 </dev/null &
  printf 'Recovery controller PID: %s\nLog: %s\n' "$!" "${LOG}"
  printf 'Watch with:\ntail -f "%s"\n' "${LOG}"
fi

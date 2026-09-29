#!/usr/bin/env bash
# No implicit production, next-stage submission, cancellation or partition migration.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect the log before retrying." >&2' ERR
if [ "$#" -lt 8 ]; then
  echo "Usage: bash $0 COMMIT JOINT_SPEC INVENTORY PROFILE DATA_ROOT NEW_ROOT debug|tier3 gate|screen [--execute --reviewed-plan-hash HASH]" >&2
  exit 2
fi
COMMIT="$1" DONOR="$2" INVENTORY="$3" PROFILE="$4" DATA_ROOT="$5" ROOT="$6" PARTITION="$7" STAGE="$8"
shift 8
EXECUTE=0
REVIEWED=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --reviewed-plan-hash) REVIEWED="${2:?Missing reviewed hash}"; shift 2 ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
done
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${PARTITION}" == debug || "${PARTITION}" == tier3 ]]
[[ "${STAGE}" == gate || "${STAGE}" == screen ]]
if [ "${EXECUTE}" = 1 ]; then [[ "${REVIEWED}" =~ ^[0-9a-f]{64}$ ]]; fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT_DIR}/src"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
SPEC="${ROOT}/stages/generation_${STAGE}_r1/stage_spec.json"
phase() {
  local child status=0 label="$1"
  shift
  printf '\n=== %s ===\n' "${label}"
  "$@" <&0 &
  child=$!
  while kill -0 "${child}" 2>/dev/null; do
    sleep 15
    if kill -0 "${child}" 2>/dev/null; then printf '%s: still working (PID %s)\n' "${label}" "${child}"; fi
  done
  wait "${child}" || status=$?
  return "${status}"
}
if [ ! -e "${ROOT}" ]; then
  test "${STAGE}" = gate
  phase "Freeze TRAIN-only generation study" python -u -s "${CLI}" create-generation-benchmark \
    --parent-spec "${DONOR}" --inventory "${INVENTORY}" --profile "${PROFILE}" --data-root "${DATA_ROOT}" \
    --root "${ROOT}" --project-dir "${PROJECT_DIR}" --source-commit "${COMMIT}" --partition "${PARTITION}"
fi
phase "Check existing request identity" python -u -s - \
  "${ROOT}/stages/generation_gate_r1/stage_spec.json" "${DONOR}" "${INVENTORY}" "${PROFILE}" \
  "${DATA_ROOT}" "${PROJECT_DIR}" "${COMMIT}" "${PARTITION}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import generation_benchmark_campaign as c
from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response.dev_data import file_ref
s = c.validate_stage(load_json(sys.argv[1]))
if (s['donor'] != file_ref(sys.argv[2]) or s['inventory'] != file_ref(sys.argv[3])
    or s['profile'] != file_ref(sys.argv[4]) or Path(s['data_root']).resolve() != Path(sys.argv[5]).resolve()
    or Path(s['project_dir']).resolve() != Path(sys.argv[6]).resolve()
    or s['source']['commit'] != sys.argv[7] or s['site']['partition'] != sys.argv[8]):
    raise SystemExit('STOP: existing study differs from requested inputs/source/partition')
print('10,000 frozen training jets; no validation/test/native-HLT reads or production generation.')
PY
if [ "${STAGE}" = screen ] && [ ! -f "${SPEC}" ]; then
  phase "Require measured gate; freeze screen" python -u -s "${CLI}" advance-generation-benchmark \
    --parent-spec "${ROOT}/stages/generation_gate_r1/stage_spec.json"
fi
phase "Exact dry plan" python -u -s "${CLI}" dry-run --spec "${SPEC}"
if [ "${EXECUTE}" = 0 ]; then
  echo "Dry review only. Repeat with --execute --reviewed-plan-hash HASH after reviewing this exact plan."
  exit 0
fi
phase "Submit exact reviewed stage" python -u -s "${CLI}" submit --spec "${SPEC}" --execute \
  --reviewed-plan-hash "${REVIEWED}" --authorization-phrase "AUTHORIZE CMS2JC2 GENERATION_${STAGE^^} EXACT PLAN"
printf '\nQUEUED: %s\nNo other jobs changed; no automatic followup.\n' "${SPEC}"

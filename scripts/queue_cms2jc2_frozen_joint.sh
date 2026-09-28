#!/usr/bin/env bash
# Source-pinned frozen confirmation. No cancellation, mutation, or next-stage loop.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect this log before retrying." >&2' ERR
if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
  echo "Usage: bash $0 COMMIT COMPLETED_JOINT_SPEC NEW_ROOT auto|debug|tier3 gate|confirm [--assert-untouched] [--execute]" >&2
  exit 2
fi
EXPECTED_COMMIT="$1" PARENT_SPEC="$2" ROOT="$3" PARTITION="$4" STAGE="$5"
shift 5
EXECUTE=0
UNTOUCHED=0
for option in "$@"; do
  case "${option}" in
    --execute) EXECUTE=1 ;;
    --assert-untouched) UNTOUCHED=1 ;;
    *) echo "Unknown option ${option}" >&2; exit 2 ;;
  esac
done
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${PARTITION}" == auto || "${PARTITION}" == debug || "${PARTITION}" == tier3 ]]
[[ "${STAGE}" == gate || "${STAGE}" == confirm ]]
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${PROJECT_DIR}/src"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${EXPECTED_COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${EXPECTED_COMMIT}" origin/main
test -f "${PARENT_SPEC}"
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
SPEC="${ROOT}/stages/frozen_${STAGE}_r1/stage_spec.json"

run_phase() {
  local label="$1" child status=0
  shift
  printf '\n=== %s ===\n' "${label}"
  "$@" <&0 &
  child=$!
  while kill -0 "${child}" 2>/dev/null; do
    sleep 15
    if kill -0 "${child}" 2>/dev/null; then
      printf '%s: authenticating/working (PID %s)\n' "${label}" "${child}"
    fi
  done
  wait "${child}" || status=$?
  return "${status}"
}
if [ ! -e "${ROOT}" ]; then
  test "${STAGE}" = gate
  test "${UNTOUCHED}" = 1 || { echo "Explicit --assert-untouched required for reserved response_confirm population." >&2; exit 2; }
  if [ "${PARTITION}" = auto ]; then
    # stdout is only the selected name; full read-only evidence goes to this log.
    PARTITION="$(python -u -s - <<'PY'
import sys
from hlt_classification.cms2jc2_response.frozen_joint_queue import probe, render
result = probe()
print(render(result), file=sys.stderr, flush=True)
if result['recommended_partition'] is None:
    raise SystemExit('No usable estimate. Choose debug or tier3 explicitly; no root/jobs created.')
print(result['recommended_partition'])
PY
    )"
  fi
  run_phase "Freeze candidate, checks and metadata-only membership" python -u -s "${CLI}" create-frozen-joint \
    --parent-spec "${PARENT_SPEC}" --project-dir "${PROJECT_DIR}" --source-commit "${EXPECTED_COMMIT}" \
    --root "${ROOT}" --partition "${PARTITION}" --assert-untouched
else
  FROZEN_PARTITION="$(python -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["site"]["partition"])' "${ROOT}/study_spec.json")"
  if [ "${PARTITION}" = auto ]; then PARTITION="${FROZEN_PARTITION}"; fi
  test "${PARTITION}" = "${FROZEN_PARTITION}"
fi
# Validate request identity BEFORE any advance or live submission.
run_phase "Authenticate existing gate against requested source and parent" python -u -s - \
  "${ROOT}/stages/frozen_gate_r1/stage_spec.json" "${PARENT_SPEC}" "${PROJECT_DIR}" "${EXPECTED_COMMIT}" "${PARTITION}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import frozen_joint_campaign as c
from hlt_classification.cms2jc2_response.frozen_joint_metrics import MIN_GROUPS
from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response.dev_data import file_ref
spec = load_json(sys.argv[1])
study = c.validate_stage(spec)
if (spec['parent_spec'] != file_ref(sys.argv[2]) or Path(study['project_dir']).resolve() != Path(sys.argv[3]).resolve()
        or study['source']['commit'] != sys.argv[4] or study['site']['partition'] != sys.argv[5]):
    raise SystemExit('Requested source/parent/partition differs; no action.')
print('Frozen JOINT; all reserved confirmation jets:', spec['membership']['jets'])
print('Source files:', len(spec['membership']['files']), 'evaluation shards:', len(spec['membership']['shards']))
if len(spec['membership']['files']) < MIN_GROUPS:
    print('Warning: too few independent source files for a supported bootstrap conclusion. Large jet counts alone do not fix this.')
print('Partition:', study['site']['partition'], '; final-test/JetClass2 transfer not authorized.')
PY
if [ ! -f "${SPEC}" ] && [ "${STAGE}" = confirm ]; then
  run_phase "Advance real acceptance; lock confirmation access" python -u -s "${CLI}" advance-frozen-joint \
    --parent-spec "${ROOT}/stages/frozen_gate_r1/stage_spec.json"
fi
run_phase "Exact dry plan" python -u -s "${CLI}" dry-run --spec "${SPEC}"
if [ "${EXECUTE}" = 0 ]; then
  echo "Dry review complete. No jobs submitted. Repeat with --execute only after review."
  exit 0
fi
PLAN_HASH="$(python -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
  "${ROOT}/stages/frozen_${STAGE}_r1/command_plan.json")"
run_phase "Explicit reviewed submission" python -u -s "${CLI}" submit --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 FROZEN_${STAGE^^} EXACT PLAN" --reviewed-plan-hash "${PLAN_HASH}"
printf '\nQUEUED: %s\nNo other jobs changed. No automatic next stage.\n' "${SPEC}"

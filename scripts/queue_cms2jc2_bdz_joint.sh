#!/usr/bin/env bash
# New isolated repair gate/comparison; never cancel or modify another study.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect the log. Do not delete roots or retry blindly." >&2' ERR
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  echo "Usage: bash $0 COMMIT COMPLETED_AUDIT_SPEC NEW_ROOT debug|tier3 gate|compare [--execute]" >&2
  exit 2
fi
EXPECTED_COMMIT="$1"
PARENT_SPEC="$2"
ROOT="$3"
PARTITION="$4"
STAGE="$5"
MODE="${6:-}"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${PARTITION}" == debug || "${PARTITION}" == tier3 ]]
[[ "${STAGE}" == gate || "${STAGE}" == compare ]]
[[ -z "${MODE}" || "${MODE}" == --execute ]]
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
SPEC="${ROOT}/stages/joint_${STAGE}_r1/stage_spec.json"

# Keep stdin attached (including Python here-docs), preserve child exit status,
# and give feedback during expensive immutable ancestor authentication.
run_phase() {
  local label="$1" task_pid status=0
  shift
  printf '\n=== %s ===\n' "${label}"
  "$@" <&0 &
  task_pid=$!
  while kill -0 "${task_pid}" 2>/dev/null; do
    sleep 15
    if kill -0 "${task_pid}" 2>/dev/null; then
      printf '%s: still working (PID %s)\n' "${label}" "${task_pid}"
    fi
  done
  wait "${task_pid}" || status=$?
  return "${status}"
}
if [ ! -e "${ROOT}" ]; then
  test "${STAGE}" = gate
  run_phase "Create source-bound joint gate" python -s "${CLI}" create-bdz-joint \
    --parent-spec "${PARENT_SPEC}" --project-dir "${PROJECT_DIR}" \
    --source-commit "${EXPECTED_COMMIT}" --root "${ROOT}" --partition "${PARTITION}"
elif [ ! -f "${SPEC}" ] && [ "${STAGE}" = compare ]; then
  run_phase "Check existing gate before creating comparison" python -s - \
    "${ROOT}/stages/joint_gate_r1/stage_spec.json" "${PARENT_SPEC}" "${PROJECT_DIR}" "${EXPECTED_COMMIT}" "${PARTITION}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import bdz_joint_campaign as joint
from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response.dev_data import file_ref
gate = load_json(sys.argv[1])
study = joint.validate_stage(gate)
assert gate["parent_spec"] == file_ref(sys.argv[2])
assert Path(study["project_dir"]).resolve() == Path(sys.argv[3]).resolve()
assert study["source"]["commit"] == sys.argv[4]
assert study["site"]["partition"] == sys.argv[5]
PY
  run_phase "Advance completed gate" python -s "${CLI}" advance-bdz-joint \
    --parent-spec "${ROOT}/stages/joint_gate_r1/stage_spec.json"
fi
test -f "${SPEC}"
run_phase "Validate exact request" python -s - "${SPEC}" "${PARENT_SPEC}" "${PROJECT_DIR}" "${EXPECTED_COMMIT}" "${PARTITION}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import bdz_joint_campaign as joint, dev_campaign as dev
from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response.dev_data import file_ref
spec = load_json(sys.argv[1])
study = joint.validate_stage(spec)
assert joint.gate(spec)["parent_spec"] == file_ref(sys.argv[2]), "Different completed audit"
assert Path(study["project_dir"]).resolve() == Path(sys.argv[3]).resolve()
assert study["source"]["commit"] == sys.argv[4]
assert study["site"]["partition"] == sys.argv[5]
plan = dev.command_plan(spec, study)
assert plan["gpus"] == 0
assert spec["protocol"]["calibration_jets"] == 4000
assert spec["protocol"]["comparison_jets"] == 10000
print("PASS: five frozen candidates; development only; CPU-only", sys.argv[5], flush=True)
for task in spec["tasks"]:
    print(f"{task['task_id']:<16} CPUs={task['cpus']:2} GiB={task['memory_gib']:3} "
          f"hours={task['hours']} dependencies={task['depends_on']}", flush=True)
PY
run_phase "Dry plan" python -s "${CLI}" dry-run --spec "${SPEC}"
if [ "${MODE}" != --execute ]; then
  echo "Dry review complete. No jobs submitted. Repeat with --execute after review."
  exit 0
fi
PLAN_HASH="$(python -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
    "${ROOT}/stages/joint_${STAGE}_r1/command_plan.json")"
run_phase "Explicit live submission" python -s "${CLI}" submit --spec "${SPEC}" --execute \
    --authorization-phrase "AUTHORIZE CMS2JC2 JOINT_${STAGE^^} EXACT PLAN" --reviewed-plan-hash "${PLAN_HASH}"
printf '\nQueued stage: %s\nResults: python -s "%s" bdz-joint-results --spec "%s"\n' "${STAGE}" "${CLI}" "${SPEC}"
echo "No further stage is submitted automatically."

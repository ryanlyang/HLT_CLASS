#!/usr/bin/env bash
# Fresh frozen tracking audit. No old job cancellation or artifact mutation.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect the error. Do not delete roots or retry blindly." >&2' ERR
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash $0 COMMIT COMPLETED_BDZ_STAGE_SPEC NEW_ROOT [--execute]" >&2
  exit 2
fi
EXPECTED_COMMIT="$1"
PARENT_SPEC="$2"
ROOT="$3"
MODE="${4:-}"
if [ -n "${MODE}" ] && [ "${MODE}" != "--execute" ]; then
  echo "Unknown mode: ${MODE}" >&2
  exit 2
fi
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
SPEC="${ROOT}/stages/bdz_audit_r1/stage_spec.json"
PLAN="${ROOT}/stages/bdz_audit_r1/command_plan.json"
if [ ! -e "${ROOT}" ]; then
  python -s "${CLI}" create-bdz-audit --parent-spec "${PARENT_SPEC}" \
    --project-dir "${PROJECT_DIR}" --source-commit "${EXPECTED_COMMIT}" --root "${ROOT}" >/dev/null
fi
test -f "${SPEC}"
python -s "${CLI}" dry-run --spec "${SPEC}" >/dev/null
python -s - "${SPEC}" "${PARENT_SPEC}" "${PROJECT_DIR}" "${EXPECTED_COMMIT}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import bdz_audit_campaign as audit, dev_campaign as dev
from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response.dev_data import file_ref
spec = load_json(sys.argv[1])
study = audit.validate_stage(spec)
assert spec["parent_spec"] == file_ref(sys.argv[2]), "Different parent; stop"
assert Path(study["project_dir"]).resolve() == Path(sys.argv[3]).resolve()
assert study["source"]["commit"] == sys.argv[4]
plan = dev.command_plan(spec, study)
assert plan["gpus"] == 0 and len(plan["commands"]) == 6
assert all("--partition=tier3" in row["argv"] for row in plan["commands"])
assert spec["protocol"]["jets"] == 10000
print("PASS: frozen 10,000-jet replay; CPU-only tier3; no refit, cancellation or confirmation.")
for task in spec["tasks"]:
    print(f"{task['task_id']:<15} CPUs={task['cpus']:2} RAM={task['memory_gib']:3} GiB "
          f"limit={task['hours']}h dependencies={task['depends_on']}")
print("Frozen historical choice:", spec["reuse"]["historical_choice"])
PY
if [ "${MODE}" != "--execute" ]; then
  echo "Dry review complete. No jobs submitted. Repeat with --execute after review."
  exit 0
fi
PLAN_HASH="$(python -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${PLAN}")"
python -s "${CLI}" submit --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 BDZ_AUDIT EXACT PLAN" --reviewed-plan-hash "${PLAN_HASH}"
python -s "${CLI}" monitor --spec "${SPEC}"
printf '\nResults: python -s "%s" bdz-audit-results --spec "%s"\n' "${CLI}" "${SPEC}"

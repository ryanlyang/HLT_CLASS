#!/usr/bin/env bash
# Fresh tier3 replacement of the exact B_DZ comparison; default is read-only
# with respect to Slurm. --execute explicitly authorizes retirement/submission.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect the error. Do not delete roots or retry blindly." >&2' ERR

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash $0 COMMIT SUBJECT_STAGE_SPEC NEW_ROOT [--execute]" >&2
  exit 2
fi
EXPECTED_COMMIT="$1"
SUBJECT_SPEC="$2"
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
test -f "${SUBJECT_SPEC}"
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
SPEC="${ROOT}/stages/bdz_compare_tier3_r1/stage_spec.json"
PLAN="${ROOT}/stages/bdz_compare_tier3_r1/command_plan.json"

if [ ! -e "${ROOT}" ]; then
  python -s "${CLI}" create-bdz-tier3 \
    --parent-spec "${SUBJECT_SPEC}" --project-dir "${PROJECT_DIR}" \
    --source-commit "${EXPECTED_COMMIT}" --root "${ROOT}" >/dev/null
fi
test -f "${SPEC}"
python -s "${CLI}" dry-run --spec "${SPEC}" >/dev/null

python -s - "${SPEC}" "${SUBJECT_SPEC}" "${PROJECT_DIR}" "${EXPECTED_COMMIT}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import bdz_tier3 as migration, dev_campaign as dev
from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response.dev_data import file_ref

spec = load_json(sys.argv[1])
study = migration.validate_stage(spec)
assert spec["subject_spec"] == file_ref(sys.argv[2]), "Different subject; stop"
assert Path(study["project_dir"]).resolve() == Path(sys.argv[3]).resolve()
assert study["source"]["commit"] == sys.argv[4]
plan = dev.command_plan(spec, study)
assert plan["gpus"] == 0 and len(plan["commands"]) == 5
assert all("--partition=tier3" in row["argv"] for row in plan["commands"])
assert spec["protocol"]["comparison_jets"] == 10000
assert spec["protocol"]["candidates"] == ["B_DZ", "TRACK_HALF", "TRACK_FULL"]
print("PASS: same frozen calibration; 10,000 development jets; CPU-only tier3.")
for task in spec["tasks"]:
    print(f"{task['task_id']:<14} CPUs={task['cpus']:2} RAM={task['memory_gib']:3} GiB "
          f"limit={task['hours']}h dependencies={task['depends_on']}")
print("Exact original jobs:", spec["migration"]["subject_jobs"])
PY

python -s "${CLI}" retire-bdz-tier3 --spec "${SPEC}"
if [ "${MODE}" != "--execute" ]; then
  echo "Dry review complete. No jobs cancelled or submitted. Repeat with --execute after review."
  exit 0
fi
PLAN_HASH="$(python -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${PLAN}")"
python -s "${CLI}" retire-bdz-tier3 --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 BDZ TIER3 REPLACEMENT" \
  --reviewed-plan-hash "${PLAN_HASH}"
python -s "${CLI}" submit --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 BDZ_COMPARE EXACT PLAN" \
  --reviewed-plan-hash "${PLAN_HASH}"
python -s "${CLI}" monitor --spec "${SPEC}"
printf '\nStudy root: %s\nResults: python -s "%s" bdz-results --spec "%s"\n' "${ROOT}" "${CLI}" "${SPEC}"

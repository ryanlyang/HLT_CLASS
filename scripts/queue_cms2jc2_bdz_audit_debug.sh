#!/usr/bin/env bash
# Fresh debug replacement; no old worktree/artifact edits. Dry review by default.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect the error. Do not delete roots or retry blindly." >&2' ERR
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash $0 COMMIT SUBJECT_AUDIT_SPEC NEW_ROOT [--execute]" >&2
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

run_phase() {
  local label="$1"
  shift
  printf '\nCMS2JC2-AUDIT-DEBUG phase=%s starting\n' "${label}" >&2
  "$@" <&0 &
  local child=$!
  while kill -0 "${child}" 2>/dev/null; do
    sleep 15
    if kill -0 "${child}" 2>/dev/null; then
      printf 'CMS2JC2-AUDIT-DEBUG phase=%s still_running pid=%s\n' "${label}" "${child}" >&2
    fi
  done
  local result=0
  wait "${child}" || result=$?
  printf 'CMS2JC2-AUDIT-DEBUG phase=%s exit=%s\n' "${label}" "${result}" >&2
  return "${result}"
}

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
test -f "${SUBJECT_SPEC}"
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
SPEC="${ROOT}/stages/bdz_audit_debug_r1/stage_spec.json"
PLAN="${ROOT}/stages/bdz_audit_debug_r1/command_plan.json"
if [ ! -e "${ROOT}" ]; then
  run_phase create python -u -s "${CLI}" create-bdz-audit-debug \
    --parent-spec "${SUBJECT_SPEC}" --project-dir "${PROJECT_DIR}" \
    --source-commit "${EXPECTED_COMMIT}" --root "${ROOT}"
fi
test -f "${SPEC}"
run_phase review python -u -s - "${SPEC}" "${SUBJECT_SPEC}" "${PROJECT_DIR}" "${EXPECTED_COMMIT}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import bdz_audit_debug as debug, dev_submission as submission
from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response.dev_data import file_ref
spec = load_json(sys.argv[1])
study = load_json(spec["study"]["path"])
assert spec["subject_spec"] == file_ref(sys.argv[2]), "Different subject; stop"
assert Path(study["project_dir"]).resolve() == Path(sys.argv[3]).resolve()
assert study["source"]["commit"] == sys.argv[4]
plan = submission.submit(spec)
assert len(plan["commands"]) == 5 and plan["gpus"] == 0
assert all("--partition=debug" in row["argv"] for row in plan["commands"])
assert spec["protocol"]["jets"] == 10000
print("PASS: unchanged frozen audit; original acceptance reused; five CPU-only debug jobs.", flush=True)
for t in spec["tasks"]:
    print(f"{t['task_id']:<14} CPUs={t['cpus']:2} GiB={t['memory_gib']:3} hours={t['hours']} dependencies={t['depends_on']}")
print("PRESERVE acceptance:", spec["migration"]["subject_jobs"]["ba_acceptance"])
print("Exact old replacement targets:", {t: spec["migration"]["subject_jobs"][t] for t in debug.TARGETS})
PY
run_phase preview python -u -s "${CLI}" retire-bdz-audit-debug --spec "${SPEC}"
if [ "${MODE}" != "--execute" ]; then
  echo "Dry review complete. No jobs cancelled or submitted. Repeat with --execute after review."
  exit 0
fi
PLAN_HASH="$(python -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${PLAN}")"
run_phase retire python -u -s "${CLI}" retire-bdz-audit-debug --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 BDZ AUDIT DEBUG REPLACEMENT" --reviewed-plan-hash "${PLAN_HASH}"
run_phase submit python -u -s "${CLI}" submit --spec "${SPEC}" --execute \
  --authorization-phrase "AUTHORIZE CMS2JC2 BDZ_AUDIT EXACT PLAN" --reviewed-plan-hash "${PLAN_HASH}"
run_phase monitor python -u -s "${CLI}" monitor --spec "${SPEC}"
printf '\nResults: python -s "%s" bdz-audit-results --spec "%s"\n' "${CLI}" "${SPEC}"

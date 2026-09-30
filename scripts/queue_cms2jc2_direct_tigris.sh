#!/usr/bin/env bash
# One explicit Tigris stage only. No SPORC jobs, copying or cancellation.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect the log before retrying." >&2' ERR
if [ "$#" -lt 3 ]; then
  echo "Usage: bash $0 COMMIT GEN_GATE_SPEC ROOT [--phase gate|screen] [--execute --reviewed-plan-hash HASH]" >&2
  exit 2
fi
COMMIT="$1" PARENT="$2" ROOT="$3"
shift 3
PHASE=gate EXECUTE=0 REVIEWED=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --phase) PHASE="${2:?missing phase}"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    --reviewed-plan-hash) REVIEWED="${2:?missing hash}"; shift 2 ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
done
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
case "${PHASE}" in gate|screen) ;; *) exit 2 ;; esac
if [ "${EXECUTE}" = 1 ]; then [[ "${REVIEWED}" =~ ^[0-9a-f]{64}$ ]]; fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
SPEC="${ROOT}/stages/tigris_direct_${PHASE}_r1/stage_spec.json"
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
  test "${PHASE}" = gate
  test "${EXECUTE}" = 0
  phase "Authenticate existing SPORC gate and pin Tigris study" python -s "${CLI}" create-direct-tigris \
    --parent-spec "${PARENT}" --root "${ROOT}" --project-dir "${PROJECT_DIR}" --source-commit "${COMMIT}"
fi
phase "Check exact Tigris study and donor" python -s - "${ROOT}" "${PARENT}" "${PROJECT_DIR}" "${COMMIT}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import generation_direct_campaign as c
from hlt_classification.cms2jc2_response.contracts import load_json
from hlt_classification.cms2jc2_response.dev_data import checked_file, file_ref
s = load_json(Path(sys.argv[1])/'study_spec.json')
c.validate_study(s)
evidence = load_json(checked_file(s['imported']))
if (evidence['donor'] != file_ref(sys.argv[2]) or Path(s['project_dir']).resolve() != Path(sys.argv[3]).resolve()
        or s['source']['commit'] != sys.argv[4]):
    raise SystemExit('Wrong source/donor for direct Tigris study')
print('TRAIN only; unchanged JOINT; separate Tigris environment; no SPORC submission.')
PY
if [ ! -e "${SPEC}" ]; then
  test "${PHASE}" = screen
  test "${EXECUTE}" = 0
  phase "Create measured 16/36/72-CPU screen" python -s "${CLI}" advance-direct-tigris \
    --parent-spec "${ROOT}/stages/tigris_direct_gate_r1/stage_spec.json"
fi
phase "Exact dry plan" python -s "${CLI}" dry-run --spec "${SPEC}"
if [ "${EXECUTE}" = 0 ]; then
  echo "Dry review only. Repeat with --execute --reviewed-plan-hash HASH after review."
  exit 0
fi
UPPER="$(printf '%s' "${PHASE}" | tr '[:lower:]' '[:upper:]')"
phase "Submit reviewed Tigris stage" python -s "${CLI}" submit --spec "${SPEC}" --execute \
  --reviewed-plan-hash "${REVIEWED}" --authorization-phrase "AUTHORIZE CMS2JC2 TIGRIS_DIRECT_${UPPER} EXACT PLAN"
printf '\nQUEUED: %s\nNo other jobs changed; no automatic followup.\n' "${SPEC}"

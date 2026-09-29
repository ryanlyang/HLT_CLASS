#!/usr/bin/env bash
# Operates only on an already-created, immutable PORT_STAGE. Never creates or
# advances another stage, cancels jobs, fetches code, copies data or deletes files.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect the log before retrying." >&2' ERR
if [ "$#" -lt 2 ]; then
  echo "Usage: bash $0 COMMIT STAGE_SPEC [--execute --reviewed-plan-hash HASH]" >&2
  exit 2
fi
COMMIT="$1" SPEC="$2"
shift 2
EXECUTE=0 REVIEWED=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1; shift ;;
    --reviewed-plan-hash) REVIEWED="${2:?missing hash}"; shift 2 ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
done
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
if [ "${EXECUTE}" = 1 ]; then [[ "${REVIEWED}" =~ ^[0-9a-f]{64}$ ]]; fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
# Site is read using the login's stdlib Python before importing science libraries.
SITE="$(python3 - "${SPEC}" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
assert s['contract'] == 'CMS2JC2_RESPONSE_PORT_STAGE/v1'
v = json.loads(Path(s['study']['path']).read_text())
print(v['site']['partition'])
PY
)"
case "${SITE}" in
  tigris)
    source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
    conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris ;;
  debug|tier3)
    source /home/ryreu/miniconda3/etc/profile.d/conda.sh
    conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc ;;
  *) exit 2 ;;
esac
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
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
phase "Check exact source and portable stage" python -s - "${SPEC}" "${PROJECT_DIR}" "${COMMIT}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.cms2jc2_response import generation_portable_campaign as c
from hlt_classification.cms2jc2_response.contracts import load_json
s = c.validate_stage(load_json(sys.argv[1]))
if Path(s['project_dir']).resolve() != Path(sys.argv[2]).resolve() or s['source']['commit'] != sys.argv[3]:
    raise SystemExit('Wrong source for portable study')
print('Validated separate training-only engineering stage.')
PY
phase "Exact dry plan" python -s "${CLI}" dry-run --spec "${SPEC}"
if [ "${EXECUTE}" = 0 ]; then
  echo "Dry review only. Repeat with --execute --reviewed-plan-hash HASH after review."
  exit 0
fi
STAGE="$(python -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["stage"].upper())' "${SPEC}")"
phase "Submit reviewed portable stage" python -s "${CLI}" submit --spec "${SPEC}" --execute \
  --reviewed-plan-hash "${REVIEWED}" --authorization-phrase "AUTHORIZE CMS2JC2 ${STAGE} EXACT PLAN"
printf '\nQUEUED: %s\nOther campaigns untouched; no automatic followup.\n' "${SPEC}"

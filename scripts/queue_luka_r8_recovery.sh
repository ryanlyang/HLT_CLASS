#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { printf 'Usage: bash queue_luka_r8_recovery.sh /absolute/recovery_spec.json\n' >&2; exit 2; }
SPEC="$1"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
CLI="${PROJECT}/scripts/luka_r8_recovery.py"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" review --spec "$SPEC"
PLAN="$(dirname -- "$SPEC")/submission/command_plan.json"
HASH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "$PLAN")"
AUTH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["authorization"])' "$PLAN")"
printf '\nReview hash: %s\nTwo replacement fits (C3/R14) and an afterany snapshot report. Existing jobs and setup stay intact.\n' "$HASH"
read -r -p 'Authorize this exact recovery plan? Type YES: ' ANSWER </dev/tty
[[ "$ANSWER" == YES ]] || { printf 'Not submitted; preserve the review.\n'; exit 1; }
"${ENV_DIR}/bin/python" -u -B -s "$CLI" submit --spec "$SPEC" --plan-hash "$HASH" --authorization "$AUTH"

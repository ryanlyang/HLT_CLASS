#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { printf 'Usage: bash queue_luka_r24_diagnostic_hydra.sh /absolute/campaign_spec.json\n' >&2; exit 2; }
SPEC="$1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PROJECT_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
CLI="${PROJECT_DIR}/scripts/luka_r24_diagnostic_hydra.py"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" review --spec "$SPEC"
PLAN="$(dirname -- "$SPEC")/submission_bootstrap/command_plan.json"
HASH="$("${ENV_DIR}/bin/python" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "$PLAN")"
AUTH="$("${ENV_DIR}/bin/python" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["authorization"])' "$PLAN")"
printf '\nReviewed plan: %s\nSubmitting setup + durable release for the lanes registered in this spec.\n' "$HASH"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" submit --spec "$SPEC" --plan-hash "$HASH" --authorization "$AUTH"

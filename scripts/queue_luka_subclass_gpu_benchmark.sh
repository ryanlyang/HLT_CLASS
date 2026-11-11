#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { printf 'Usage: bash queue_luka_subclass_gpu_benchmark.sh /absolute/benchmark_spec.json\n' >&2; exit 2; }
SPEC="$1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PROJECT_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CLI="${PROJECT_DIR}/scripts/luka_subclass_gpu_benchmark.py"
"${ENV_DIR}/bin/python" -B -s "$CLI" review --spec "$SPEC"
PLAN="$(dirname -- "$SPEC")/submission/command_plan.json"
HASH="$("${ENV_DIR}/bin/python" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "$PLAN")"
AUTH="$("${ENV_DIR}/bin/python" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["authorization"])' "$PLAN")"
printf '\nReview hash: %s\nOne two-GTX1080Ti job, at most two hours. No scientific campaign release.\n' "$HASH"
read -r -p 'Submit this benchmark? Type YES: ' ANSWER </dev/tty
[[ "$ANSWER" == YES ]] || { printf 'Not submitted; preserve the registration.\n'; exit 1; }
"${ENV_DIR}/bin/python" -B -s "$CLI" submit --spec "$SPEC" --plan-hash "$HASH" --authorization "$AUTH"

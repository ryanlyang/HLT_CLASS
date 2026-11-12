#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]] || { printf 'Usage: bash queue_luka_embedding_bridge_hydra.sh /absolute/campaign_spec.json\n' >&2; exit 2; }
SPEC="$1"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PROJECT_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
CLI="${PROJECT_DIR}/scripts/luka_embedding_bridge_hydra.py"
"${ENV_DIR}/bin/python" -B -s "$CLI" review --spec "$SPEC"
PLAN="$(dirname -- "$SPEC")/submission_bootstrap/command_plan.json"
HASH="$("${ENV_DIR}/bin/python" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "$PLAN")"
AUTH="$("${ENV_DIR}/bin/python" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["authorization"])' "$PLAN")"
printf '\nReview hash: %s\n' "$HASH"
printf 'Setup and TP: one RTX A6000. Other 29 fits: two GTX1080Ti. FP32/global batch128.\n'
printf 'Completed Pascal qualification is reused; new A6000 setup gates automatic release and CPU report.\n'
read -r -p 'Authorize this exact bounded campaign? Type YES: ' ANSWER </dev/tty
[[ "$ANSWER" == YES ]] || { printf 'Not submitted. Preserve the review.\n'; exit 1; }
"${ENV_DIR}/bin/python" -B -s "$CLI" submit --spec "$SPEC" --plan-hash "$HASH" --authorization "$AUTH"

#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 || ( "$2" != setup && "$2" != bootstrap && "$2" != science ) ]]; then
    printf 'Usage: bash queue_luka_next_supervision_hydra.sh /absolute/campaign_spec.json setup|bootstrap|science\n' >&2
    exit 2
fi
SPEC="$1"
STAGE="$2"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PROJECT_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_DIR}/src"
PYTHON="${ENV_DIR}/bin/python"
CLI="${PROJECT_DIR}/scripts/luka_next_supervision_hydra.py"
"${PYTHON}" -B -s "${CLI}" review --spec "${SPEC}" --stage "${STAGE}"
PLAN="$(dirname -- "${SPEC}")/submission_${STAGE}/command_plan.json"
HASH="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${PLAN}")"
AUTHORIZATION="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["authorization"])' "${PLAN}")"
printf '\nReview hash: %s\n' "${HASH}"
printf 'Scope: the exact group in the reviewed plan. Science jobs have no dependencies, holds, lanes or priority adjustments.\n'
if [[ "$STAGE" == bootstrap ]]; then
    printf 'This authorizes setup plus durable CPU release of every fit in the reviewed plan after setup passes. No second confirmation; safe to log off after submission completes.\n'
fi
read -r -p "Authorize this exact ${STAGE} plan? Type YES: " ANSWER
[[ "${ANSWER}" == YES ]] || { printf 'Not submitted. Preserve this review.\n'; exit 1; }
"${PYTHON}" -B -s "${CLI}" submit --spec "${SPEC}" --stage "${STAGE}" --plan-hash "${HASH}" --authorization "${AUTHORIZATION}"

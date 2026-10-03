#!/usr/bin/env bash
# Run with bash, never source. No implicit creation, cancellation, or stage jump.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash queue_cms_fullsim_feature_ladder.sh SPEC gate|science [REVIEWED_PLAN_HASH]" >&2
  exit 2
fi
SPEC=$1
STAGE=$2
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PROJECT_DIR JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
ARGS=(submit --spec "${SPEC}" --stage "${STAGE}")
if [[ $# -eq 3 ]]; then
  [[ $3 =~ ^[0-9a-f]{64}$ ]] || { echo 'Use the actual 64-character reviewed plan hash.' >&2; exit 2; }
  ARGS+=(--execute --reviewed-plan-hash "$3" --authorization-phrase 'AUTHORIZE CMS FULLSIM FEATURE LADDER EXACT PLAN')
fi
python -u -s "${PROJECT_DIR}/scripts/cms_fullsim_feature_ladder.py" "${ARGS[@]}"

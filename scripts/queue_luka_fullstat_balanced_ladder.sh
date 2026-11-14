#!/usr/bin/env bash
# Manual dry/exact-plan path. Use the detached launcher instead for unattended submission.
set -euo pipefail
[[ $# -eq 4 || $# -eq 6 ]] || { echo 'Usage: COMMIT FRESH_ROOT PARENT_SPEC QUOTA_GIB [--execute PLAN_HASH]' >&2; exit 2; }
COMMIT="$1"; ROOT="$2"; PARENT="$3"; QUOTA="$4"
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${PARENT}" == /* && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
CLI="${PROJECT_DIR}/scripts/luka_fullstat_balanced_ladder.py"
if [[ $# -eq 4 ]]; then
    [[ ! -e "${ROOT}" && ! -L "${ROOT}" ]]
    python -u -B -s "${CLI}" create --root "${ROOT}" --parent-spec "${PARENT}" --expected-commit "${COMMIT}" --result-quota-gib "${QUOTA}"
    exec python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json"
fi
[[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
python -B -s -c 'import sys; from hlt_classification.operations.fullstat_pt_metadata import load_json; s=load_json(sys.argv[1]); assert s["source_commit"]==sys.argv[2] and s["inputs"]["refs"][0]["path"]==sys.argv[3] and s["available_result_bytes"]==int(sys.argv[4])*2**30' "${ROOT}/campaign_spec.json" "${COMMIT}" "${PARENT}" "${QUOTA}"
exec python -u -B -s "${CLI}" submit --spec "${ROOT}/campaign_spec.json" --execute --plan-hash "$6" --authorization AUTHORIZE_FULLSTAT_BALANCED_LADDER_V1

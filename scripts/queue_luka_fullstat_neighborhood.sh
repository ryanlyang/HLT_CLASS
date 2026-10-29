#!/usr/bin/env bash
# Manual dry/live alternative to the detached automatic launcher.
set -euo pipefail
[[ $# -eq 6 || $# -eq 8 ]] || { echo 'Usage: COMMIT ROOT PARENT_SPEC RESULT_QUOTA_GIB TARGET_ROOT TARGET_QUOTA_GIB [--execute HASH]' >&2; exit 2; }
COMMIT="$1"; ROOT="$2"; PARENT="$3"; QUOTA="$4"; TARGET="$5"; TARGET_QUOTA="$6"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${TARGET}" == /* && "${PARENT}" == /* ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${TARGET_QUOTA}" =~ ^[1-9][0-9]*$ ]]
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
CLI="${PROJECT_DIR}/scripts/luka_fullstat_neighborhood.py"
SPEC="${ROOT}/campaign_spec.json"
if [[ -f "${SPEC}" ]]; then
    python -B -s -c 'import sys; from pathlib import Path; from hlt_classification.data.cache_contracts import load_json; s=load_json(sys.argv[1]); assert s["source_commit"]==sys.argv[2] and Path(s["reuse"]["parent_spec"]["path"]).resolve()==Path(sys.argv[3]).resolve() and s["available_result_bytes"]==int(sys.argv[4])*2**30 and Path(s["target_root"]).resolve()==Path(sys.argv[5]).resolve() and s["available_target_bytes"]==int(sys.argv[6])*2**30, "Different execution requested"' "${SPEC}" "${COMMIT}" "${PARENT}" "${QUOTA}" "${TARGET}" "${TARGET_QUOTA}"
else
    [[ $# -eq 6 ]]
    test ! -e "${ROOT}"
    python -u -B -s "${CLI}" create --root "${ROOT}" --previous-spec "${PARENT}" --expected-commit "${COMMIT}" --result-quota-gib "${QUOTA}" --target-root "${TARGET}" --target-quota-gib "${TARGET_QUOTA}"
fi
ARGS=(submit --spec "${SPEC}")
if [[ $# -eq 8 ]]; then
    [[ "$7" == --execute && "$8" =~ ^[0-9a-f]{64}$ ]]
    ARGS+=(--execute --plan-hash "$8" --authorization AUTHORIZE_FULLSTAT_NEIGHBORHOOD_V1)
fi
exec python -u -B -s "${CLI}" "${ARGS[@]}"

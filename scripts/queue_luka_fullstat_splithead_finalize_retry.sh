#!/usr/bin/env bash
# Explicit successor; never delete/replace failed submission claims or ledgers.
set -euo pipefail
if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo 'Usage: helper COMMIT NEW_ROOT FAILED_FINALIZER_SPEC RESULT_QUOTA_GIB [--execute HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
PREVIOUS="$3"
QUOTA="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${PREVIOUS}" == /* ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
CLI="${PROJECT_DIR}/scripts/luka_fullstat_splithead_finalize.py"
SPEC="${ROOT}/campaign_spec.json"
if [[ -f "${SPEC}" ]]; then
    python -B -s -c 'import sys; from pathlib import Path; from hlt_classification.data.cache_contracts import load_json; s=load_json(sys.argv[1]); assert s["source_commit"]==sys.argv[2] and Path(s["retry"]["failed_spec"]["path"]).resolve()==Path(sys.argv[3]).resolve() and s["available_result_bytes"]==int(sys.argv[4])*2**30, "Different retry requested"' "${SPEC}" "${COMMIT}" "${PREVIOUS}" "${QUOTA}"
elif [[ $# -eq 4 ]]; then
    test ! -e "${ROOT}"
    python -u -B -s "${CLI}" create-retry --root "${ROOT}" --previous-spec "${PREVIOUS}" \
        --expected-commit "${COMMIT}" --result-quota-gib "${QUOTA}"
else
    echo 'Review the new retry first. Preserve all failed claims and output.' >&2
    exit 1
fi
ARGS=(submit --spec "${SPEC}")
if [[ $# -eq 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
    ARGS+=(--execute --plan-hash "$6" --authorization AUTHORIZE_FULLSTAT_SPLITHEAD_FINALIZE_RETRY_V1)
fi
exec python -u -B -s "${CLI}" "${ARGS[@]}"

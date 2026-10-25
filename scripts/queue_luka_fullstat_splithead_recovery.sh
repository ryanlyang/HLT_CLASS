#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo 'Usage: helper COMMIT NEW_ROOT FAILED_SPEC RESULT_QUOTA_GIB [--execute HASH]' >&2
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
export JC2_SITE=oscar_l40s
unset PYTHONHOME PYTHONPATH LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_fullstat_splithead_recovery.py"
SPEC="${ROOT}/campaign_spec.json"
if [[ -f "${SPEC}" ]]; then
    python -B -s -c 'import sys; from pathlib import Path; from hlt_classification.data.cache_contracts import load_json; s=load_json(sys.argv[1]); assert s["source_commit"]==sys.argv[2] and Path(s["recovery"]["parent_spec"]["path"]).resolve()==Path(sys.argv[3]).resolve() and s["available_result_bytes"]==int(sys.argv[4])*2**30, "Different recovery requested"' "${SPEC}" "${COMMIT}" "${PREVIOUS}" "${QUOTA}"
elif [[ $# -eq 4 ]]; then
    test ! -e "${ROOT}"
    python -u -B -s "${CLI}" create --root "${ROOT}" --previous-spec "${PREVIOUS}" \
        --expected-commit "${COMMIT}" --result-quota-gib "${QUOTA}"
else
    echo 'Review a fresh recovery first. Preserve partial output and claims.' >&2
    exit 1
fi
ARGS=(submit --spec "${SPEC}")
if [[ $# -eq 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
    ARGS+=(--execute --plan-hash "$6" --authorization AUTHORIZE_FULLSTAT_SPLITHEAD_RECOVERY_V1)
fi
exec python -u -B -s "${CLI}" "${ARGS[@]}"

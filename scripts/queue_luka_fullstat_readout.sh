#!/usr/bin/env bash
# Manual exact-hash dry/live alternative; do not use after detached launch.
set -euo pipefail
[[ $# -eq 4 || $# -eq 6 ]] || { echo 'Usage: COMMIT ROOT U100_W5_SPEC QUOTA_GIB [--execute PLAN_HASH]' >&2; exit 2; }
COMMIT="$1"; ROOT="$2"; PARENT="$3"; QUOTA="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${PARENT}" == /* ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 4 ))
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
CLI="${PROJECT_DIR}/scripts/luka_fullstat_readout.py"
SPEC="${ROOT}/campaign_spec.json"
if [[ -f "${SPEC}" ]]; then
    python -B -s -c 'import sys; from pathlib import Path; from hlt_classification.data.cache_contracts import load_json; s=load_json(sys.argv[1]); assert s["source_commit"]==sys.argv[2] and Path(s["reuse"]["parent_spec"]["path"]).resolve()==Path(sys.argv[3]).resolve() and s["available_quota_gib"]==int(sys.argv[4]), "Different execution requested"' "${SPEC}" "${COMMIT}" "${PARENT}" "${QUOTA}"
else
    [[ $# -eq 4 ]]
    test ! -e "${ROOT}"
    python -u -B -s "${CLI}" create --root "${ROOT}" --previous-spec "${PARENT}" --expected-commit "${COMMIT}" --result-quota-gib "${QUOTA}"
fi
ARGS=(submit --spec "${SPEC}")
if [[ $# -eq 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
    ARGS+=(--execute --plan-hash "$6" --authorization AUTHORIZE_FULLSTAT_READOUT_V1)
fi
exec python -u -B -s "${CLI}" "${ARGS[@]}"

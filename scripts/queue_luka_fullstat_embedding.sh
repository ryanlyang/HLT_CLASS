#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 && $# -ne 7 ]]; then
    echo 'Usage: helper COMMIT ROOT ORIGINAL_SCIENCE_SPEC RESULT_QUOTA_GIB gate|science [--execute HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
ORIGINAL="$3"
QUOTA="$4"
MODE="$5"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${ORIGINAL}" == /* ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${MODE}" =~ ^(gate|science)$ ]]
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
export JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_fullstat_embedding.py"
GATE="${ROOT}/gate_spec.json"
if [[ -f "${GATE}" ]]; then
    python -B -s -c 'import sys; from pathlib import Path; from hlt_classification.data.cache_contracts import load_json; g=load_json(sys.argv[1]); assert g["source_commit"]==sys.argv[2] and Path(g["anchor"]["spec"]["path"]).resolve()==Path(sys.argv[3]).resolve() and g["available_result_bytes"]==int(sys.argv[4])*2**30, "Different execution requested"' "${GATE}" "${COMMIT}" "${ORIGINAL}" "${QUOTA}"
elif [[ $# -eq 5 && "${MODE}" == gate ]]; then
    test ! -e "${ROOT}"
    python -u -B -s "${CLI}" create-gate --root "${ROOT}" --original-spec "${ORIGINAL}" \
        --expected-commit "${COMMIT}" --result-quota-gib "${QUOTA}"
else
    echo 'Dry-review a fresh gate first. Preserve partial results.' >&2
    exit 1
fi
SPEC="${GATE}"
if [[ "${MODE}" == science ]]; then
    SPEC="${ROOT}/science/campaign_spec.json"
    if [[ $# -eq 5 && ! -f "${SPEC}" ]]; then
        python -u -B -s "${CLI}" create-science --spec "${GATE}"
    fi
fi
test -f "${SPEC}"
if [[ $# -eq 7 ]]; then
    [[ "$6" == --execute && "$7" =~ ^[0-9a-f]{64}$ ]]
    python -u -B -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$7" \
        --authorization 'AUTHORIZE FULLSTAT LAST EMBEDDING EXACT PLAN'
else
    python -u -B -s "${CLI}" submit --spec "${SPEC}"
fi

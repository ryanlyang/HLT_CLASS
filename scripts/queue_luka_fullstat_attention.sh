#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 7 && $# -ne 9 ]]; then
    echo 'Usage: helper COMMIT ROOT PARENT_SPEC BANK_ROOT BANK_QUOTA_GIB RESULT_QUOTA_GIB gate|science [--execute HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
PARENT="$3"
BANK_ROOT="$4"
QUOTA="$5"
RESULT_QUOTA="$6"
MODE="$7"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ && "${RESULT_QUOTA}" =~ ^[1-9][0-9]*$ ]]
[[ "${MODE}" =~ ^(gate|science)$ && "${ROOT}" == /* && "${BANK_ROOT}" == /* && "${PARENT}" == /* ]]
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
export JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_fullstat_attention.py"
GATE="${ROOT}/gate_spec.json"
if [[ -f "${GATE}" ]]; then
    python -B -s -c 'import sys; from pathlib import Path; from hlt_classification.data.cache_contracts import load_json; g=load_json(sys.argv[1]); assert Path(g["anchor"]["spec"]["path"]).resolve()==Path(sys.argv[2]).resolve() and Path(g["bank_root"]).resolve()==Path(sys.argv[3]).resolve() and g["available_bank_bytes"]==int(sys.argv[4])*2**30 and g["available_result_bytes"]==int(sys.argv[5])*2**30 and g["source_commit"]==sys.argv[6], "Different campaign requested"' "${GATE}" "${PARENT}" "${BANK_ROOT}" "${QUOTA}" "${RESULT_QUOTA}" "${COMMIT}"
elif [[ $# -eq 7 && "${MODE}" == gate ]]; then
    test ! -e "${ROOT}"
    test ! -e "${BANK_ROOT}"
    python -u -B -s "${CLI}" create-gate --root "${ROOT}" --parent "${PARENT}" \
        --bank-root "${BANK_ROOT}" --quota-gib "${QUOTA}" --result-quota-gib "${RESULT_QUOTA}" --expected-commit "${COMMIT}"
else
    echo 'Dry-review a fresh gate first. Preserve partial outputs.' >&2
    exit 1
fi
SPEC="${GATE}"
if [[ "${MODE}" == science ]]; then
    SPEC="${ROOT}/science/campaign_spec.json"
    if [[ $# -eq 7 && ! -f "${SPEC}" ]]; then
        python -u -B -s "${CLI}" create-science --spec "${GATE}"
    fi
fi
test -f "${SPEC}"
if [[ $# -eq 9 ]]; then
    [[ "$8" == --execute && "$9" =~ ^[0-9a-f]{64}$ ]]
    python -u -B -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$9" \
        --authorization 'AUTHORIZE FULLSTAT LAST ATTENTION EXACT PLAN'
    echo "QUEUE_HELPER_EXIT=0; submitted reviewed ${MODE}. No automatic follow-up."
else
    python -u -B -s "${CLI}" submit --spec "${SPEC}"
fi

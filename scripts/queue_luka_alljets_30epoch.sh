#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo 'Usage: helper COMMIT NEW_ROOT ORIGINAL_BATCH128_GATE gate|science [--execute REVIEWED_HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
FAILED_GATE="$3"
MODE="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${MODE}" =~ ^(gate|science)$ ]]
PROJECT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT}/scripts/luka_fullsim_alljets_fusion.py"
GATE="${ROOT}/gate_spec.json"
if [[ -f "${GATE}" ]]; then
    python -B -s -c 'import json,sys; from pathlib import Path; g=json.load(open(sys.argv[1])); assert g["schema_version"]==6 and Path(g["resource_reuse"]["original_gate"]["path"]).resolve()==Path(sys.argv[2]).resolve(), "Different fixed30 campaign requested"' "${GATE}" "${FAILED_GATE}"
elif [[ $# -eq 4 && "${MODE}" == gate ]]; then
    test ! -e "${ROOT}"
    python -u -B -s "${CLI}" create-30epoch --root "${ROOT}" --failed-gate "${FAILED_GATE}" --expected-commit "${COMMIT}"
else
    echo 'Dry-review fixed30 first; preserve prior artifacts.' >&2
    exit 1
fi
SPEC="${GATE}"
if [[ "${MODE}" == science ]]; then
    SPEC="${ROOT}/science/campaign_spec.json"
    if [[ $# -eq 4 && ! -f "${SPEC}" ]]; then
        python -u -B -s "${CLI}" create-science --spec "${GATE}"
    fi
fi
test -f "${SPEC}"
if [[ $# -eq 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
    python -u -B -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$6" \
        --authorization 'AUTHORIZE LUKA MATCHED BATCH128 FIXED30 EXACT PLAN'
    echo "QUEUE_HELPER_EXIT=0; submitted reviewed ${MODE}. No automatic followup."
else
    python -u -B -s "${CLI}" submit --spec "${SPEC}"
    echo 'Dry review only. Fixed30 gate is ONE CPU-only evidence/budget check; science is30 jobs.'
fi

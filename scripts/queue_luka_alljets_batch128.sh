#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo 'Usage: helper COMMIT NEW_ROOT FAILED_ISOLATED_GATE gate|science [--execute REVIEWED_HASH]' >&2
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
    python -s -c 'import json,sys; from pathlib import Path; g=json.load(open(sys.argv[1])); assert g["schema_version"]==4 and Path(g["recovery"]["failed_gate"]["path"]).resolve()==Path(sys.argv[2]).resolve(), "Different batch128 campaign requested"' "${GATE}" "${FAILED_GATE}"
elif [[ $# -eq 4 && "${MODE}" == gate ]]; then
    test ! -e "${ROOT}"
    python -u -s "${CLI}" create-batch128 --root "${ROOT}" --failed-gate "${FAILED_GATE}" --expected-commit "${COMMIT}"
else
    echo 'Missing batch128 gate; dry-review the gate first.' >&2
    exit 1
fi
SPEC="${GATE}"
if [[ "${MODE}" == science ]]; then
    SPEC="${ROOT}/science/campaign_spec.json"
    if [[ $# -eq 4 && ! -f "${SPEC}" ]]; then
        python -u -s "${CLI}" create-science --spec "${GATE}"
    fi
fi
test -f "${SPEC}"
if [[ $# -eq 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$6" \
        --authorization 'AUTHORIZE LUKA MATCHED BATCH128 EXACT PLAN'
    echo "QUEUE_HELPER_EXIT=0; submitted reviewed ${MODE}. No automatic followup."
else
    python -u -s "${CLI}" submit --spec "${SPEC}"
    echo 'Dry review only. No jobs submitted. Repeat with --execute and the plan content_hash.'
fi

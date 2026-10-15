#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo 'Usage: helper COMMIT ROOT PARENT_SPEC gate|science [--execute REVIEWED_HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
PARENT="$3"
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
SPEC="${ROOT}/gate_spec.json"
if [[ "${MODE}" == science ]]; then SPEC="${ROOT}/science/campaign_spec.json"; fi
if [[ -f "${ROOT}/gate_spec.json" ]]; then
    python -s -c 'import json,sys; from pathlib import Path; g=json.load(open(sys.argv[1])); assert Path(g["parent_import"]["spec"]["path"]).resolve()==Path(sys.argv[2]).resolve(), "Different parent requested"' "${ROOT}/gate_spec.json" "${PARENT}"
fi
if [[ $# -eq 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
    test -f "${SPEC}"
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$6" \
        --authorization 'AUTHORIZE LUKA ALLJETS FUSION BOTH ROUTES EXACT PLAN'
    echo 'QUEUE_HELPER_EXIT=0; exact reviewed jobs registered. Parent campaign unchanged.'
    exit 0
fi
if [[ ! -f "${SPEC}" ]]; then
    if [[ "${MODE}" == gate ]]; then
        test ! -e "${ROOT}"
        python -u -s "${CLI}" create-gate --root "${ROOT}" --parent "${PARENT}" --expected-commit "${COMMIT}"
    else
        test ! -e "${ROOT}/science"
        python -u -s "${CLI}" create-science --spec "${ROOT}/gate_spec.json"
    fi
fi
python -u -s "${CLI}" submit --spec "${SPEC}"
echo 'Dry review only. No jobs submitted. Repeat with --execute and the plan content_hash.'

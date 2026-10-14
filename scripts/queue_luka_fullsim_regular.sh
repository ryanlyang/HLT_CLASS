#!/usr/bin/env bash
# Source-bound gate and science, each dry reviewed before explicit submission.
set -euo pipefail
if [[ $# -ne 3 && $# -ne 5 ]]; then
    echo 'Usage: helper COMMIT ROOT gate|science [--execute REVIEWED_PLAN_HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
MODE="$3"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MODE}" == gate || "${MODE}" == science ]]
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
CLI="${PROJECT}/scripts/luka_fullsim_regular.py"
SPEC="${ROOT}/gate_spec.json"
if [[ "${MODE}" == science ]]; then SPEC="${ROOT}/science/campaign_spec.json"; fi
if [[ $# -eq 5 ]]; then
    [[ "$4" == --execute && "$5" =~ ^[0-9a-f]{64}$ ]]
    test -f "${SPEC}"
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$5" \
        --authorization 'AUTHORIZE LUKA REGULAR EXACT PLAN'
    echo 'QUEUE_HELPER_EXIT=0; exact reviewed jobs registered. No other jobs changed.'
    exit 0
fi
if [[ ! -f "${SPEC}" ]]; then
    if [[ "${MODE}" == gate ]]; then
        test ! -e "${ROOT}"
        python -u -s "${CLI}" create-gate --root "${ROOT}" --expected-commit "${COMMIT}" \
            --container /home/ryreu/atlas/datasets/cms_fullsim_630k_trial_luka_v1
    else
        test ! -e "${ROOT}/science"
        python -u -s "${CLI}" create-science --spec "${ROOT}/gate_spec.json"
    fi
fi
python -u -s "${CLI}" submit --spec "${SPEC}"
echo 'Dry review only. No jobs submitted. Repeat with --execute and the plan content_hash.'

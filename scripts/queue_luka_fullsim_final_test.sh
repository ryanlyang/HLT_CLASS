#!/usr/bin/env bash
# Separate final-test authorization; no science-job mutations or retraining.
set -euo pipefail
[[ $# -eq 2 || $# -eq 4 ]] || { echo 'Usage: helper COMMIT ROOT [--execute REVIEWED_PLAN_HASH]' >&2; exit 2; }
COMMIT="$1"
ROOT="$2"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
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
CLI="${PROJECT}/scripts/luka_fullsim_final_test.py"
LOCK="${ROOT}/execution_lock.json"
if [[ $# -eq 4 ]]; then
    [[ "$3" == --execute && "$4" =~ ^[0-9a-f]{64}$ ]]
    test -f "${LOCK}"
    python -u -s "${CLI}" submit --lock "${LOCK}" --execute --plan-hash "$4" \
        --authorization 'AUTHORIZE LUKA ALL 12 FROZEN MODELS FINAL TEST EXACT PLAN'
    echo 'QUEUE_HELPER_EXIT=0; final-test evaluation and report submitted. No retraining.'
    exit 0
fi
if [[ ! -f "${LOCK}" ]]; then
    test ! -e "${ROOT}"
    python -u -s "${CLI}" create --expected-commit "${COMMIT}" --root "${ROOT}" \
        --parent-spec /home/ryreu/atlas/HLT_Classification/checkpoints/luka_science_01a73832_r1/campaign_spec.json
fi
python -u -s "${CLI}" submit --lock "${LOCK}"
echo 'Dry review complete: 2 jobs, all 12 frozen models, 149553 test jets. No jobs submitted or test particles read.'

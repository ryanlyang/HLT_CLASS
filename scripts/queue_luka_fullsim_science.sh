#!/usr/bin/env bash
# Dry review first. No probes, old-job mutations, retries, or test evaluation.
set -euo pipefail
if [[ $# -ne 2 && $# -ne 4 ]]; then
    echo 'Usage: helper COMMIT ROOT [--execute REVIEWED_PLAN_HASH]' >&2
    exit 2
fi
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
CLI="${PROJECT}/scripts/luka_fullsim_science.py"
BASE=/home/ryreu/atlas/HLT_Classification/checkpoints
SPEC="${ROOT}/campaign_spec.json"
if [[ $# -eq 4 ]]; then
    [[ "$3" == --execute && "$4" =~ ^[0-9a-f]{64}$ ]]
    test -f "${SPEC}"
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$4" \
        --authorization 'AUTHORIZE LUKA FULLSIM FUSION SCIENCE EXACT PLAN'
    echo 'QUEUE_HELPER_EXIT=0; all 21 science jobs registered. No old jobs changed.'
    exit 0
fi
if [[ ! -f "${SPEC}" ]]; then
    test ! -e "${ROOT}"
    python -u -s "${CLI}" create --expected-commit "${COMMIT}" --root "${ROOT}" \
        --prepared-root "${BASE}/luka_fullsim_1256c756_r2/prepared_mm_r1" \
        --preflight-path "${BASE}/luka_preflight_v2_bcccd498_r1/preflight.json" \
        --prepared-project /home/ryreu/atlas/HLT_Classification_luka_1256c756 \
        --preflight-project /home/ryreu/atlas/HLT_Classification_luka_preflight_v2_bcccd498
fi
python -u -s "${CLI}" submit --spec "${SPEC}"
echo 'Dry review complete. No jobs submitted. Repeat with --execute and the reviewed plan content_hash.'

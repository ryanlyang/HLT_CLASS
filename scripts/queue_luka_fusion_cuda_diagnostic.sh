#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 && $# -ne 5 ]]; then
    echo 'Usage: helper COMMIT ROOT ORIGINAL_GATE [--execute REVIEWED_HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
ORIGINAL_GATE="$3"
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
CLI="${PROJECT}/scripts/luka_fusion_cuda_diagnostic.py"
SPEC="${ROOT}/diagnostic_spec.json"
if [[ -f "${SPEC}" ]]; then
    python -s -c 'import json,sys; from pathlib import Path; s=json.load(open(sys.argv[1])); assert Path(s["original_gate"]["path"]).resolve()==Path(sys.argv[2]).resolve(), "Different original gate"' "${SPEC}" "${ORIGINAL_GATE}"
fi
if [[ $# -eq 5 ]]; then
    [[ "$4" == --execute && "$5" =~ ^[0-9a-f]{64}$ ]]
    test -f "${SPEC}"
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$5" \
        --authorization 'AUTHORIZE LUKA FUSION CUDA DIAGNOSTIC EXACT PLAN'
    echo 'QUEUE_HELPER_EXIT=0; one diagnostic job. No science or automatic followup.'
else
    if [[ ! -f "${SPEC}" ]]; then
        test ! -e "${ROOT}"
        python -u -s "${CLI}" create --root "${ROOT}" --original-gate "${ORIGINAL_GATE}" --expected-commit "${COMMIT}"
    fi
    python -u -s "${CLI}" submit --spec "${SPEC}"
    echo 'Dry review only; no jobs submitted. Review the plan content_hash.'
fi

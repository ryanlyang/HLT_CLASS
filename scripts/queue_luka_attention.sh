#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 5 && $# -ne 7 ]]; then
    echo 'Usage: helper COMMIT ROOT PARENT_SPEC AVAILABLE_QUOTA_GIB gate|science [--execute REVIEWED_HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
PARENT="$3"
QUOTA="$4"
MODE="$5"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ && "${MODE}" =~ ^(gate|science)$ ]]
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
CLI="${PROJECT}/scripts/luka_attention.py"
GATE="${ROOT}/gate_spec.json"
if [[ -f "${GATE}" ]]; then
    python -B -s -c 'import json,sys; from pathlib import Path; g=json.load(open(sys.argv[1])); assert Path(g["anchor"]["spec"]["path"]).resolve()==Path(sys.argv[2]).resolve() and g["available_quota_gib"]==int(sys.argv[3]), "Different campaign requested"' "${GATE}" "${PARENT}" "${QUOTA}"
elif [[ $# -eq 5 && "${MODE}" == gate ]]; then
    test ! -e "${ROOT}"
    python -u -B -s "${CLI}" create-gate --root "${ROOT}" --parent "${PARENT}" --available-quota-gib "${QUOTA}" --expected-commit "${COMMIT}"
else
    echo 'Dry-review a fresh gate first; preserve partial outputs.' >&2
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
        --authorization 'AUTHORIZE LUKA D ATTENTION EXACT PLAN'
    echo "QUEUE_HELPER_EXIT=0; submitted reviewed ${MODE}. No automatic follow-up."
else
    python -u -B -s "${CLI}" submit --spec "${SPEC}"
fi

#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 7 && $# -ne 9 ]]; then
    echo 'Usage: helper COMMIT ROOT PARENT_SPEC REFERENCE_SPEC logit|attn QUOTA_GIB gate|science [--execute HASH] (both arms: debug)' >&2
    exit 2
fi
COMMIT="$1"; ROOT="$2"; PARENT="$3"; REFERENCE="$4"; ARM="$5"; QUOTA="$6"; MODE="$7"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ARM}" =~ ^(logit|attn)$ && "${QUOTA}" =~ ^[1-9][0-9]*$ && "${MODE}" =~ ^(gate|science)$ ]]
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
CLI="${PROJECT}/scripts/luka_particle_swap.py"
GATE="${ROOT}/gate_spec.json"
if [[ -f "${GATE}" ]]; then
    python -B -s -c 'import json,sys; from pathlib import Path; from hlt_classification.luka_particle_swap.campaign import validate_execution; g=json.load(open(sys.argv[1])); validate_execution(g); assert g["schema_version"]==1, "Use a fresh debug gate root; never migrate an old gate in place"; assert Path(g["anchor"]["spec"]["path"]).resolve()==Path(sys.argv[2]).resolve() and Path(g["reference_import"]["spec"]["path"]).resolve()==Path(sys.argv[3]).resolve() and g["arm"]==sys.argv[4] and g["available_quota_gib"]==int(sys.argv[5]) and g["source_commit"]==sys.argv[6] and Path(g["project_dir"]).resolve()==Path(sys.argv[7]).resolve(), "Different campaign requested"' "${GATE}" "${PARENT}" "${REFERENCE}" "${ARM}" "${QUOTA}" "${COMMIT}" "${PROJECT}"
elif [[ $# -eq 7 && "${MODE}" == gate ]]; then
    test ! -e "${ROOT}"
    python -u -B -s "${CLI}" create-gate --root "${ROOT}" --parent "${PARENT}" --reference "${REFERENCE}" --arm "${ARM}" --available-quota-gib "${QUOTA}" --expected-commit "${COMMIT}"
else
    echo 'Dry-review a fresh gate first; preserve partial outputs.' >&2
    exit 1
fi
SPEC="${GATE}"
if [[ "${MODE}" == science ]]; then
    SPEC="${ROOT}/science/campaign_spec.json"
    if [[ $# -eq 7 && ! -f "${SPEC}" ]]; then
        python -u -B -s "${CLI}" create-science --spec "${GATE}"
    fi
fi
if [[ $# -eq 9 ]]; then
    [[ "$8" == --execute && "$9" =~ ^[0-9a-f]{64}$ ]]
    python -u -B -s "${CLI}" submit --spec "${SPEC}" --execute --plan-hash "$9" --authorization 'AUTHORIZE LUKA PARTICLE SWAP EXACT PLAN AND OWNED RETIREMENT'
    echo 'QUEUE_HELPER_EXIT=0; no automatic follow-up or retry.'
else
    python -u -B -s "${CLI}" submit --spec "${SPEC}"
fi

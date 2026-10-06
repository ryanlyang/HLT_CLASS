#!/usr/bin/env bash
# Dry by default. Execute submits the complete gate -> array -> finalizer DAG.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve the root and submission journals." >&2' ERR
if [[ $# != 3 && $# != 5 ]]; then
    echo 'Usage: queue_jetclass2_literature_context_dataset.sh COMMIT ROOT AVAILABLE_QUOTA_GIB [--execute PLAN_HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
QUOTA="$3"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${CONTEXT_PERSISTENT_STORAGE:-}" == YES ]] || { echo 'Set CONTEXT_PERSISTENT_STORAGE=YES only after confirming persistence and available quota.' >&2; exit 2; }
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PILOT="${CONTEXT_PILOT_SPEC:-/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_context_pilot_6cb5f32a_r1/study_spec.json}"
PROFILE="${CONTEXT_TRAIN_PROFILE:-/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2_delphes_scaling_splits_20260918_v1/profiles/TRAIN_1M.json}"
CLI="${PROJECT_DIR}/scripts/jetclass2_literature_context_dataset.py"
if [[ ! -e "${ROOT}" ]]; then
    [[ $# == 3 ]] || { echo 'Create and review the dry plan before execution.' >&2; exit 2; }
    python -u -s "${CLI}" create --project "${PROJECT_DIR}" --commit "${COMMIT}" \
        --parent-spec "${PILOT}" --profile "${PROFILE}" --root "${ROOT}" \
        --persistent-parent "$(dirname -- "${ROOT}")" --available-quota-gib "${QUOTA}" \
        --persistent-attested --acknowledge-synthetic
fi
ATTEMPT="${ROOT}/attempts/initial/attempt_spec.json"
# Never let a stale shell root silently address another commit or recipe source.
python -s - "${ROOT}" "${PROJECT_DIR}" "${COMMIT}" "${PILOT}" "${PROFILE}" "${QUOTA}" <<'PY'
import sys
import math
from pathlib import Path
from hlt_classification.literature_context_production import campaign as c
from hlt_classification.literature_context_production.contracts import load_json
root, project, commit, pilot, profile, quota = sys.argv[1:]
s = load_json(Path(root)/'study_spec.json')
f = c.bundle(s)
if (s['source']['commit'] != commit or Path(s['root']).resolve() != Path(root).resolve()
        or Path(s['project_dir']).resolve() != Path(project).resolve()
        or Path(f['pilot']['path']).resolve() != Path(pilot).resolve()
        or Path(s['profile']['path']).resolve() != Path(profile).resolve()):
    raise SystemExit('STOP: saved study differs; preserve it and inspect.')
if not math.isfinite(float(quota)) or float(quota)*2**30 < s['storage']['budget_bytes']+2*2**30:
    raise SystemExit('STOP: current available quota is below the frozen budget plus headroom.')
PY
if [[ $# == 5 ]]; then
    [[ "$4" == --execute && "$5" =~ ^[0-9a-f]{64}$ ]]
    python -u -s "${CLI}" submit --attempt "${ATTEMPT}" --execute --plan-hash "$5" \
        --phrase 'AUTHORIZE JC2 CONTEXT DATASET INITIAL EXACT PLAN'
    echo 'Entire DAG submitted. No further submission is needed after preflight passes.'
else
    python -u -s "${CLI}" plan --attempt "${ATTEMPT}"
    echo 'Dry review only. No jobs submitted. Repeat with --execute and the exact plan content_hash.'
fi

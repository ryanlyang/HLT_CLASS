#!/usr/bin/env bash
# Run from an exact clean pushed worktree. Dry review first; never auto-follow-up.
set -euo pipefail
if [[ "$#" -ne 2 && "$#" -ne 4 ]]; then
    echo "Usage: bash $0 COMMIT NEW_PILOT_ROOT [--execute REVIEWED_PLAN_HASH]" >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/jetclass2_literature_proxy.py"
DATA=/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2
INVENTORY=/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_partial_inventory_0b3ed523_r1/inventory.json
PROFILE=/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2_delphes_scaling_splits_20260918_v1/profiles/TRAIN_1M.json
SPEC="${ROOT}/study_spec.json"
if [[ "$#" -eq 2 ]]; then
    python -u -s "${CLI}" create --project "${PROJECT_DIR}" --commit "${COMMIT}" \
        --data-root "${DATA}" --inventory "${INVENTORY}" --profile "${PROFILE}" --root "${ROOT}"
    python -u -s "${CLI}" submit --spec "${SPEC}"
    echo "Dry review only. No jobs submitted. Review the single CPU-only job and plan content_hash."
else
    [[ "$3" == --execute && "$4" =~ ^[0-9a-f]{64}$ ]]
    test -f "${SPEC}"
    python -u -s - "${SPEC}" "${COMMIT}" "${PROJECT_DIR}" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
if s['source']['commit'] != sys.argv[2] or Path(s['project_dir']).resolve() != Path(sys.argv[3]).resolve():
    raise SystemExit('STOP: helper/source mismatch')
PY
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --reviewed-hash "$4" \
        --authorization 'AUTHORIZE JC2 LITERATURE PROXY PILOT EXACT PLAN'
fi

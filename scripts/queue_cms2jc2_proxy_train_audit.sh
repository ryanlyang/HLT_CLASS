#!/usr/bin/env bash
# New worktree required. No job mutation; dry-run first, exact hash for submission.
set -euo pipefail
if [[ "$#" != 3 && "$#" != 5 ]]; then
    echo 'Usage: bash queue_cms2jc2_proxy_train_audit.sh COMMIT DATASET_ROOT AUDIT_ROOT [--execute PLAN_HASH]' >&2
    exit 2
fi
COMMIT="$1"
DATASET_ROOT="$2"
AUDIT_ROOT="$3"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" == "${COMMIT}" ]]
if [[ "$#" == 5 ]]; then
    [[ "$4" == --execute && "$5" =~ ^[0-9a-f]{64}$ ]]
fi
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/cms2jc2_proxy_train_audit.py"
SPEC="${AUDIT_ROOT}/audit_spec.json"
if [[ ! -f "${SPEC}" ]]; then
    python -u -s "${CLI}" prepare --dataset-root "${DATASET_ROOT}" --root "${AUDIT_ROOT}" \
        --project "${PROJECT_DIR}" --commit "${COMMIT}"
fi
# Reject accidental reuse of a different campaign's spec, even if it exists.
python -s - "${SPEC}" "${DATASET_ROOT}" "${AUDIT_ROOT}" "${PROJECT_DIR}" "${COMMIT}" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
assert Path(s['study']['path']).resolve() == Path(sys.argv[2]).resolve()/'study_spec.json'
assert Path(s['root']).resolve() == Path(sys.argv[3]).resolve()
assert Path(s['project_dir']).resolve() == Path(sys.argv[4]).resolve()
assert s['source']['commit'] == sys.argv[5]
PY
if [[ "$#" == 5 ]]; then
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --reviewed-hash "$5"
else
    python -u -s "${CLI}" submit --spec "${SPEC}"
    echo 'Dry review only. Repeat with --execute and the printed plan content_hash.'
fi

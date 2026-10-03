#!/usr/bin/env bash
# One job, dry review first. Never replaces v1/v2 or automatically submits followups.
set -euo pipefail
if [[ "$#" -ne 3 && "$#" -ne 5 ]]; then
    echo "Usage: bash $0 COMMIT PARENT_SPEC NEW_ROOT [--execute REVIEWED_PLAN_HASH]" >&2
    exit 2
fi
COMMIT="$1"
PARENT="$2"
ROOT="$3"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/jetclass2_literature_proxy_noise.py"
SPEC="${ROOT}/study_spec.json"
if [[ ! -e "${SPEC}" ]]; then
    [[ "$#" -eq 3 ]]
    echo "Preparing v3: authenticating the completed v2 training pilot and frozen rates. No jobs submitted yet."
    python -u -s "${CLI}" create --project "${PROJECT_DIR}" --commit "${COMMIT}" \
        --parent-spec "${PARENT}" --root "${ROOT}"
fi
python -s - "${SPEC}" "${COMMIT}" "${PROJECT_DIR}" "${PARENT}" "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
if (s['source']['commit'] != sys.argv[2]
    or Path(s['project_dir']).resolve() != Path(sys.argv[3]).resolve()
    or Path(s['parent_spec']['path']).resolve() != Path(sys.argv[4]).resolve()
    or Path(s['root']).resolve() != Path(sys.argv[5]).resolve()):
    raise SystemExit('STOP: helper/source/parent/root mismatch')
PY
if [[ "$#" -eq 3 ]]; then
    python -u -s "${CLI}" submit --spec "${SPEC}"
    echo "Dry review only. One CPU-only job; no recalibration or automatic production. No jobs submitted."
else
    [[ "$4" == --execute && "$5" =~ ^[0-9a-f]{64}$ ]]
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --reviewed-hash "$5" \
        --authorization 'AUTHORIZE JC2 LITERATURE PROXY NOISE V3 EXACT PLAN'
fi

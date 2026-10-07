#!/usr/bin/env bash
# One new pilot only, immutable dry review before live submission.
set -euo pipefail
if [[ "$#" -ne 3 && "$#" -ne 5 ]]; then
    echo "Usage: bash $0 COMMIT ORIGINAL_LITERATURE_PILOT_SPEC NEW_ROOT [--execute REVIEWED_PLAN_HASH]" >&2
    exit 2
fi
COMMIT="$1"
PARENT="$2"
ROOT="$3"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
if [[ "$#" -eq 5 ]]; then
    [[ "$4" == --execute && "$5" =~ ^[0-9a-f]{64}$ ]]
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/jetclass2_correlated_tracking.py"
SPEC="${ROOT}/study_spec.json"
if [[ ! -e "${SPEC}" ]]; then
    [[ "$#" -eq 3 ]]
    echo "Authenticating original literature pilot OFFLINE banks; no jobs submitted yet."
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
    or Path(s['root']).resolve() != Path(sys.argv[5]).resolve()
    or s['population']['jets'] != 20000):
    raise SystemExit('STOP: helper/source/parent/root/20k population mismatch')
PY
if [[ "$#" -eq 3 ]]; then
    python -u -s "${CLI}" submit --spec "${SPEC}"
    echo "Dry review only. One CPU-only job; no classifiers or automatic followup."
else
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --reviewed-hash "$5" \
        --authorization 'AUTHORIZE JC2 CORRELATED TRACKING PILOT EXACT PLAN'
fi

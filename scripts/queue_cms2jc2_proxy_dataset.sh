#!/usr/bin/env bash
# Dry by default. Run under nohup/tmux to survive SSH disconnects.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo 'Usage: queue_cms2jc2_proxy_dataset.sh ATTEMPT_SPEC [--execute REVIEWED_PLAN_HASH]' >&2
  exit 2
fi
ATTEMPT="$1"
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/cms2jc2_proxy_dataset.py"
if [[ $# -eq 1 ]]; then
  python -s "${CLI}" submit --attempt "${ATTEMPT}"
  echo 'Dry review only; no jobs submitted.'
else
  [[ $# -eq 3 && "$2" == '--execute' && "$3" =~ ^[0-9a-f]{64}$ ]]
  PHRASE="$(python -s - "${ATTEMPT}" <<'PY'
import json, sys
from pathlib import Path
print(json.loads((Path(sys.argv[1]).parent/'command_plan.json').read_text())['authorization_phrase'])
PY
  )"
  python -s "${CLI}" submit --attempt "${ATTEMPT}" --execute \
    --plan-hash "$3" --authorization-phrase "${PHRASE}" &
  HELPER_PID=$!
  while kill -0 "${HELPER_PID}" 2>/dev/null; do
    echo "Production submit helper active: PID ${HELPER_PID}"
    sleep 15
  done
  wait "${HELPER_PID}"
fi

#!/bin/bash
set -euo pipefail
PROJECT_DIR="${1:?absolute project directory required}"
CAMPAIGN_SPEC="${2:?campaign spec required}"
TASK="${3:?exact task required}"
case "${PROJECT_DIR}" in /*) ;; *) echo "Absolute project path required" >&2; exit 1;; esac
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate atlas_kd_sporc
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Spawned preprocessing processes each use one native-math thread.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "${PROJECT_DIR}"
exec python -s "${PROJECT_DIR}/scripts/cms_salience_learned.py" run \
    --spec "${CAMPAIGN_SPEC}" --task "${TASK}"

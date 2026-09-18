#!/bin/bash
set -euo pipefail
PROJECT_DIR="${1:?absolute project directory required}"
PREPARATION_SPEC="${2:?absolute campaign or preparation spec required}"
TASK_ID="${3:-}"
case "${PROJECT_DIR}" in /*) ;; *) echo "Absolute project path required" >&2; exit 1;; esac
case "${PREPARATION_SPEC}" in /*) ;; *) echo "Absolute spec path required" >&2; exit 1;; esac
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1
cd "${PROJECT_DIR}"
if [ -n "${TASK_ID}" ]; then
  exec python -s "${PROJECT_DIR}/scripts/cms2jc2_response.py" run-task --spec "${PREPARATION_SPEC}" --task "${TASK_ID}"
fi
exec python -s "${PROJECT_DIR}/scripts/cms2jc2_response.py" run-preparation --spec "${PREPARATION_SPEC}"

#!/bin/bash
set -euo pipefail
PROJECT_DIR="${1:?absolute project directory required}"
STAGE_SPEC="${2:?absolute stage spec required}"
TASK_ID="${3:?exact task required}"
case "${PROJECT_DIR}" in /*) ;; *) exit 1;; esac
case "${STAGE_SPEC}" in /*) ;; *) exit 1;; esac
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=16
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
# Native crashes produce a traceback, never a multi-GB core dump.
ulimit -c 0
cd "${PROJECT_DIR}"
exec python -s "${PROJECT_DIR}/scripts/cms2jc2_response_dev.py" run-task --spec "${STAGE_SPEC}" --task "${TASK_ID}"

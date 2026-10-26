#!/usr/bin/env bash
# Execution-only runtime override. Never change old pinned sources or environments.
set -euo pipefail
[[ "${PROJECT_DIR:?absolute project required}" == /* ]]
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
module load miniforge3/25.3.0-3-a6hh
export CONDA_ACTIVATE_TARGET=/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar
export CONDA_SH="$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=64
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
test "${CONDA_PREFIX}" = "${CONDA_ACTIVATE_TARGET}"
export PYTHONPATH="${PROJECT_DIR}/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

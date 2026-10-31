#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]]
PROJECT_DIR="$1"
SPEC="$2"
TASK="$3"
[[ "${PROJECT_DIR}" == /* && "${SPEC}" == /* ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd -- "${PROJECT_DIR}"
exec python -u -B -s "${PROJECT_DIR}/scripts/luka_u100_contrastive.py" run --spec "${SPEC}" --task "${TASK}"

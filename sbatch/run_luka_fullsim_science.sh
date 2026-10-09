#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo 'Usage: worker PROJECT SPEC TASK' >&2
    exit 2
fi
export PROJECT_DIR="$1"
SPEC="$2"
TASK="$3"
source "${PROJECT_DIR}/sbatch/common.sh"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec python -u -s "${PROJECT_DIR}/scripts/luka_fullsim_science.py" run --spec "${SPEC}" --task "${TASK}"

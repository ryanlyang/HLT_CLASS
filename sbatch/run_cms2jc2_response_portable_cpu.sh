#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${1:?absolute project directory required}"
STAGE_SPEC="${2:?absolute stage spec required}"
TASK_ID="${3:?exact task required}"
case "${PROJECT_DIR}" in /*) ;; *) exit 1;; esac
case "${STAGE_SPEC}" in /*) ;; *) exit 1;; esac
case "${SLURM_JOB_PARTITION:?Slurm allocation required}" in
  tigris)
    source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
    conda activate /home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris ;;
  debug|tier3)
    source /home/ryreu/miniconda3/etc/profile.d/conda.sh
    conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc ;;
  *) echo "Unregistered CPU-only site" >&2; exit 1 ;;
esac
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=16 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
ulimit -c 0
cd "${PROJECT_DIR}"
exec python -s "${PROJECT_DIR}/scripts/cms2jc2_response_dev.py" run-task --spec "${STAGE_SPEC}" --task "${TASK_ID}"

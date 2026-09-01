#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_MT20_RECOVERY:?HCWDL_MT20_RECOVERY is required}"
: "${HCWDL_MT20_TASK:?HCWDL_MT20_TASK is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
test "${SLURM_NNODES}" -eq 1
test "${SLURM_NTASKS}" -eq 1
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_tri100_spine4_mt20_recovery_task.py" \
  --recovery "${HCWDL_MT20_RECOVERY}" \
  --task "${HCWDL_MT20_TASK}" --device cuda

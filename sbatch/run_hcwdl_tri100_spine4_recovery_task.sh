#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_SPINE4_RECOVERY:?HCWDL_SPINE4_RECOVERY is required}"
: "${HCWDL_SPINE4_TASK:?HCWDL_SPINE4_TASK is required}"
: "${HCWDL_SPINE4_EXECUTION_WORLD_SIZE:?HCWDL_SPINE4_EXECUTION_WORLD_SIZE is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
test "${HCWDL_SPINE4_EXECUTION_WORLD_SIZE}" -eq 1
test "${SLURM_NNODES}" -eq 1
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_tri100_spine4_recovery_task.py" \
  --recovery "${HCWDL_SPINE4_RECOVERY}" \
  --task "${HCWDL_SPINE4_TASK}" --device cuda --execution-world-size 1

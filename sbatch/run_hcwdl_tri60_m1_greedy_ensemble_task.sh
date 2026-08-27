#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_M1_GREEDY_SPEC:?HCWDL_M1_GREEDY_SPEC is required}"
: "${HCWDL_M1_GREEDY_TASK:?HCWDL_M1_GREEDY_TASK is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NUMEXPR_MAX_THREADS=64
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HCWDL_UB_VIEW_SOURCE_BACKEND=process
export HCWDL_M1_GREEDY_REDUCER_WORKERS=32
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_tri60_m1_greedy_ensemble_task.py" \
  --spec "${HCWDL_M1_GREEDY_SPEC}" --task "${HCWDL_M1_GREEDY_TASK}" --device cuda

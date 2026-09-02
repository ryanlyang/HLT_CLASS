#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_OFH_SPEC:?HCWDL_OFH_SPEC is required}"
: "${HCWDL_OFH_TASK:?HCWDL_OFH_TASK is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 NUMEXPR_MAX_THREADS=64 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_adjacent_output_handoff_task.py" --spec "${HCWDL_OFH_SPEC}" --task "${HCWDL_OFH_TASK}" --device cuda

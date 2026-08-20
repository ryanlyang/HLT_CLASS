#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_TRI60_FOUNDATION_LOCK:?HCWDL_TRI60_FOUNDATION_LOCK is required}"
: "${HCWDL_TRI60_SOURCE_COMMIT:?HCWDL_TRI60_SOURCE_COMMIT is required}"
: "${HCWDL_TRI60_PROFILE_OUTPUT:?HCWDL_TRI60_PROFILE_OUTPUT is required}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_mhpe_tri60_acceptance.py" \
  --foundation-lock "${HCWDL_TRI60_FOUNDATION_LOCK}" \
  --source-commit "${HCWDL_TRI60_SOURCE_COMMIT}" \
  --output "${HCWDL_TRI60_PROFILE_OUTPUT}" \
  --device cuda

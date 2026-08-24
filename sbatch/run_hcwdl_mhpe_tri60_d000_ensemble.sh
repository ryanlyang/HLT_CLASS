#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_TRI60_SPEC:?HCWDL_TRI60_SPEC is required}"
: "${HCWDL_TRI60_D000_ENSEMBLE_OUTPUT:?HCWDL_TRI60_D000_ENSEMBLE_OUTPUT is required}"
: "${HCWDL_TRI60_D000_ENSEMBLE_COMMIT:?HCWDL_TRI60_D000_ENSEMBLE_COMMIT is required}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec python -s \
  "${PROJECT_DIR}/scripts/evaluate_hcwdl_mhpe_tri60_d000_ensemble.py" \
  --campaign-spec "${HCWDL_TRI60_SPEC}" \
  --output "${HCWDL_TRI60_D000_ENSEMBLE_OUTPUT}" \
  --producer-commit "${HCWDL_TRI60_D000_ENSEMBLE_COMMIT}" \
  --device cuda

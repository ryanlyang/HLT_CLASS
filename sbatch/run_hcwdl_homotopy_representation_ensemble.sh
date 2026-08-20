#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_U_RKD_SPEC:?HCWDL_U_RKD_SPEC is required}"
: "${HCWDL_U_RKD_ENSEMBLE_RUNG:?HCWDL_U_RKD_ENSEMBLE_RUNG is required}"
: "${HCWDL_U_RKD_ENSEMBLE_OUTPUT:?HCWDL_U_RKD_ENSEMBLE_OUTPUT is required}"
: "${HCWDL_U_RKD_ENSEMBLE_COMMIT:?HCWDL_U_RKD_ENSEMBLE_COMMIT is required}"

source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec python -s \
  "${PROJECT_DIR}/scripts/evaluate_hcwdl_homotopy_representation_ensemble.py" \
  --campaign-spec "${HCWDL_U_RKD_SPEC}" \
  --rung "${HCWDL_U_RKD_ENSEMBLE_RUNG}" \
  --output "${HCWDL_U_RKD_ENSEMBLE_OUTPUT}" \
  --source-commit "${HCWDL_U_RKD_ENSEMBLE_COMMIT}"

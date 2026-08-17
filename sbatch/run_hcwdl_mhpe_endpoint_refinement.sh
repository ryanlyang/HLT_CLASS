#!/usr/bin/env bash
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_MHPE_SOURCE_SPEC:?HCWDL_MHPE_SOURCE_SPEC is required}"
: "${HCWDL_MHPE_BLEND_OUTPUT:?HCWDL_MHPE_BLEND_OUTPUT is required}"
: "${HCWDL_MHPE_BLEND_COMMIT:?HCWDL_MHPE_BLEND_COMMIT is required}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec python -s "${PROJECT_DIR}/scripts/evaluate_hcwdl_mhpe_endpoint_refinement.py" \
  --campaign-spec "${HCWDL_MHPE_SOURCE_SPEC}" \
  --output "${HCWDL_MHPE_BLEND_OUTPUT}" \
  --producer-commit "${HCWDL_MHPE_BLEND_COMMIT}" \
  --device cuda

#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_U_RKD_SPEC:?HCWDL_U_RKD_SPEC is required}"
: "${HCWDL_U_RKD_PROFILE_NODE:?HCWDL_U_RKD_PROFILE_NODE is required}"
: "${HCWDL_U_RKD_PROFILE_OUTPUT:?HCWDL_U_RKD_PROFILE_OUTPUT is required}"

source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec python -s \
  "${PROJECT_DIR}/scripts/profile_hcwdl_homotopy_representation_node.py" \
  --campaign-spec "${HCWDL_U_RKD_SPEC}" \
  --node "${HCWDL_U_RKD_PROFILE_NODE}" \
  --batches "${HCWDL_U_RKD_PROFILE_BATCHES:-50}" \
  --warmup-batches "${HCWDL_U_RKD_PROFILE_WARMUP_BATCHES:-5}" \
  --output "${HCWDL_U_RKD_PROFILE_OUTPUT}"

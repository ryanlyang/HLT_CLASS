#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_TRI60_SPEC:?HCWDL_TRI60_SPEC is required}"
: "${HCWDL_TRI60_CE_CONTROL_SPEC:?HCWDL_TRI60_CE_CONTROL_SPEC is required}"
: "${HCWDL_TRI60_FLAT8_OUTPUT:?HCWDL_TRI60_FLAT8_OUTPUT is required}"
: "${HCWDL_TRI60_FLAT8_COMMIT:?HCWDL_TRI60_FLAT8_COMMIT is required}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec python -s \
  "${PROJECT_DIR}/scripts/evaluate_hcwdl_mhpe_tri60_d000_logit_rset_flat8.py" \
  --campaign-spec "${HCWDL_TRI60_SPEC}" \
  --ce-control-spec "${HCWDL_TRI60_CE_CONTROL_SPEC}" \
  --output "${HCWDL_TRI60_FLAT8_OUTPUT}" \
  --producer-commit "${HCWDL_TRI60_FLAT8_COMMIT}"

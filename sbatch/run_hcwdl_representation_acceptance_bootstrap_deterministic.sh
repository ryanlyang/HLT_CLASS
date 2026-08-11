#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${HCWDL_REPRESENTATION_BOOTSTRAP:?HCWDL_REPRESENTATION_BOOTSTRAP is required}"
: "${HCWDL_REPRESENTATION_TASK:?HCWDL_REPRESENTATION_TASK is required}"

exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_representation_acceptance_bootstrap.py" \
  --bootstrap-spec "${HCWDL_REPRESENTATION_BOOTSTRAP}" \
  --task "${HCWDL_REPRESENTATION_TASK}" \
  --deterministic-worker

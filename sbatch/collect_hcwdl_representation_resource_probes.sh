#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${HCWDL_REPRESENTATION_PROBE_PLAN:?probe plan is required}"
: "${HCWDL_REPRESENTATION_PROBE_AUTHORIZATION:?probe authorization is required}"

exec python -s "${PROJECT_DIR}/scripts/collect_hcwdl_representation_dense_resource_probes.py" \
  --plan "${HCWDL_REPRESENTATION_PROBE_PLAN}" \
  --authorization "${HCWDL_REPRESENTATION_PROBE_AUTHORIZATION}"

#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${HCWDL_REPRESENTATION_PROBE_SPEC:?probe spec is required}"
: "${HCWDL_REPRESENTATION_PROBE_DATA_ROOT:?probe data root is required}"
: "${HCWDL_REPRESENTATION_PROBE_CONDA:?probe Conda environment is required}"
: "${HCWDL_REPRESENTATION_PROBE_RESOURCE_CLASS:?probe resource class is required}"
: "${HCWDL_REPRESENTATION_PROBE_OUTPUT:?probe output is required}"
: "${HCWDL_REPRESENTATION_PROBE_RUNTIME_OUTPUT:?probe runtime output is required}"

exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_representation_resource_probe.py" \
  --planning-campaign-spec "${HCWDL_REPRESENTATION_PROBE_SPEC}" \
  --data-root "${HCWDL_REPRESENTATION_PROBE_DATA_ROOT}" \
  --conda-environment "${HCWDL_REPRESENTATION_PROBE_CONDA}" \
  --resource-class "${HCWDL_REPRESENTATION_PROBE_RESOURCE_CLASS}" \
  --runtime-output "${HCWDL_REPRESENTATION_PROBE_RUNTIME_OUTPUT}" \
  --output "${HCWDL_REPRESENTATION_PROBE_OUTPUT}"

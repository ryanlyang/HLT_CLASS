#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_TRI60_SOURCE_SPEC:?HCWDL_TRI60_SOURCE_SPEC is required}"
: "${HCWDL_TRI60_DENSE_SPEC:?HCWDL_TRI60_DENSE_SPEC is required}"
: "${HCWDL_TRI60_LOGIT_ZOOM_OUTPUT_DIR:?HCWDL_TRI60_LOGIT_ZOOM_OUTPUT_DIR is required}"
: "${HCWDL_TRI60_LOGIT_ZOOM_COMMIT:?HCWDL_TRI60_LOGIT_ZOOM_COMMIT is required}"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export HCWDL_UB_VIEW_BUILD_WORKERS="${SLURM_CPUS_PER_TASK}"
export HCWDL_UB_VIEW_SOURCE_BACKEND=process

exec python -s \
  "${PROJECT_DIR}/scripts/plot_hcwdl_mhpe_tri60_logit_ladder_zoom_roc.py" \
  --source-campaign-spec "${HCWDL_TRI60_SOURCE_SPEC}" \
  --dense-campaign-spec "${HCWDL_TRI60_DENSE_SPEC}" \
  --output-dir "${HCWDL_TRI60_LOGIT_ZOOM_OUTPUT_DIR}" \
  --producer-commit "${HCWDL_TRI60_LOGIT_ZOOM_COMMIT}" \
  --device cuda

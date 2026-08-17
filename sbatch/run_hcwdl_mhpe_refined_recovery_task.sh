#!/usr/bin/env bash
#SBATCH --account=reu-aisocial
#SBATCH --partition=tigris
set -euo pipefail
: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${HCWDL_MHPE_REFINED_RECOVERY_SPEC:?HCWDL_MHPE_REFINED_RECOVERY_SPEC is required}"
: "${HCWDL_MHPE_REFINED_TASK:?HCWDL_MHPE_REFINED_TASK is required}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_mhpe_refined_recovery_task.py" \
  --recovery-spec "${HCWDL_MHPE_REFINED_RECOVERY_SPEC}" \
  --task "${HCWDL_MHPE_REFINED_TASK}" --device cuda

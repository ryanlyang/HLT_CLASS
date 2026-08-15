#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
exec python -s "${PROJECT_DIR}/scripts/autolaunch_hcwdl_unified_balanced_full_arms.py" \
  --foundation-lock "${HCWDL_UB_FULL_FOUNDATION_LOCK}" \
  --arms-root "${HCWDL_UB_FULL_ARMS_ROOT}" \
  --ledger-root "${HCWDL_UB_FULL_LEDGER_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${HCWDL_UB_FULL_SOURCE_COMMIT}" \
  --output "${HCWDL_UB_FULL_AUTOLAUNCH_RECEIPT}" \
  --authorization-phrase "AUTOLAUNCH HCWDL UB FULL3 THREE ARMS EXACT LEDGERS"

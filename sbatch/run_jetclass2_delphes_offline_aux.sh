#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?absolute source checkout}"
STAGE_SPEC="${2:?stage spec}"
TASK_ID="${3:?task ID}"
ATTEMPT_ROOT="${4:?attempt root}"
export JC2_SITE=sporc_a100
case "${PROJECT_DIR}" in /*) ;; *) echo "Absolute project required" >&2; exit 2 ;; esac
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
exec python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_offline_aux.py" run \
  --stage "${STAGE_SPEC}" --task "${TASK_ID}" --attempt "${ATTEMPT_ROOT##*/}"

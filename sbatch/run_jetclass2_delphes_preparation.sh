#!/usr/bin/env bash
set -euo pipefail

# Resources are explicit sbatch arguments, not implicit old-campaign defaults.
export PROJECT_DIR="${1:?absolute pinned project directory}"
COMMIT="${2:?exact pushed source commit}"
FOUNDATION_ROOT="${3:?absolute new foundation directory}"
DELPHES_DATA="${4:?absolute read-only new raw snapshot}"
TASK="${5:?sample, assign, lock or profile}"
export JC2_SITE="${6:?explicit execution site required}"
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"

if [[ "${TASK}" == "profile" ]]; then
  PROFILE_ROOT="${7:?fresh runtime evidence directory}"
  WORKERS="${8:?bounded preprocessing worker count}"
  MAX_TRAIN_MINUTES="${9:?explicit maximum fit-walltime envelope}"
  python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" profile \
    --foundation-root "${FOUNDATION_ROOT}" --data-root "${DELPHES_DATA}" \
    --source-commit "${COMMIT}" --output-root "${PROFILE_ROOT}" --workers "${WORKERS}" \
    --site "${JC2_SITE}" --max-train-minutes "${MAX_TRAIN_MINUTES}"
else
  EXTRA=()
  if [[ "${TASK}" == "assign" ]]; then
    EXTRA=(--array-index "${SLURM_ARRAY_TASK_ID:?assignment requires registered array index}")
  fi
  python -s "${PROJECT_DIR}/scripts/jetclass2_delphes_production.py" prepare \
    --foundation-root "${FOUNDATION_ROOT}" --data-root "${DELPHES_DATA}" \
    --source-commit "${COMMIT}" --task "${TASK}" "${EXTRA[@]}"
fi

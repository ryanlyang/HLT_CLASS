#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
: "${PROJECT_DIR:=/home/ryreu/atlas/HLT_Classification}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${PMARD_EXPLORATORY_TEST_SPEC:?PMARD_EXPLORATORY_TEST_SPEC is required}"
: "${PMARD_EXPLORATORY_TEST_TASK:?PMARD_EXPLORATORY_TEST_TASK is required}"
exec python -s "${PROJECT_DIR}/scripts/run_pmard_exploratory_test_task.py" \
  --exploratory-test-spec "${PMARD_EXPLORATORY_TEST_SPEC}" \
  --task "${PMARD_EXPLORATORY_TEST_TASK}"

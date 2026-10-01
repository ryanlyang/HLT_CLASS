#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export PYTHONPATH="${PROJECT_DIR}/src"

exec python -s \
  "${PROJECT_DIR}/scripts/run_jetclass2_cms_proxy_ladder_task.py" "$@"

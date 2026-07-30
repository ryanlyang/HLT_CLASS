#!/usr/bin/env bash
set -euo pipefail
: "${PROJECT_DIR:=/home/ryreu/atlas/HLT_Classification}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
exec python -s "${PROJECT_DIR}/scripts/submit_campaign.py" "$@"

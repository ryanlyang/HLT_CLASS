#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?absolute pinned project directory}"
PROFILE_SPEC="${2:?absolute profile attempt spec}"
export JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT_DIR}/scripts/prepare_jetclass2_delphes_debug_profile.py" run \
  --spec "${PROFILE_SPEC}"

#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?original pinned scientific project}"
RECOVERY_PROJECT="${2:?pinned recovery helper project}"
RECOVERY_SPEC="${3:?recovery spec}"
TASK="${4:?task}"
export JC2_SITE=sporc_a100
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
exec python -s "${RECOVERY_PROJECT}/scripts/recover_jetclass2_dzfix_fusion.py" \
    run --spec "${RECOVERY_SPEC}" --task "${TASK}"

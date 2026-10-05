#!/usr/bin/env bash
set -euo pipefail
export PROJECT_DIR="${1:?absolute pinned recovery project}"
RECOVERY_SPEC="${2:?immutable recovery spec}"
RECOVERY_TASK="${3:?registered replacement task}"
export JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
exec python -s "${PROJECT_DIR}/scripts/noise_k2_recovery.py" run --spec "${RECOVERY_SPEC}" --task "${RECOVERY_TASK}"

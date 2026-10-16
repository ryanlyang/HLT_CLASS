#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 3 ]]; then
    echo 'Usage: worker ABS_PROJECT ABS_SPEC TASK' >&2
    exit 2
fi
export PROJECT_DIR="$1" JC2_SITE=oscar_l40s
[[ "${PROJECT_DIR}" = /* && "$2" = /* ]]
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ARGS=(task --spec "$2" --task "$3")
if [[ "$3" = prepare ]]; then
    ARGS+=(--file-index "${SLURM_ARRAY_TASK_ID:?array index required}")
fi
exec python -u -s "${PROJECT_DIR}/scripts/luka_fullstat.py" "${ARGS[@]}"

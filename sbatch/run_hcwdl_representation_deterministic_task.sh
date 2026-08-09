#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
: "${PROJECT_DIR:?PROJECT_DIR is required}"
source "${PROJECT_DIR}/sbatch/common.sh"
hlt_activate
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
: "${HCWDL_REPRESENTATION_SPEC:?HCWDL_REPRESENTATION_SPEC is required}"
: "${HCWDL_REPRESENTATION_TASK:?HCWDL_REPRESENTATION_TASK is required}"
: "${HCWDL_REPRESENTATION_RUNTIME_BINDING:?HCWDL_REPRESENTATION_RUNTIME_BINDING is required}"

arguments=(
  --campaign-spec "${HCWDL_REPRESENTATION_SPEC}"
  --task "${HCWDL_REPRESENTATION_TASK}"
  --deterministic-worker
)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  arguments+=(--array-index "${SLURM_ARRAY_TASK_ID}")
fi

exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_representation_task.py" "${arguments[@]}"

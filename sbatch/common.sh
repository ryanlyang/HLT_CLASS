#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:=/home/ryreu/atlas/HLT_Classification}"
: "${DATA_DIR:=/home/ryreu/atlas/PracticeTagging/data}"
: "${OUTPUT_ROOT:=/home/ryreu/atlas/HLT_Classification/checkpoints}"
: "${CONDA_BASE:=/home/ryreu/miniforge3-aarch64}"
: "${CONDA_ENV:=atlas_kd_tigris}"
: "${SBATCH_ACCOUNT:=reu-aisocial}"
: "${SBATCH_PARTITION:=tigris}"
: "${GPU_GRES:=gpu:gh200:1}"

hlt_activate() {
  export PYTHONNOUSERSITE=1
  export PYTHONDONTWRITEBYTECODE=1
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  cd "${PROJECT_DIR}"
  python -s -c 'import sys; assert sys.version_info[:2] == (3, 10)'
}

hlt_run_task() {
  : "${CAMPAIGN_SPEC:?CAMPAIGN_SPEC is required}"
  : "${CAMPAIGN_TASK:?CAMPAIGN_TASK is required}"
  python -s "${PROJECT_DIR}/scripts/run_campaign_task.py" \
    --campaign-spec "${CAMPAIGN_SPEC}" \
    --task "${CAMPAIGN_TASK}"
}

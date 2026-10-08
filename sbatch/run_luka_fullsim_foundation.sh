#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --partition=debug
#SBATCH --account=reu-aisocial
#SBATCH --qos=qos_tier3
#SBATCH --export=NONE
#SBATCH --no-requeue
#SBATCH --job-name=luka_foundation
set -euo pipefail
if [[ $# -ne 4 ]]; then
    echo 'Usage: run_luka_fullsim_foundation.sh PROJECT CONTAINER OUTPUT COMMIT' >&2
    exit 2
fi
PROJECT_DIR="$1"
CONTAINER="$2"
OUTPUT="$3"
COMMIT="$4"
[[ "${PROJECT_DIR}" = /* && "${CONTAINER}" = /* && "${OUTPUT}" = /* ]]
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_DIR}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -s "${PROJECT_DIR}/scripts/luka_fullsim_foundation.py" build \
    --container "${CONTAINER}" --output "${OUTPUT}" --expected-commit "${COMMIT}"

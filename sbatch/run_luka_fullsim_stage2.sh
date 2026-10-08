#!/usr/bin/env bash
# Resource arguments are supplied explicitly by sbatch, per phase; no hidden GPU.
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo 'Usage: run_luka_fullsim_stage2.sh PROJECT COMMIT {audit|prepare|preflight} ARGS...' >&2
    exit 2
fi
PROJECT_DIR="$1"
COMMIT="$2"
MODE="$3"
shift 3
[[ "${PROJECT_DIR}" = /* && "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
case "${MODE}" in audit|prepare|preflight) ;; *) exit 2 ;; esac
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_DIR}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec python -u -s "${PROJECT_DIR}/scripts/luka_fullsim_stage2.py" "${MODE}" \
    --expected-commit "${COMMIT}" "$@"

#!/usr/bin/env bash
# Only Delphes workers source this helper. Do not change sbatch/common.sh defaults.
set -euo pipefail

case "${JC2_SITE:?explicit Delphes execution site required}" in
  sporc_a100|sporc_a100_debug)
    export CONDA_BASE=/home/ryreu/miniconda3
    export CONDA_ENV=atlas_kd_sporc
    ;;
  tigris_gh200)
    export CONDA_BASE=/home/ryreu/miniforge3-aarch64
    export CONDA_ENV=atlas_kd_tigris
    ;;
  *) echo "Unknown Delphes execution site: ${JC2_SITE}" >&2; exit 2 ;;
esac

# Do not inherit another project's Python import or loader paths from submission.
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=64
source "${PROJECT_DIR:?absolute project required}/sbatch/common.sh"
hlt_activate
export PYTHONPATH="${PROJECT_DIR}/src"

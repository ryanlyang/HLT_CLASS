#!/usr/bin/env bash
# Read saved scalar metadata; no Slurm submission or training environment mutation.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_DIR
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ -z "${AUDIT_PYTHON:-}" ]]; then
    export JC2_SITE=oscar_l40s
    source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
    AUDIT_PYTHON="${CONDA_PREFIX}/bin/python"
fi
unset PYTHONHOME PYTHONPATH
"${AUDIT_PYTHON}" -I -B "${PROJECT_DIR}/scripts/audit_luka_fullstat_pt.py" "$@"

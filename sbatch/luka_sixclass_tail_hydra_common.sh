#!/usr/bin/env bash
# Explicit persistent environment; neither modules nor package installation.
set -euo pipefail
: "${PROJECT_DIR:?absolute source required}"
: "${ENV_DIR:?explicit environment required}"
[[ "$PROJECT_DIR" == /* && "$ENV_DIR" == /* ]]
test -x "$ENV_DIR/bin/python"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
unset TORCH_CUDNN_V8_API_DISABLED
export CONDA_PREFIX="$ENV_DIR"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
cd "$PROJECT_DIR"
"$ENV_DIR/bin/python" -I -B -c 'import ssl,torch.nn.functional,uproot,scipy,threadpoolctl; from weaver.nn.model.ParticleTransformer import ParticleTransformer'

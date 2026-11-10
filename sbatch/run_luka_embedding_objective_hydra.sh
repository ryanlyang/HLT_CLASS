#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$1"
SPEC="$2"
TASK="$3"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PROJECT_DIR}/src"
exec "${ENV_DIR}/bin/python" -B -s "${PROJECT_DIR}/scripts/luka_embedding_objective_hydra.py" run --spec "$SPEC" --task "$TASK"

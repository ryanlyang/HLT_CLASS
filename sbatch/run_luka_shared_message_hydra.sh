#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]]
PROJECT_DIR="$1"
SPEC="$2"
TASK="$3"
[[ "${PROJECT_DIR}" == /* && "${SPEC}" == /* ]]
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "${SPEC}")"
[[ "${ENV_DIR}" == /* ]]
export VIRTUAL_ENV="${ENV_DIR}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd -- "${PROJECT_DIR}"
exec "${ENV_DIR}/bin/python" -u -B -s "${PROJECT_DIR}/scripts/luka_shared_message_hydra.py" run --spec "${SPEC}" --task "${TASK}"

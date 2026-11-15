#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$1"
OPERATION="$2"
TASK="$3"
PARENT=/cs/home/rlyang/hlt_u100_pairs/runs/new_balanced_4a61a34c_r1/campaign_spec.json
ENV_DIR="$(python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH PYTHONHOME
exec "$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_hydra_cache_retirement.py" \
    run-resume --operation "$OPERATION" --task "$TASK"

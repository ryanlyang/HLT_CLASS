#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 ]]
SPEC="$1"
: "${PROJECT_DIR:?absolute source required}"
: "${ENV_DIR:?persistent runtime required}"
CLI="$PROJECT_DIR/scripts/luka_d20_full_hydra.py"
"$ENV_DIR/bin/python" -I -u -B "$CLI" queue --spec "$SPEC"
"$ENV_DIR/bin/python" -I -u -B "$CLI" queue --spec "$SPEC" \
    --execute --authorize AUTHORIZE_D20_FULL_23_FOUR_1080TI_V1

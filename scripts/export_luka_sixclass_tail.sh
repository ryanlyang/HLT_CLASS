#!/usr/bin/env bash
# Run from the NEW pushed checkout on Oscar. No job submission or source changes.
set -euo pipefail
[[ $# -eq 1 && "$1" == /* ]] || { echo 'Expected absolute OUTPUT_MANIFEST.json'; exit 2; }
OUTPUT="$1"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shopt -s nullglob
LOGS=(/oscar/home/rlyang/hlt_classification/checkpoints/luka_sixclass_balanced_oscar_*/train_OFFLINE-9198207.out)
[[ ${#LOGS[@]} -eq 1 ]] || { echo 'Expected one source campaign for job 9198207'; exit 1; }
SPEC="$(dirname "${LOGS[0]}")/campaign_spec.json"
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]]
ENV_DIR="$(/usr/bin/python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PROJECT_DIR ENV_DIR
source "$PROJECT_DIR/sbatch/luka_sixclass_tail_hydra_common.sh"
exec "$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_tail_hydra.py" export --spec "$SPEC" --output "$OUTPUT"

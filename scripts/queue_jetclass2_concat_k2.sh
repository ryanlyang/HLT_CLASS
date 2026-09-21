#!/usr/bin/env bash
# Run from the exact pushed, clean pinned checkout on sporcsubmit.
# Default is a dry run. Pass --execute to authorize this registered full chain.
set -euo pipefail
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CONCAT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
CONCAT_SHORT="${CONCAT_COMMIT:0:8}"
CHECKPOINTS=/home/ryreu/atlas/HLT_Classification/checkpoints
SCREEN_SPEC="${SCREEN_SPEC:-${CHECKPOINTS}/jc2_dzfix_salience_debug_0d25a4a5_r1/screen_spec.json}"
INVENTORY="${INVENTORY:-${CHECKPOINTS}/jc2_dzfix_partial_inventory_0b3ed523_r1/inventory.json}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${CHECKPOINTS}/jc2_dzfix_concat_k2_launch_${CONCAT_SHORT}_r1}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${CHECKPOINTS}/jc2_dzfix_concat_k2_${CONCAT_SHORT}_r1}"
case "${1:-}" in
  ''|--execute) ;;
  *) echo "Usage: bash scripts/queue_jetclass2_concat_k2.sh [--execute]" >&2; exit 2 ;;
esac
test -f "${SCREEN_SPEC}"
test -f "${INVENTORY}"
if [ ! -f "${LAUNCH_ROOT}/launch_spec.json" ]; then
  python -s "${PROJECT_DIR}/scripts/jetclass2_concat_k2.py" create-launch \
    --screen-spec "${SCREEN_SPEC}" --inventory "${INVENTORY}" \
    --launch-root "${LAUNCH_ROOT}" --campaign-root "${CAMPAIGN_ROOT}" \
    --source-commit "${CONCAT_COMMIT}"
fi
CONCAT_ARGS=()
if [ "${1:-}" = --execute ]; then
  CONCAT_ARGS+=(--execute --authorization-phrase "AUTHORIZE JETCLASS2 DZFIX CONCAT K2 DEBUG 500K EXACT SPEC")
fi
python -s "${PROJECT_DIR}/scripts/jetclass2_concat_k2.py" schedule \
  --spec "${LAUNCH_ROOT}/launch_spec.json" "${CONCAT_ARGS[@]}"
echo "Launch:   ${LAUNCH_ROOT}"
echo "Campaign: ${CAMPAIGN_ROOT}"
echo "Only new jc2k2 jobs are in scope; existing campaigns are unchanged."

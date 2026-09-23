#!/usr/bin/env bash
# Run from the exact pushed, clean pinned checkout on sporcsubmit.
# Default is a dry run. Pass --execute to authorize this registered full chain.
set -euo pipefail
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
K2_PARTITION="${K2_PARTITION:-tier3}"
case "${K2_PARTITION}" in
  tier3) export JC2_SITE=sporc_a100 ;;
  debug) export JC2_SITE=sporc_a100_debug ;;
  *) echo "K2_PARTITION must be tier3 or debug" >&2; exit 2 ;;
esac
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CONCAT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
CONCAT_SHORT="${CONCAT_COMMIT:0:8}"
CHECKPOINTS=/home/ryreu/atlas/HLT_Classification/checkpoints
SCREEN_SPEC="${SCREEN_SPEC:-${CHECKPOINTS}/jc2_dzfix_salience_debug_0d25a4a5_r1/screen_spec.json}"
INVENTORY="${INVENTORY:-${CHECKPOINTS}/jc2_dzfix_partial_inventory_0b3ed523_r1/inventory.json}"
LAUNCH_ROOT="${LAUNCH_ROOT:-${CHECKPOINTS}/jc2_dzfix_concat_k2_launch_${CONCAT_SHORT}_r1}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:-${CHECKPOINTS}/jc2_dzfix_concat_k2_${CONCAT_SHORT}_r1}"
PROFILE_ARGS=()
if [ -n "${K2_PROFILE:-}" ]; then
  PROFILE_ARGS+=(--profile "${K2_PROFILE}")
fi
case "${1:-}" in
  ''|--execute) ;;
  *) echo "Usage: bash scripts/queue_jetclass2_concat_k2.sh [--execute]" >&2; exit 2 ;;
esac
test -f "${SCREEN_SPEC}"
test -f "${INVENTORY}"
REUSE_ARGS=()
if [ -n "${K2_REUSE_SPEC:-}" ]; then
  test -f "${K2_REUSE_SPEC}"
  REUSE_ARGS+=(--reuse-preparation-spec "${K2_REUSE_SPEC}")
fi
if [ ! -f "${LAUNCH_ROOT}/launch_spec.json" ]; then
  python -s "${PROJECT_DIR}/scripts/jetclass2_concat_k2.py" create-launch \
    --screen-spec "${SCREEN_SPEC}" --inventory "${INVENTORY}" \
    --launch-root "${LAUNCH_ROOT}" --campaign-root "${CAMPAIGN_ROOT}" \
    --source-commit "${CONCAT_COMMIT}" --partition "${K2_PARTITION}" "${REUSE_ARGS[@]}" "${PROFILE_ARGS[@]}"
fi
# Do not silently retarget an existing immutable launch when an environment
# variable changes. Moving an individual pending job is a separate operation.
python -s - "${LAUNCH_ROOT}/launch_spec.json" "${K2_PARTITION}" "${K2_REUSE_SPEC:-}" "${K2_PROFILE:-}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.jetclass2_delphes.concat_k2_source import validate_launch
spec = load_json(sys.argv[1])
validate_launch(spec)
if spec["registration"].get("experiment_profile", "") != sys.argv[4]:
    raise SystemExit("Existing launch has a different profile; use a fresh root.")
if spec["registration"]["execution_site"]["partition"] != sys.argv[2]:
    raise SystemExit("Existing launch has a different submission partition; use its original setting or a fresh root.")
expected = str(Path(sys.argv[3]).resolve()) if sys.argv[3] else None
actual = (spec.get("preparation_import") or {}).get("donor_spec_path")
if actual != expected:
    raise SystemExit("Existing launch has a different K2 donor; restore K2_REUSE_SPEC or use a fresh root.")
print("K2 matching: verified import from " + actual if actual else "K2 matching: fresh assignment jobs")
print("K2 physical training batch: " + str(spec["registration"]["training"]["batch_size"]))
print("K2 inference batch: " + str(spec["registration"]["inference_batch_size"]))
if sys.argv[4]:
    print("Pilot: 100k/50k, 60 maximum passes, D100 -> D050 -> D000 -> HLT-x1")
    print("All jobs use the selected partition; 24h training requests remain debug/tier3 portable.")
else:
    print("Training: tier3 only, 96h requested, 95h maximum projected fit")
    print("Only short pending jobs retain tier3/debug partition-only portability.")
print("Fresh memory preflight remains mandatory before science.")
PY
CONCAT_ARGS=()
if [ "${1:-}" = --execute ]; then
  if [ "${K2_PROFILE:-}" = pilot_100k_50k_60 ]; then
    CONCAT_ARGS+=(--execute --authorization-phrase "AUTHORIZE JETCLASS2 DZFIX CONCAT K2 PILOT 100K EXACT SPEC")
  else
    CONCAT_ARGS+=(--execute --authorization-phrase "AUTHORIZE JETCLASS2 DZFIX CONCAT K2 PORTABLE 500K EXACT SPEC")
  fi
fi
python -s "${PROJECT_DIR}/scripts/jetclass2_concat_k2.py" schedule \
  --spec "${LAUNCH_ROOT}/launch_spec.json" "${CONCAT_ARGS[@]}"
echo "Launch:   ${LAUNCH_ROOT}"
echo "Campaign: ${CAMPAIGN_ROOT}"
echo "Only new jc2k2 jobs are in scope; existing campaigns are unchanged."

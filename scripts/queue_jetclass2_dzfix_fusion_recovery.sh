#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---dry-run}"
[[ "$MODE" == --dry-run || "$MODE" == --execute ]] || { echo 'Use --dry-run or --execute'; exit 2; }
RECOVERY_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECOVERY_COMMIT="$(git -C "$RECOVERY_PROJECT" rev-parse HEAD)"
SOURCE_SPEC="${SOURCE_SPEC:-/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_fusion_tier3_coarsefd1c0578_r1/campaign_spec.json}"
RECOVERY_ROOT="${RECOVERY_ROOT:-${SOURCE_SPEC%/*}/recoveries/final_direct_nodefail_${RECOVERY_COMMIT:0:8}_r1}"
test -f "$SOURCE_SPEC"
export PROJECT_DIR="$(python -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SOURCE_SPEC")"
export JC2_SITE=sporc_a100
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CLI="${RECOVERY_PROJECT}/scripts/recover_jetclass2_dzfix_fusion.py"
if [[ ! -e "${RECOVERY_ROOT}/recovery_spec.json" ]]; then
    python -s "$CLI" create --source-spec "$SOURCE_SPEC" --recovery-root "$RECOVERY_ROOT" \
        --source-commit "$RECOVERY_COMMIT"
fi
python -s "$CLI" submit --spec "${RECOVERY_ROOT}/recovery_spec.json"
if [[ "$MODE" == --execute ]]; then
    python -s "$CLI" submit --spec "${RECOVERY_ROOT}/recovery_spec.json" --execute \
        --authorization-phrase 'AUTHORIZE DZFIX FUSION FINAL DIRECT NODE FAILURE RECOVERY'
fi
echo "Recovery: ${RECOVERY_ROOT}"
echo 'Only final-direct plus aggregate/complete replacements. The fusion branch is retained.'

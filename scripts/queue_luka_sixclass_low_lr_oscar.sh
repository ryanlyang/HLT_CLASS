#!/usr/bin/env bash
# Execute from an existing clone after pushing the exact implementation commit.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Expected pushed COMMIT and currently free HOME GiB'; exit 2; }
COMMIT="$1"; QUOTA="$2"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
(( QUOTA >= 8 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET=/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1
ENV_DIR="${BASE}/environments/new_balanced_oscar_v1"
PROJECT="${BASE}/source/HLT_Classification_sixclass_low_lr_oscar_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_sixclass_low_lr_oscar_${COMMIT:0:8}_r1"
test -f "${BASE}/checkpoints/luka_sixclass_balanced_oscar_8a25f126_r1/campaign_spec.json"
test -f "$DATASET/transfer_complete.json"
test -x "$ENV_DIR/bin/python"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
exec bash "$PROJECT/scripts/launch_luka_sixclass_low_lr_oscar.sh" detach \
    "$PROJECT" "$DATASET" "$ROOT" "$COMMIT" "$QUOTA" "$ENV_DIR" \
    --authorize AUTHORIZE_SIXCLASS_LOW_LR_OSCAR_V1

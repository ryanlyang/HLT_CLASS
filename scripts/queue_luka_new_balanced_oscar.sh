#!/usr/bin/env bash
# One-paste entrypoint after committing and pushing this campaign.
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Expected full pushed COMMIT and available HOME GiB'; exit 2; }
COMMIT="$1"; QUOTA="$2"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
(( QUOTA >= 14 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET=/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1
PROJECT="${BASE}/source/HLT_Classification_new_balanced_oscar_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_new_balanced_oscar_${COMMIT:0:8}_r1"
ENV_DIR="${LUKA_OSCAR_ENV:-${BASE}/environments/new_balanced_oscar_v1}"
test -x "$ENV_DIR/bin/python" || { echo 'STOP: provision the persistent HOME runtime first (restore_luka_oscar_runtime.py).'; exit 1; }
test -f "$DATASET/transfer_complete.json"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
exec bash "$PROJECT/scripts/launch_luka_new_balanced_oscar.sh" detach \
    "$PROJECT" "$DATASET" "$ROOT" "$COMMIT" "$QUOTA" "$ENV_DIR" \
    --authorize AUTHORIZE_NEW_BALANCED_OSCAR_SIX_V1

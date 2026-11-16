#!/usr/bin/env bash
# One-paste entrypoint after committing and pushing this campaign.
set -euo pipefail
[[ $# -eq 2 || $# -eq 3 ]] || { echo 'Expected pushed COMMIT, current free HOME GiB, optional healthy base CONDA_EXE'; exit 2; }
COMMIT="$1"; QUOTA="$2"; RUNTIME_CONDA="${3:-}"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
(( QUOTA >= 14 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET=/oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1
PROJECT="${BASE}/source/HLT_Classification_sixclass_balanced_oscar_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_sixclass_balanced_oscar_${COMMIT:0:8}_r1"
test -f "$DATASET/transfer_complete.json"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
exec bash "$PROJECT/scripts/bootstrap_luka_sixclass_balanced_oscar.sh" detach \
    "$COMMIT" "$QUOTA" "$RUNTIME_CONDA" \
    --authorize AUTHORIZE_SIXCLASS_BALANCED_OSCAR_V1

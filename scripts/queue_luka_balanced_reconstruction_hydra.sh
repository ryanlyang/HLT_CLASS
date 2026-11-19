#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 2 ]] || { echo 'Expected pushed40-character COMMIT and CURRENT_FREE_GIB'; exit 2; }
COMMIT="$1"; QUOTA="$2"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$BASE/code_balanced_reco_${COMMIT:0:8}"
ROOT="$BASE/runs/balanced_reco_600k_${COMMIT:0:8}_r1"
ENV_DIR="$BASE/environment_pair_hydra_v1/venv"
test -f "$BASE/datasets/luka_qcdrun2com_higgs2p_sb_v1/transfer_complete.json"
test -f "$BASE/runs/sixclass_tail_504a8314_r1/campaign_spec.json"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
exec bash "$PROJECT/scripts/launch_luka_balanced_reconstruction_hydra.sh" detach \
    "$PROJECT" "$ROOT" "$COMMIT" "$QUOTA" "$ENV_DIR" \
    --authorize AUTHORIZE_BALANCED_RECONSTRUCTION_18_FOUR_1080TI_V1

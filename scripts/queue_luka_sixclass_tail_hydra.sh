#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 3 ]] || { echo 'Expected pushed COMMIT, BUNDLE_HASH, free HOME GiB AFTER transfer'; exit 2; }
COMMIT="$1"; BUNDLE="$2"; QUOTA="$3"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$BUNDLE" =~ ^[0-9a-f]{64}$ && "$QUOTA" =~ ^[0-9]+$ ]]
(( QUOTA >= 14 ))
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${BASE}/code_sixclass_tail_${COMMIT:0:8}"
ROOT="${BASE}/runs/sixclass_tail_${COMMIT:0:8}_r1"
IMPORT="${BASE}/imports/sixclass_oscar_${BUNDLE:0:16}"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
test -f "$IMPORT/transfer_complete.json"
test -f "$BASE/datasets/luka_qcdrun2com_higgs2p_sb_v1/transfer_complete.json"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
exec bash "$PROJECT/scripts/launch_luka_sixclass_tail_hydra.sh" detach \
    "$PROJECT" "$ROOT" "$COMMIT" "$QUOTA" "$ENV_DIR" "$IMPORT" "$BUNDLE" \
    --authorize AUTHORIZE_SIXCLASS_TAIL_HYDRA_V1

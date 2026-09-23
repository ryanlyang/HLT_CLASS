#!/usr/bin/env bash
# Separate 100k/50k pilot: dry by default, automatic gated release with --execute.
set -euo pipefail
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PILOT_COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
PILOT_SHORT="${PILOT_COMMIT:0:8}"
PILOT_CHECKPOINTS=/home/ryreu/atlas/HLT_Classification/checkpoints
export K2_PROFILE=pilot_100k_50k_60
# Separate names prevent stale full-campaign shell variables changing this pilot.
export K2_PARTITION="${K2_PILOT_PARTITION:-debug}"
export K2_REUSE_SPEC="${K2_PILOT_REUSE_SPEC:-${PILOT_CHECKPOINTS}/jc2_dzfix_concat_k2_1f930650_r1/campaign_spec.json}"
export LAUNCH_ROOT="${K2_PILOT_LAUNCH_ROOT:-${PILOT_CHECKPOINTS}/jc2_dzfix_concat_k2_pilot100k_launch_${PILOT_SHORT}_r1}"
export CAMPAIGN_ROOT="${K2_PILOT_CAMPAIGN_ROOT:-${PILOT_CHECKPOINTS}/jc2_dzfix_concat_k2_pilot100k_${PILOT_SHORT}_r1}"
exec bash "${PROJECT_DIR}/scripts/queue_jetclass2_concat_k2.sh" "$@"

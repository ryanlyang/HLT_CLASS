#!/usr/bin/env bash
# Run from a separate controller worktree. Never update the pending S3 source.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve files and paste the error." >&2' ERR
if [[ $# != 2 && $# != 3 ]]; then
    echo 'Usage: helper CONTROLLER_COMMIT PREFLIGHT_JOB [--execute]' >&2
    exit 2
fi
COMMIT=$1 JOB=$2
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$JOB" =~ ^[1-9][0-9]*$ ]]
if [[ $# == 3 ]]; then [[ "$3" == --execute ]]; fi
CONTROLLER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test "$(git -C "$CONTROLLER_DIR" rev-parse HEAD)" = "$COMMIT"
test -z "$(git -C "$CONTROLLER_DIR" status --porcelain --untracked-files=all)"
git -C "$CONTROLLER_DIR" merge-base --is-ancestor "$COMMIT" origin/main
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="$CONTROLLER_DIR/scripts/jetclass2_s3_ladder_followup.py"
if [[ $# == 2 ]]; then
    "${CONDA_PREFIX}/bin/python" -u -s "$CLI" --controller-commit "$COMMIT" --preflight-job "$JOB"
else
    LOG="$(mktemp /home/ryreu/atlas/HLT_Classification/checkpoints/s3-science-followup.XXXXXX.log)"
    nohup "${CONDA_PREFIX}/bin/python" -u -s "$CLI" --controller-commit "$COMMIT" \
        --preflight-job "$JOB" --execute >"$LOG" 2>&1 </dev/null &
    printf 'Controller PID: %s\nLog: %s\n' "$!" "$LOG"
    printf 'Confirm ARMED (or inspect any error):\ntail -f "%s"\n' "$LOG"
    printf 'Ctrl-C stops tail only. The controller survives SSH disconnects, not host reboots.\n'
fi

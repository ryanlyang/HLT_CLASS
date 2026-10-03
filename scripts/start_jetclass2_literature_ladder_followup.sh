#!/usr/bin/env bash
# Run from a separate controller checkout, NOT by updating the active gate worktree.
set -euo pipefail
if [[ $# != 4 ]]; then
    echo 'Usage: start_jetclass2_literature_ladder_followup.sh GATE_SPEC GATE_HASH SCIENCE_COMMIT PREFLIGHT_JOB' >&2
    exit 2
fi
SPEC="$1"
GATE_HASH="$2"
SCIENCE_COMMIT="$3"
PREFLIGHT_JOB="$4"
[[ "${GATE_HASH}" =~ ^[0-9a-f]{64}$ && "${SCIENCE_COMMIT}" =~ ^[0-9a-f]{40}$ && "${PREFLIGHT_JOB}" =~ ^[1-9][0-9]*$ ]]
test -f "${SPEC}"
CONTROLLER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
STUDY="$(cd -- "$(dirname -- "${SPEC}")/.." && pwd -P)"
LOG="$(mktemp "${STUDY}/science-followup.XXXXXX.log")"
nohup "${CONDA_PREFIX}/bin/python" -u -s \
    "${CONTROLLER_DIR}/scripts/jetclass2_literature_ladder_followup.py" \
    --gate-spec "${SPEC}" --gate-hash "${GATE_HASH}" --source-commit "${SCIENCE_COMMIT}" \
    --preflight-job "${PREFLIGHT_JOB}" --execute >"${LOG}" 2>&1 </dev/null &
printf 'Follow-up PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Confirm it prints ARMED (or inspect any error):\ntail -f "%s"\n' "${LOG}"
printf 'Ctrl-C stops tail only. Once armed, the controller survives SSH disconnects.\n'

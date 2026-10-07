#!/usr/bin/env bash
# Separate NEW pinned checkout. Never update the active gate checkout.
set -euo pipefail
if [[ $# != 4 && $# != 5 ]]; then
    echo 'Usage: start_jetclass2_correlated_tier3_followup.sh EXECUTOR_COMMIT GATE_SPEC GATE_HASH PREFLIGHT_JOB [--execute]' >&2
    exit 2
fi
COMMIT="$1"
SPEC="$2"
GATE_HASH="$3"
PREFLIGHT="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${GATE_HASH}" =~ ^[0-9a-f]{64}$ && "${PREFLIGHT}" =~ ^[1-9][0-9]*$ ]]
test -f "${SPEC}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
export PROJECT_DIR JC2_SITE=sporc_a100
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
ARGS=("${PROJECT_DIR}/scripts/jetclass2_correlated_tier3.py" followup
      --gate-spec "${SPEC}" --gate-hash "${GATE_HASH}" --preflight-job "${PREFLIGHT}"
      --project-dir "${PROJECT_DIR}" --executor-commit "${COMMIT}")
if [[ $# == 4 ]]; then
    python -u -s "${ARGS[@]}"
    exit 0
fi
[[ "$5" == --execute ]]
STUDY="$(cd -- "$(dirname -- "${SPEC}")/.." && pwd -P)"
LOG="$(mktemp "${STUDY}/tier3-followup.XXXXXX.log")"
nohup "${CONDA_PREFIX}/bin/python" -u -s "${ARGS[@]}" --execute >"${LOG}" 2>&1 </dev/null &
printf 'Follow-up PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Confirm ARMED or inspect the error:\ntail -f "%s"\n' "${LOG}"
printf 'Ctrl-C stops tail only. The armed controller survives SSH disconnects.\n'

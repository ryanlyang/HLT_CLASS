#!/usr/bin/env bash
# Separate executor checkout; NEVER update the active gate checkout.
set -euo pipefail
if [[ $# != 5 && $# != 6 ]]; then
    echo 'Usage: start_jetclass2_context_full_followup.sh EXECUTOR_COMMIT GATE_SPEC GATE_HASH SCIENCE_COMMIT PREFLIGHT_JOB [--execute]' >&2
    exit 2
fi
COMMIT="$1"
SPEC="$2"
GATE_HASH="$3"
SCIENCE_COMMIT="$4"
PREFLIGHT="$5"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${SCIENCE_COMMIT}" =~ ^[0-9a-f]{40}$ && "${GATE_HASH}" =~ ^[0-9a-f]{64}$ && "${PREFLIGHT}" =~ ^[1-9][0-9]*$ ]]
test -f "${SPEC}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
export PROJECT_DIR JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
ARGS=("${PROJECT_DIR}/scripts/jetclass2_context_full_followup.py"
      --gate-spec "${SPEC}" --gate-hash "${GATE_HASH}" --source-commit "${SCIENCE_COMMIT}"
      --preflight-job "${PREFLIGHT}" --project-dir "${PROJECT_DIR}" --executor-commit "${COMMIT}")
if [[ $# == 5 ]]; then
    python -u -s "${ARGS[@]}"
    exit 0
fi
[[ "$6" == --execute ]]
STUDY="$(cd -- "$(dirname -- "${SPEC}")/.." && pwd -P)"
LOG="$(mktemp "${STUDY}/context1m-followup.XXXXXX.log")"
nohup bash -c '
    result=0
    "$@" || result=$?
    printf "\nFOLLOWUP_EXIT=%s\n" "${result}"
    exit "${result}"
' _ "${CONDA_PREFIX}/bin/python" -u -s "${ARGS[@]}" --execute >"${LOG}" 2>&1 </dev/null &
printf 'Follow-up PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Wait for ARMED or inspect the error:\ntail -f "%s"\n' "${LOG}"
printf 'Ctrl-C stops tail only. Do not also submit MODE=science manually.\n'

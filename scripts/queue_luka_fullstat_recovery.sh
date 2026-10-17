#!/usr/bin/env bash
# Repository-owned launcher: no pasted Python/heredoc submission fragments.
set -euo pipefail
if [[ "$#" -lt 3 ]]; then
    echo 'Usage: helper review COMMIT RECOVERY_ROOT ORIGINAL_GATE | arm COMMIT RECOVERY_ROOT REVIEW_HASH' >&2
    exit 2
fi
MODE="$1"
COMMIT="$2"
ROOT="$3"
[[ "$#" -eq 4 && "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" = /* ]]
export PROJECT_DIR JC2_SITE=oscar_l40s
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "${MODE}" = review ]]; then
    exec python -u -s "${PROJECT_DIR}/scripts/luka_fullstat_recovery.py" review \
        --commit "${COMMIT}" --root "${ROOT}" --original-gate "$4"
elif [[ "${MODE}" = arm ]]; then
    [[ "$4" =~ ^[0-9a-f]{64}$ ]]
    test -f "${ROOT}/followup_review.json"
    read -r -p 'Authorize recovery AND all 16 science jobs after the full gate passes? Type YES: ' CONFIRM </dev/tty
    test "${CONFIRM}" = YES
    LOG="$(mktemp "${ROOT}/recovery-followup.XXXXXX.log")"
    nohup python -u -s "${PROJECT_DIR}/scripts/luka_fullstat_recovery.py" run \
        --commit "${COMMIT}" --root "${ROOT}" --review-hash "$4" \
        --authorize 'AUTHORIZE FULLSTAT SPARSE RECOVERY AND AUTOMATIC SCIENCE' \
        >"${LOG}" 2>&1 </dev/null &
    printf 'Controller PID: %s\nLog: %s\nWait for ARMED or inspect the error:\ntail -f "%s"\n' "$!" "${LOG}" "${LOG}"
    echo 'Ctrl-C stops tail only. Do not submit science manually while this controller is armed.'
else
    echo 'Use review or arm.' >&2
    exit 2
fi

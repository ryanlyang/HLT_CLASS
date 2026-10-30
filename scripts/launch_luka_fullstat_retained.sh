#!/usr/bin/env bash
# Detach before costly source/data authentication. No retry or old-job changes.
set -euo pipefail
BACKGROUND=false
if [[ "${1:-}" == --background ]]; then BACKGROUND=true; shift; fi
if [[ $# -ne 6 || "$5" != --authorize || "$6" != AUTHORIZE_FULLSTAT_RETAINED_AUTOSUBMIT_V1 ]]; then
    echo 'Usage: launch COMMIT ROOT COMPLETED_READOUT_SPEC RESULT_QUOTA_GIB --authorize AUTHORIZE_FULLSTAT_RETAINED_AUTOSUBMIT_V1' >&2
    exit 2
fi
COMMIT="$1"; ROOT="$2"; PARENT="$3"; QUOTA="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${PARENT}" == /* ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 9 ))
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
SCRIPT="${PROJECT_DIR}/scripts/launch_luka_fullstat_retained.sh"
LAUNCH="${ROOT}.launch"
if [[ "${BACKGROUND}" == false ]]; then
    test -r "${PARENT}"
    test ! -e "${ROOT}"
    test -d "$(dirname "${ROOT}")"
    command -v nohup >/dev/null
    command -v setsid >/dev/null
    mkdir -- "${LAUNCH}"
    LOG="${LAUNCH}/launcher.log"
    : >"${LOG}"
    trap '' HUP
    nohup setsid bash "${SCRIPT}" --background "$@" >>"${LOG}" 2>&1 </dev/null &
    printf 'DETACHED: launcher PID %s\nStudy: %s\nLog: %s\n' "$!" "${ROOT}" "${LOG}"
    trap - HUP
    printf 'You may disconnect now; this tail is only a view. Do not launch again.\n'
    printf 'Reconnect with: tail -n 60 -f "%s"\n' "${LOG}"
    exec tail -n +1 -f -- "${LOG}"
fi
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
test -d "${LAUNCH}"
test ! -e "${LAUNCH}/process_claim"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_retained.py" launch \
    --root "${ROOT}" --previous-spec "${PARENT}" --expected-commit "${COMMIT}" \
    --result-quota-gib "${QUOTA}" --authorization "$6"


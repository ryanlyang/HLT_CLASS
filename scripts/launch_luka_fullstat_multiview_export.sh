#!/usr/bin/env bash
# One fresh export, then one unchanged student fit. No teacher retraining.
set -euo pipefail
BACKGROUND=false
if [[ "${1:-}" == --background ]]; then BACKGROUND=true; shift; fi
if [[ $# -ne 6 || "$5" != --authorize || "$6" != AUTHORIZE_FULLSTAT_MULTIVIEW_EXPORT_AUTOSUBMIT_V1 ]]; then
    echo 'Usage: launch COMMIT FRESH_ROOT FAILED_MULTIVIEW_SPEC RESULT_QUOTA_GIB --authorize AUTHORIZE_FULLSTAT_MULTIVIEW_EXPORT_AUTOSUBMIT_V1' >&2
    exit 2
fi
COMMIT="$1"; ROOT="$2"; PARENT="$3"; QUOTA="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${PARENT}" == /* ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 22 ))
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
SCRIPT="${PROJECT_DIR}/scripts/launch_luka_fullstat_multiview_export.sh"
LAUNCH="${ROOT}.launch"
if [[ "${BACKGROUND}" == false ]]; then
    test -r "${PARENT}"
    [[ ! -e "${ROOT}" && ! -L "${ROOT}" && ! -e "${LAUNCH}" && ! -L "${LAUNCH}" ]]
    test -d "$(dirname "${ROOT}")"
    command -v nohup >/dev/null
    command -v setsid >/dev/null
    mkdir -- "${LAUNCH}"
    LOG="${LAUNCH}/launcher.log"
    : >"${LOG}"
    trap '' HUP
    nohup setsid bash "${SCRIPT}" --background "$@" >>"${LOG}" 2>&1 </dev/null &
    printf 'DETACHED: export-recovery launcher PID %s\nStudy: %s\nLog: %s\n' "$!" "${ROOT}" "${LOG}"
    trap - HUP
    printf 'You may disconnect now; this tail is only a view. Do not launch again.\n'
    printf 'Reconnect with: tail -n 60 -f "%s"\n' "${LOG}"
    exec tail -n +1 -f -- "${LOG}"
fi
trap 'code=$?; printf "LAUNCHER_EXIT=%s\n" "$code"' EXIT
[[ -d "${LAUNCH}" && ! -L "${LAUNCH}" && ! -e "${ROOT}" && ! -L "${ROOT}" ]]
test ! -e "${LAUNCH}/process_claim"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
export PYTHONFAULTHANDLER=1
python -X faulthandler -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_multiview_export.py" launch \
    --root "${ROOT}" --previous-spec "${PARENT}" --expected-commit "${COMMIT}" \
    --result-quota-gib "${QUOTA}" --authorization "$6"

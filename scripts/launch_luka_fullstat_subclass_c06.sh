#!/usr/bin/env bash
set -euo pipefail
BACKGROUND=false
if [[ "${1:-}" == --background ]]; then BACKGROUND=true; shift; fi
if [[ ( $# -ne 6 && $# -ne 8 ) || "$5" != --authorize || "$6" != AUTHORIZE_FULLSTAT_SUBCLASS_C06_AUTOSUBMIT_V1 ]]; then
    echo 'Usage: launch COMMIT FRESH_ROOT COMPLETED_SUBCLASS_C03_SPEC QUOTA_GIB --authorize AUTHORIZE_FULLSTAT_SUBCLASS_C06_AUTOSUBMIT_V1' >&2
    exit 2
fi
COMMIT="$1"; ROOT="$2"; PARENT="$3"; QUOTA="$4"
EXTRA=()
if [[ $# -eq 8 ]]; then
    [[ "$7" == --resume-spec && "$8" == /* ]]
    test -r "$8"
    EXTRA=(--resume-spec "$8")
fi
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${PARENT}" == /* ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 8 ))
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
SCRIPT="${PROJECT_DIR}/scripts/launch_luka_fullstat_subclass_c06.sh"
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
    printf 'DETACHED: contrastive0.6/subclass0.5 launcher PID %s\nStudy: %s\nLog: %s\n' "$!" "${ROOT}" "${LOG}"
    trap - HUP
    printf 'You may disconnect now; authentication, dry review and submission continue. Do not launch again.\n'
    printf 'Watch later: tail -n 60 -f "%s"\n' "${LOG}"
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
python -X faulthandler -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_subclass_c06.py" launch \
    --root "${ROOT}" --previous-spec "${PARENT}" --expected-commit "${COMMIT}" \
    --result-quota-gib "${QUOTA}" --authorization "$6" "${EXTRA[@]}"

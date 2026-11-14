#!/usr/bin/env bash
set -euo pipefail
BACKGROUND=false
if [[ "${1:-}" == --background ]]; then BACKGROUND=true; shift; fi
if [[ ( $# -ne 6 && $# -ne 7 ) || "$5" != --authorize || "$6" != AUTHORIZE_FULLSTAT_BALANCED_LADDER_AUTOSUBMIT_V1 ]]; then
    echo 'Usage: COMMIT FRESH_ROOT PARENT_SPEC QUOTA_GIB --authorize AUTHORIZE_FULLSTAT_BALANCED_LADDER_AUTOSUBMIT_V1 [--recover]' >&2
    exit 2
fi
COMMIT="$1"; ROOT="$2"; PARENT="$3"; QUOTA="$4"; EXTRA=()
if [[ $# -eq 7 ]]; then [[ "$7" == --recover ]]; EXTRA=(--recover); fi
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${PARENT}" == /* ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 36 ))
export PROJECT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
SCRIPT="${PROJECT_DIR}/scripts/launch_luka_fullstat_balanced_ladder.sh"
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
    printf 'DETACHED: balanced launcher PID %s\nStudy: %s\nLog: %s\n' "$!" "${ROOT}" "${LOG}"
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
# Check real modules, not namespace-only imports; catches the past scratch-runtime damage.
python -I -B -c 'import ssl, sqlite3, bz2, lzma, numpy, scipy, awkward, uproot, sklearn.metrics, threadpoolctl, torch.nn.functional, weaver.nn.model.ParticleTransformer; print("RUNTIME_IMPORTS_OK", flush=True)'
python -X faulthandler -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_balanced_ladder.py" launch \
    --root "${ROOT}" --parent-spec "${PARENT}" --expected-commit "${COMMIT}" \
    --result-quota-gib "${QUOTA}" --authorization "$6" "${EXTRA[@]}"

#!/usr/bin/env bash
# Run on the original controller host. No new preflight or scientific source.
set -euo pipefail
if [[ $# -ne 5 && $# -ne 7 ]]; then
    echo 'Usage: helper COMMIT ROOT ORIGINAL_PROJECT CONTROLLER_PID CONTROLLER_HOST [--execute HASH]' >&2
    exit 2
fi
COMMIT="$1"
ROOT="$2"
ORIGINAL="$3"
CONTROLLER_PID="$4"
CONTROLLER_HOST="$5"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${ROOT}" == /* && "${ORIGINAL}" == /* ]]
[[ "${CONTROLLER_PID}" =~ ^[1-9][0-9]*$ ]]
test "$(hostname -s)" = "${CONTROLLER_HOST}"
TOOL_PROJECT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${TOOL_PROJECT}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${TOOL_PROJECT}" status --porcelain --untracked-files=all)"
git -C "${TOOL_PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
test "$(git -C "${ORIGINAL}" rev-parse HEAD)" = fd687181a60c1557f98d7498f577e26d6dbb35a7
test -z "$(git -C "${ORIGINAL}" status --porcelain --untracked-files=all)"
export PROJECT_DIR="${ORIGINAL}"
export JC2_SITE=oscar_l40s
unset PYTHONHOME PYTHONPATH LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ARGS=(--commit "${COMMIT}" --gate "${ROOT}/gate_spec.json"
      --controller-pid "${CONTROLLER_PID}" --controller-host "${CONTROLLER_HOST}")
if [[ $# -eq 7 ]]; then
    [[ "$6" == --execute && "$7" =~ ^[0-9a-f]{64}$ ]]
    ARGS+=(--execute --plan-hash "$7" --authorization AUTHORIZE_FULLSTAT_EMBEDDING_ORDER_V2)
fi
exec python -u -B -s "${TOOL_PROJECT}/scripts/order_luka_fullstat_embedding_sweep.py" "${ARGS[@]}"

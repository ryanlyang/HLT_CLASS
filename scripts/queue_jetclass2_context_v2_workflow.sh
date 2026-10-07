#!/usr/bin/env bash
# Run on Oscar from a clean, pushed detached worktree. No pilot or manual science followup.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve the workflow and journals." >&2' ERR

if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo 'Usage: bash queue_jetclass2_context_v2_workflow.sh COMMIT ROOT DATASET_ROOT AVAILABLE_QUOTA_GIB [--execute PLAN_HASH]' >&2
    exit 2
fi
COMMIT=$1
ROOT=$2
DATASET_ROOT=$3
QUOTA=$4
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${ROOT}" = /* && "${DATASET_ROOT}" = /* ]]
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_DIR JC2_SITE=oscar_l40s
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CLI="${PROJECT_DIR}/scripts/jetclass2_context_v2_workflow.py"
SPEC="${ROOT}/workflow_spec.json"
ORIGINAL=/oscar/home/rlyang/hlt_classification/checkpoints/jc2_context_100k50k_direct_coarse_f889f359_r1/gate/gate_spec.json

if [[ -f "${SPEC}" ]]; then
    python -s - "${SPEC}" "${PROJECT_DIR}" "${COMMIT}" "${ROOT}" "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text())
for name, expected in zip(('project_dir', 'source_commit', 'root', 'dataset_root'), sys.argv[2:]):
    if spec[name] != expected:
        raise SystemExit('STOP: existing workflow differs: '+name)
PY
fi

if [[ $# -eq 6 ]]; then
    test "$5" = --execute
    [[ "$6" =~ ^[0-9a-f]{64}$ ]]
    test -f "${SPEC}"
    python -u -s "${CLI}" submit --spec "${SPEC}" --execute --reviewed-hash "$6" \
        --authorize 'AUTHORIZE CONTEXT V2 DATASET AND DIRECT COARSE OSCAR WORKFLOW'
else
    if [[ -f "${SPEC}" ]]; then
        python -u -s "${CLI}" plan --spec "${SPEC}"
    else
        test ! -e "${ROOT}"
        test ! -e "${DATASET_ROOT}"
        python -u -s "${CLI}" create --original-gate "${ORIGINAL}" --root "${ROOT}" \
            --dataset-root "${DATASET_ROOT}" --project-dir "${PROJECT_DIR}" --commit "${COMMIT}" \
            --available-quota-gib "${QUOTA}" --persistent
    fi
    echo 'Dry review only. Review the plan including automatic 16-job science policy, then repeat with --execute HASH.'
fi

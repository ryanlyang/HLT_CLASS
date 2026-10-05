#!/usr/bin/env bash
# Dry-first, explicit sidecar migration. Never edit the original spec/worktree.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?absolute new pushed executor checkout}"
DONOR="${K2_SEGMENT_DONOR_SPEC:?original 71c1bde2 segmented campaign spec}"
ROOT="${K2_128G_ROOT:?fresh 128-GiB continuation root}"
COMMIT="${K2_128G_COMMIT:?full pushed executor commit}"
[[ $# -le 1 && ( "${1:-}" == '' || "${1:-}" == --execute ) ]] || { echo 'Only --execute is supported' >&2; exit 2; }
[[ "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" == "${COMMIT}" ]] || { echo 'Wrong executor commit' >&2; exit 2; }
export JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CLI="${PROJECT_DIR}/scripts/jetclass2_k2_segmented.py"
if [[ ! -e "${ROOT}" ]]; then
  python -s "${CLI}" create-128g --donor-spec "${DONOR}" --campaign-root "${ROOT}" --source-commit "${COMMIT}"
fi
SPEC="${ROOT}/campaign_spec.json"
python -s - "${SPEC}" "${PROJECT_DIR}" "${COMMIT}" "${DONOR}" <<'PY'
import json
from pathlib import Path
import sys
s = json.loads(Path(sys.argv[1]).read_text())
if (s['contract'] != 'K2_SEGMENTED_CAMPAIGN_SPEC/v2'
        or Path(s['project_dir']).resolve() != Path(sys.argv[2]).resolve()
        or s['source_commit'] != sys.argv[3]
        or Path(s['resume_import']['donor_spec']['path']).resolve() != Path(sys.argv[4]).resolve()):
    raise SystemExit('Existing root does not match requested 128-GiB migration')
print('D025 resumes after pass', s['resume_import']['checkpoint']['pass_number'])
print('D025 already complete:', s['resume_import']['checkpoint']['fit_complete'])
print('128 GiB host RAM; batch128 and all scientific settings unchanged.')
print('Only the exact registered pending segmented remainder is in scope.')
PY
python -s "${CLI}" retire --spec "${SPEC}"
python -s "${CLI}" submit --spec "${SPEC}" --stage full
if [[ "${1:-}" == --execute ]]; then
  [[ "${K2_DEBUG_POLICY_CONFIRMED:-}" == yes ]] || { echo 'Confirm RC permits segmented production on debug first.' >&2; exit 2; }
  python -s "${CLI}" retire --spec "${SPEC}" --execute --authorization-phrase 'CANCEL ONLY REGISTERED PENDING K2 REMAINDER'
  python -s "${CLI}" submit --spec "${SPEC}" --stage gate --execute --confirm-debug-policy --authorization-phrase 'AUTHORIZE K2 SEGMENTED DEBUG CONTINUATION EXACT SPEC'
fi

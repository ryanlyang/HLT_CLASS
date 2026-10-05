#!/usr/bin/env bash
# Dry by default. Reuse the original native gate; never submit a new preflight.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?absolute new pushed executor checkout}"
ROOT="${K2_RESTORE_ROOT:?fresh restoration root}"
UNUSED="${K2_ABANDONED_SPEC:?unused c640942f 128-GiB campaign spec}"
COMMIT="${K2_RESTORE_COMMIT:?full pushed executor commit}"
[[ $# -le 1 && ( "${1:-}" == '' || "${1:-}" == --execute ) ]] || { echo 'Only --execute is supported' >&2; exit 2; }
[[ "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" == "${COMMIT}" ]] || { echo 'Wrong executor commit' >&2; exit 2; }
export JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CLI="${PROJECT_DIR}/scripts/jetclass2_k2_segmented.py"
if [[ ! -e "${ROOT}" ]]; then
  python -s "${CLI}" create-restored --abandoned-spec "${UNUSED}" --campaign-root "${ROOT}" --source-commit "${COMMIT}"
fi
SPEC="${ROOT}/campaign_spec.json"
python -s - "${SPEC}" "${PROJECT_DIR}" "${COMMIT}" "${UNUSED}" <<'PY'
import json
from pathlib import Path
import sys
s = json.loads(Path(sys.argv[1]).read_text())
if (s['contract'] != 'K2_SEGMENTED_CAMPAIGN_SPEC/v3'
        or Path(s['project_dir']).resolve() != Path(sys.argv[2]).resolve()
        or s['source_commit'] != sys.argv[3]
        or Path(s['abandoned_execution']['spec']['path']).resolve() != Path(sys.argv[4]).resolve()):
    raise SystemExit('Existing root does not match requested restoration')
print('Original 312.5-GiB GPU-job envelope; original native gate REUSED.')
print('Resume D025 after epoch:', s['resume_import']['checkpoint']['pass_number'])
print('Eleven science jobs; no preflight, after-gate, matching, or D025 parts1/2.')
PY
if [[ ! -f "${ROOT}/reused_native_gate.json" ]]; then
  python -s "${CLI}" prepare-restored --spec "${SPEC}"
fi
python -s "${CLI}" gate --spec "${SPEC}"
python -s "${CLI}" retire --spec "${SPEC}"
python -s "${CLI}" submit --spec "${SPEC}" --stage full
if [[ "${1:-}" == --execute ]]; then
  [[ "${K2_DEBUG_POLICY_CONFIRMED:-}" == yes ]] || { echo 'Confirm RC permits segmented production on debug first.' >&2; exit 2; }
  python -s "${CLI}" retire --spec "${SPEC}" --execute --authorization-phrase 'CANCEL ONLY REGISTERED PENDING K2 REMAINDER'
  python -s "${CLI}" submit --spec "${SPEC}" --stage science --execute --confirm-debug-policy --authorization-phrase 'AUTHORIZE K2 SEGMENTED DEBUG CONTINUATION EXACT SPEC'
fi

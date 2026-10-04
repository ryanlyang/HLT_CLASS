#!/usr/bin/env bash
# No cancellations or live submission by default. Reuse the immutable old root.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:?absolute new pushed executor checkout}"
SOURCE_SPEC="${K2_SOURCE_SPEC:?original c891da0d campaign_spec.json}"
CAMPAIGN_ROOT="${K2_SEGMENT_ROOT:?fresh segmented continuation root}"
COMMIT="${K2_SEGMENT_COMMIT:?full pushed executor commit}"
[[ "${1:-}" == '' || "${1:-}" == --execute ]] || { echo 'Only --execute is supported' >&2; exit 2; }
[[ "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" == "${COMMIT}" ]] || { echo 'Wrong executor commit' >&2; exit 2; }
export JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CLI="${PROJECT_DIR}/scripts/jetclass2_k2_segmented.py"
if [[ ! -e "${CAMPAIGN_ROOT}" ]]; then
  python -s "${CLI}" create --source-spec "${SOURCE_SPEC}" --campaign-root "${CAMPAIGN_ROOT}" --source-commit "${COMMIT}"
fi
SPEC="${CAMPAIGN_ROOT}/campaign_spec.json"
python -s - "${SPEC}" "${PROJECT_DIR}" "${COMMIT}" "${SOURCE_SPEC}" <<'PY'
import json
from pathlib import Path
import sys
spec = json.loads(Path(sys.argv[1]).read_text())
if (Path(spec['project_dir']).resolve() != Path(sys.argv[2]).resolve()
        or spec['source_commit'] != sys.argv[3]
        or Path(spec['source_spec']['path']).resolve() != Path(sys.argv[4]).resolve()):
    raise SystemExit('Existing continuation does not match the requested source/executor')
PY
python -s "${CLI}" retire --spec "${SPEC}"
python -s "${CLI}" submit --spec "${SPEC}" --stage full
if [[ "${1:-}" == --execute ]]; then
  [[ "${K2_DEBUG_POLICY_CONFIRMED:-}" == yes ]] || { echo 'Confirm RC permits segmented production on debug first.' >&2; exit 2; }
  python -s "${CLI}" retire --spec "${SPEC}" --execute --authorization-phrase 'CANCEL ONLY REGISTERED PENDING K2 REMAINDER'
  python -s "${CLI}" submit --spec "${SPEC}" --stage gate --execute --confirm-debug-policy --authorization-phrase 'AUTHORIZE K2 SEGMENTED DEBUG CONTINUATION EXACT SPEC'
fi

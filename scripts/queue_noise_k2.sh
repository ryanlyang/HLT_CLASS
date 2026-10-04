#!/usr/bin/env bash
# Run on OSCAR. Dry by default; never cancels or modifies another campaign.
set -euo pipefail
export PROJECT_DIR="${PROJECT_DIR:?absolute clean pushed checkout}"
: "${NOISE_K2_COMMIT:?full pushed commit}"
: "${CAMPAIGN_ROOT:?fresh separate OSCAR campaign root}"
: "${K2_FORMULA_SPEC:?copied original K2 campaign JSON}"
: "${K2_FORMULA_SHA256:?raw sha256 recorded on source host}"
MODE="${1:---dry-run}"
[[ "${MODE}" == --dry-run || "${MODE}" == --execute ]] || { echo "Use --dry-run or --execute" >&2; exit 2; }
[[ "$#" -le 1 ]] || { echo "Unexpected arguments" >&2; exit 2; }
[[ "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" == "${NOISE_K2_COMMIT}" ]]
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
SPEC="${CAMPAIGN_ROOT}/campaign_spec.json"
if [[ ! -e "${CAMPAIGN_ROOT}" ]]; then
    python -s "${PROJECT_DIR}/scripts/noise_k2.py" create \
        --formula-spec "${K2_FORMULA_SPEC}" --formula-sha256 "${K2_FORMULA_SHA256}" \
        --campaign-root "${CAMPAIGN_ROOT}" --source-commit "${NOISE_K2_COMMIT}"
fi
test -f "${SPEC}"
# Refuse an existing campaign from another checkout/formula even if self-valid.
python -s - "${SPEC}" <<'PY'
import json, os, sys
from pathlib import Path
s = json.load(open(sys.argv[1]))
assert s['source_commit'] == os.environ['NOISE_K2_COMMIT']
assert Path(s['project_dir']).resolve() == Path(os.environ['PROJECT_DIR']).resolve()
assert s['formula']['donor_file_sha256'] == os.environ['K2_FORMULA_SHA256']
PY
python -s "${PROJECT_DIR}/scripts/noise_k2.py" submit --spec "${SPEC}" --stage all
python -s "${PROJECT_DIR}/scripts/noise_k2.py" submit --spec "${SPEC}" --stage gate
python -s "${PROJECT_DIR}/scripts/noise_k2.py" submit --spec "${SPEC}" --stage science
if [[ "${MODE}" == --execute ]]; then
    python -s "${PROJECT_DIR}/scripts/noise_k2.py" submit --spec "${SPEC}" --stage gate \
        --execute --auto-science --authorization-phrase "AUTHORIZE OSCAR NOISE V3 K2 100K 50K EXACT SPEC"
else
    echo "Dry run only. --execute authorizes fresh preparation/GPU gates and automatic science after success."
fi

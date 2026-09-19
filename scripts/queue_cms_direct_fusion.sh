#!/bin/bash
# Parallel direct-fusion study. Never cancel or modify another campaign.
# Run with bash, not source: a failed check must not close the login shell.
set -euo pipefail
MODE="${1:?use prepare DENSE_SPEC NEW_ROOT or submit DIRECT_SPEC or results DIRECT_SPEC}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/cms_salience_learned.py"
AUTH="AUTHORIZE CMS DIRECT FUSION 500K EXACT SPEC"

case "${MODE}" in
  prepare)
    DENSE_SPEC="${2:?original accepted dense debug spec required}"
    DIRECT_ROOT="${3:?fresh isolated direct-fusion root required}"
    COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
    echo "Creating isolated study and validating source evidence (hash checks can take several minutes)."
    python -s "${CLI}" create-direct-fusion --source-spec "${DENSE_SPEC}" \
      --campaign-root "${DIRECT_ROOT}" --source-commit "${COMMIT}"
    DIRECT_SPEC="${DIRECT_ROOT}/campaign_spec.json"
    # Hash/receipt verification only, no GPU fit or raw-data preprocessing here.
    python -s "${CLI}" run --spec "${DIRECT_SPEC}" --task foundation
    python -s "${CLI}" run --spec "${DIRECT_SPEC}" --task preflight
    python -s "${CLI}" gate --spec "${DIRECT_SPEC}"
    python -s "${CLI}" submit --spec "${DIRECT_SPEC}" --stage science
    echo "Direct-fusion audit/dry run complete. No jobs submitted or cancelled."
    echo "Submit: bash ${PROJECT_DIR}/scripts/queue_cms_direct_fusion.sh submit ${DIRECT_SPEC}"
    ;;
  submit|results)
    DIRECT_SPEC="${2:?direct-fusion spec required}"
    # Refuse a mistaken coarse/dense path even for read-only results.
    python -s - "${DIRECT_SPEC}" "${PROJECT_DIR}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_salience_learned.campaign import validate_campaign
spec = load_json(sys.argv[1])
validate_campaign(spec, check_source=True)
if (spec.get("ladder") != "direct_fusion" or spec["schema_version"] != 6
    or spec["site"]["partition"] != "debug"
    or Path(spec["project_dir"]).resolve() != Path(sys.argv[2]).resolve()):
    raise SystemExit("Use this pinned checkout's v6 direct-fusion debug spec")
PY
    if [ "${MODE}" = results ]; then
      exec python -s "${CLI}" results --spec "${DIRECT_SPEC}"
    fi
    python -s "${CLI}" gate --spec "${DIRECT_SPEC}"
    python -s "${CLI}" submit --spec "${DIRECT_SPEC}" --stage science
    python -s "${CLI}" submit --spec "${DIRECT_SPEC}" --stage science \
      --execute --authorization-phrase "${AUTH}"
    echo "Direct-fusion science submitted on debug (cmsdf_). Coarse and source jobs were not changed."
    ;;
  *) echo "Use prepare, submit or results." >&2; exit 2 ;;
esac

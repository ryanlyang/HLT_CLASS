#!/bin/bash
# Isolated paired-teacher chain. Run with bash, never source this helper.
set -euo pipefail
MODE="${1:?use prepare COARSE_SPEC NEW_ROOT or submit CHAIN_SPEC or results CHAIN_SPEC}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/cms_salience_learned.py"

case "${MODE}" in
  prepare)
    COARSE_SPEC="${2:?completed first-acquisition coarse-v5 spec required}"
    CHAIN_ROOT="${3:?fresh isolated fusion-chain root required}"
    COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
    echo "Checking the installed Weaver same-view fusion route on CPU (no training campaign or GPU job)."
    python -s -c 'import weaver'
    python -s -m pytest -q -p no:cacheprovider \
      "${PROJECT_DIR}/tests/test_cms_fusion_chain.py::test_installed_weaver_same_view_chain_native_contract"
    echo "Verifying read-only preparation, references, acquisition and GPU evidence; hashes can take several minutes."
    python -s "${CLI}" create-fusion-chain --source-spec "${COARSE_SPEC}" \
      --campaign-root "${CHAIN_ROOT}" --source-commit "${COMMIT}"
    CHAIN_SPEC="${CHAIN_ROOT}/campaign_spec.json"
    python -s "${CLI}" run --spec "${CHAIN_SPEC}" --task foundation
    python -s "${CLI}" run --spec "${CHAIN_SPEC}" --task preflight
    python -s "${CLI}" gate --spec "${CHAIN_SPEC}"
    python -s "${CLI}" submit --spec "${CHAIN_SPEC}" --stage science
    echo "Seven fresh fits; both terminal routes; CPU imports verified; full dry plan ready. No jobs submitted."
    echo "Next: bash ${PROJECT_DIR}/scripts/queue_cms_fusion_chain.sh submit ${CHAIN_SPEC}"
    ;;
  submit|results|status)
    CHAIN_SPEC="${2:?canonical fusion-chain spec required}"
    python -s - "${CHAIN_SPEC}" "${PROJECT_DIR}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.cms_salience_learned.campaign import validate_campaign
spec = load_json(sys.argv[1])
validate_campaign(spec, check_source=True)
if (spec.get("ladder") != "fusion_chain" or spec["schema_version"] != 7
    or spec["site"]["partition"] != "debug"
    or Path(spec["project_dir"]).resolve() != Path(sys.argv[2]).resolve()
    or Path(sys.argv[1]).resolve() != Path(spec["campaign_root"]) / "campaign_spec.json"):
    raise SystemExit("Use this pinned checkout's canonical v7 fusion-chain debug spec")
PY
    if [ "${MODE}" != submit ]; then
      exec python -s "${CLI}" "${MODE}" --spec "${CHAIN_SPEC}"
    fi
    python -s "${CLI}" gate --spec "${CHAIN_SPEC}"
    python -s "${CLI}" submit --spec "${CHAIN_SPEC}" --stage science
    python -s "${CLI}" submit --spec "${CHAIN_SPEC}" --stage science \
      --execute --authorization-phrase "AUTHORIZE CMS FUSION CHAIN 500K EXACT SPEC"
    echo "Fusion-chain science submitted on debug (cmsfc_). Existing campaigns were not modified."
    ;;
  *) echo "Use prepare, submit, results or status." >&2; exit 2 ;;
esac

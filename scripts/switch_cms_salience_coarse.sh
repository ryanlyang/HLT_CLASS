#!/bin/bash
# Explicit, staged dense-to-coarse replacement. Run with bash, never source.
set -euo pipefail
MODE="${1:?use prepare/reuse-and-switch DENSE_SPEC NEW_ROOT or preview/finish COARSE_SPEC}"
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate atlas_kd_sporc
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/cms_salience_learned.py"
AUTH="AUTHORIZE CMS SALIENCE LEARNED COARSE 500K EXACT SPEC"

case "${MODE}" in
  prepare|reuse-and-switch)
    DENSE_SPEC="${2:?explicit dense source spec required}"
    COARSE_ROOT="${3:?fresh coarse root required}"
    COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
    REUSE_ARGS=()
    if [ "${MODE}" = reuse-and-switch ]; then
      REUSE_ARGS=(--reuse-dense-preflight)
    else
      CMS_TEST_TMP="$(mktemp -d /tmp/cms-coarse-tests.XXXXXXXX)"
      python -s -m pytest -q -p no:cacheprovider --basetemp="${CMS_TEST_TMP}/pytest" \
        "${PROJECT_DIR}/tests/test_cms_salience_learned.py" \
        "${PROJECT_DIR}/tests/test_cms_salience_coarse.py" \
        "${PROJECT_DIR}/tests/test_cms_fusion_temporary_memory.py"
    fi
    python -s "${CLI}" create-coarse --source-spec "${DENSE_SPEC}" \
      --campaign-root "${COARSE_ROOT}" --source-commit "${COMMIT}" "${REUSE_ARGS[@]}"
    COARSE_SPEC="${COARSE_ROOT}/campaign_spec.json"
    # Hash verification only: no raw-data preprocessing or GPU fit on login.
    python -s "${CLI}" run --spec "${COARSE_SPEC}" --task foundation
    if [ "${MODE}" = reuse-and-switch ]; then
      python -s "${CLI}" run --spec "${COARSE_SPEC}" --task preflight
      echo "Compatible accepted dense GPU evidence imported; no new GPU preflight submitted."
      exec bash "${PROJECT_DIR}/scripts/switch_cms_salience_coarse.sh" finish "${COARSE_SPEC}"
    fi
    python -s "${CLI}" submit --spec "${COARSE_SPEC}" --stage gate \
      --execute --authorization-phrase "${AUTH}"
    echo "Fresh coarse preflight submitted. No existing job was cancelled."
    echo "After it passes: bash ${PROJECT_DIR}/scripts/switch_cms_salience_coarse.sh finish ${COARSE_SPEC}"
    ;;
  preview)
    python -s "${CLI}" retire-dense --spec "${2:?coarse spec required}"
    ;;
  finish)
    COARSE_SPEC="${2:?coarse spec required}"
    # Any missing/failed gate stops this script before any cancellation.
    python -s "${CLI}" gate --spec "${COARSE_SPEC}"
    python -s "${CLI}" submit --spec "${COARSE_SPEC}" --stage science
    python -s "${CLI}" retire-dense --spec "${COARSE_SPEC}"
    python -s "${CLI}" retire-dense --spec "${COARSE_SPEC}" --execute \
      --authorization-phrase "CANCEL CMS DENSE LADDER KEEP SHARED"
    python -s "${CLI}" submit --spec "${COARSE_SPEC}" --stage science \
      --execute --authorization-phrase "${AUTH}"
    echo "Coarse science submitted. Shared dense reference/control jobs and all files were preserved."
    ;;
  *)
    echo "Unknown mode. Use prepare, reuse-and-switch, preview, or finish." >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
# All tasks on Oscar; the GPU tasks request one L40S. Gate and science remain separately dry-reviewed.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve the study and journals, and paste the error." >&2' ERR
if [[ $# != 4 && $# != 6 ]]; then
    echo 'Usage: queue_jetclass2_context_ladder.sh COMMIT COPIED_CONTAINER STUDY_ROOT gate|science [--execute PLAN_HASH]' >&2
    exit 2
fi
COMMIT="$1"
CONTAINER="$2"
DATASET="${CONTAINER}/jetclass2_literature_context_v1_2250k_4f473c3e_r1"
PROVENANCE="${CONTAINER}/oscar_provenance_v1"
OFFLINE=/oscar/home/rlyang/datasets/literature_noise_v3_2250k_ebd5bc1a_r1/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2
ROOT="$3"
MODE="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MODE}" == gate || "${MODE}" == science ]]
if [[ $# == 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_DIR JC2_SITE=oscar_l40s
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${PROJECT_DIR}" merge-base --is-ancestor "${COMMIT}" origin/main
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CLI="${PROJECT_DIR}/scripts/jetclass2_context_ladder.py"
if [[ "${MODE}" == gate ]]; then
    SPEC="${ROOT}/gate/gate_spec.json"
    PHRASE='AUTHORIZE JETCLASS2 CONTEXT 100K OSCAR GATE'
    if [[ ! -e "${ROOT}/gate" ]]; then
        [[ $# == 4 ]] || { echo 'Create and review the dry gate first.' >&2; exit 2; }
        python -u -s "${CLI}" create-gate --dataset-root "${DATASET}" \
            --offline-root "${OFFLINE}" --provenance-root "${PROVENANCE}" \
            --gate-root "${ROOT}/gate" --project-dir "${PROJECT_DIR}" --source-commit "${COMMIT}"
    fi
else
    SPEC="${ROOT}/science/campaign_spec.json"
    PHRASE='AUTHORIZE JETCLASS2 CONTEXT 100K OSCAR DIRECT COARSE SCIENCE'
    if [[ ! -e "${ROOT}/science" ]]; then
        [[ $# == 4 ]] || { echo 'Create and review the dry science plan first.' >&2; exit 2; }
        python -u -s "${CLI}" create-campaign --gate-root "${ROOT}/gate" --campaign-root "${ROOT}/science"
    fi
fi
python -s - "${ROOT}/gate/gate_spec.json" "${PROJECT_DIR}" "${COMMIT}" "${DATASET}" "${ROOT}" "${OFFLINE}" "${PROVENANCE}" <<'PY'
import json, sys
from pathlib import Path
path, project, commit, dataset, root, offline, provenance = sys.argv[1:]
gate = json.loads(Path(path).read_text())
if (gate['source_commit'] != commit or Path(gate['project_dir']).resolve() != Path(project).resolve()
        or Path(gate['gate_root']).resolve() != Path(root).resolve()/'gate'
        or Path(gate['request']['study_root']).resolve() != Path(dataset).resolve()
        or Path(gate['request']['offline_root']).resolve() != Path(offline).resolve()
        or Path(gate['request']['provenance_root']).resolve() != Path(provenance).resolve()):
    raise SystemExit('STOP: saved gate/source/dataset differs; nothing changed.')
PY
if [[ $# == 6 ]]; then
    python -u -s "${CLI}" submit --spec "${SPEC}" --mode "${MODE}" \
        --plan-hash "$6" --authorization-phrase "${PHRASE}"
else
    python -u -s "${CLI}" plan --spec "${SPEC}" --mode "${MODE}"
    echo "Dry review only; no jobs submitted. Mode: ${MODE}."
fi

#!/usr/bin/env bash
# All tasks on SPORC debug. Gate and science remain separately dry-reviewed.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve the study and journals, and paste the error." >&2' ERR
if [[ $# != 4 && $# != 6 ]]; then
    echo 'Usage: queue_jetclass2_literature_proxy_ladder.sh COMMIT DATASET_ROOT STUDY_ROOT gate|science [--execute PLAN_HASH]' >&2
    exit 2
fi
COMMIT="$1"
DATASET="$2"
ROOT="$3"
MODE="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MODE}" == gate || "${MODE}" == science ]]
if [[ $# == 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export PROJECT_DIR JC2_SITE=sporc_a100_debug
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
CLI="${PROJECT_DIR}/scripts/jetclass2_literature_proxy_ladder.py"
if [[ "${MODE}" == gate ]]; then
    SPEC="${ROOT}/gate/gate_spec.json"
    PHRASE='AUTHORIZE JETCLASS2 LITERATURE V3 200K SPORC DEBUG GATE'
    if [[ ! -e "${ROOT}/gate" ]]; then
        [[ $# == 4 ]] || { echo 'Create and review the dry gate first.' >&2; exit 2; }
        python -u -s "${CLI}" create-gate --dataset-root "${DATASET}" \
            --gate-root "${ROOT}/gate" --project-dir "${PROJECT_DIR}" --source-commit "${COMMIT}"
    fi
else
    SPEC="${ROOT}/science/campaign_spec.json"
    PHRASE='AUTHORIZE JETCLASS2 LITERATURE V3 200K DIRECT COARSE SCIENCE'
    if [[ ! -e "${ROOT}/science" ]]; then
        [[ $# == 4 ]] || { echo 'Create and review the dry science plan first.' >&2; exit 2; }
        python -u -s "${CLI}" create-campaign --gate-root "${ROOT}/gate" --campaign-root "${ROOT}/science"
    fi
fi
python -s - "${ROOT}/gate/gate_spec.json" "${PROJECT_DIR}" "${COMMIT}" "${DATASET}" "${ROOT}" <<'PY'
import json, sys
from pathlib import Path
path, project, commit, dataset, root = sys.argv[1:]
gate = json.loads(Path(path).read_text())
if (gate['source_commit'] != commit or Path(gate['project_dir']).resolve() != Path(project).resolve()
        or Path(gate['gate_root']).resolve() != Path(root).resolve()/'gate'
        or Path(gate['request']['study_root']).resolve() != Path(dataset).resolve()):
    raise SystemExit('STOP: saved gate/source/dataset differs; nothing changed.')
PY
if [[ $# == 6 ]]; then
    python -u -s "${CLI}" submit --spec "${SPEC}" --mode "${MODE}" \
        --plan-hash "$6" --authorization-phrase "${PHRASE}"
else
    python -u -s "${CLI}" plan --spec "${SPEC}" --mode "${MODE}"
    echo "Dry review only; no jobs submitted. Mode: ${MODE}."
fi

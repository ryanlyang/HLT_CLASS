#!/usr/bin/env bash
# Run under nohup; dry by default. Only the reviewed amendment retires jobs.
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; inspect the log before retrying." >&2' ERR
if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo 'Usage: COMMIT ORIGINAL_CONFIRM_SPEC NEW_ROOT debug|tier3 [--execute REVIEWED_PLAN_HASH]' >&2
    exit 2
fi
COMMIT="$1"
SUBJECT="$2"
ROOT="$3"
PARTITION="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${PARTITION}" == debug || "${PARTITION}" == tier3 ]]
if [[ $# -eq 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
fi
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CLI="${PROJECT_DIR}/scripts/cms2jc2_reduced_confirmation.py"
SPEC="${ROOT}/stages/frozen_reduced_r1/stage_spec.json"
step() {
    local description="$1"
    shift
    echo "${description}"
    "$@" &
    local pid=$!
    while kill -0 "${pid}" 2>/dev/null; do
        echo "${description}: working (PID ${pid})"
        sleep 15
    done
    wait "${pid}"
}
if [[ ! -e "${ROOT}" ]]; then
    # An execute call must operate on a previously reviewed, existing plan.
    [[ $# -eq 4 ]]
    step 'Freeze exact 36-shard amendment' python -s "${CLI}" create \
        --subject-spec "${SUBJECT}" --project-dir "${PROJECT_DIR}" --source-commit "${COMMIT}" \
        --root "${ROOT}" --partition "${PARTITION}"
else
    test -f "${SPEC}"
fi
python -s - "${SPEC}" "${SUBJECT}" "${ROOT}" "${PROJECT_DIR}" "${COMMIT}" "${PARTITION}" <<'PY'
import json, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
study = json.loads(Path(s['study']['path']).read_text())
if (Path(s['subject_spec']['path']).resolve() != Path(sys.argv[2]).resolve()
        or Path(s['root']).resolve() != Path(sys.argv[3]).resolve()
        or Path(study['project_dir']).resolve() != Path(sys.argv[4]).resolve()
        or study['source']['commit'] != sys.argv[5] or study['site']['partition'] != sys.argv[6]):
    raise SystemExit('Existing amendment differs; stop without modifying it.')
PY
if [[ $# -eq 4 ]]; then
    step 'Dry retirement review' python -s "${CLI}" retire --spec "${SPEC}"
    step 'Dry report submission review' python -s "${CLI}" submit --spec "${SPEC}"
    echo 'No jobs changed. Repeat with --execute and the reviewed plan hash.'
else
    step 'Retire exact excluded jobs' python -s "${CLI}" retire --spec "${SPEC}" \
        --execute --plan-hash "$6" \
        --authorization-phrase 'AUTHORIZE CMS2JC2 STOP REMAINING 21 AND ORIGINAL REPORT'
    step 'Submit single reduced report' python -s "${CLI}" submit --spec "${SPEC}" \
        --execute --plan-hash "$6" --authorization-phrase 'AUTHORIZE CMS2JC2 FROZEN_REDUCED EXACT PLAN'
fi
echo "Study: ${ROOT}"
echo 'QUEUE_HELPER_EXIT=0'

#!/usr/bin/env bash
# One replacement preflight, then separately reviewed science; old artifacts read-only.
set -euo pipefail
if [[ $# -ne 4 && $# -ne 6 ]]; then
    echo 'Usage: helper COMMIT ORIGINAL_GATE ROOT gate|science [--execute PLAN_HASH]' >&2
    exit 2
fi
COMMIT="$1"
ORIGINAL="$2"
ROOT="$3"
MODE="$4"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MODE}" == gate || "${MODE}" == science ]]
if [[ $# -eq 6 ]]; then
    [[ "$5" == --execute && "$6" =~ ^[0-9a-f]{64}$ ]]
fi
PROJECT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
if [[ ! -f "${ROOT}/gate_spec.json" ]]; then
    [[ $# -eq 4 && "${MODE}" == gate ]]
    test ! -e "${ROOT}"
    python -u -s "${PROJECT}/scripts/luka_fullsim_regular.py" create-recovery \
        --root "${ROOT}" --original-gate "${ORIGINAL}" --expected-commit "${COMMIT}"
fi
python -s -c 'import json,sys; from pathlib import Path; s=json.load(open(sys.argv[1])); assert s["contract"] == "LUKA_FULLSIM_REGULAR_RECOVERY_GATE/v1"; assert Path(s["reuse"]["original_gate"]["path"]).resolve() == Path(sys.argv[2]).resolve()' \
    "${ROOT}/gate_spec.json" "${ORIGINAL}"
ARGS=("${COMMIT}" "${ROOT}" "${MODE}")
if [[ $# -eq 6 ]]; then ARGS+=(--execute "$6"); fi
exec bash "${PROJECT}/scripts/queue_luka_fullsim_regular.sh" "${ARGS[@]}"

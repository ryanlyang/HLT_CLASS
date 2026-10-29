#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve outputs and journals. Do not blindly retry." >&2' ERR
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_meta_guard_${COMMIT:0:8}_debug_r1"
PARENT="${MAIN}/checkpoints/luka_meta_embedding_7a6dc268_debug_r1/campaign_spec.json"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" fetch origin main
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
test ! -e "${ROOT}"
test -s "${PARENT}"
if ! test -s "$(dirname -- "${PARENT}")/fixed/receipt.json"; then
    echo 'Original FIXED must finish first; its selected full checkpoint is published at completion.' >&2
    exit 1
fi
df -h /home/ryreu
read -r -p 'Personally available GiB reserved (at least 4; actual registered budget is 4 GiB): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT}/scripts/luka_meta_guard.py" create --root "${ROOT}" \
    --parent "${PARENT}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
SPEC="${ROOT}/campaign_spec.json"
python -u -B -s "${PROJECT}/scripts/luka_meta_guard.py" submit --spec "${SPEC}"
read -r -p 'Paste the printed 64-character PLAN_HASH value: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue shared inline checks and BOTH guarded debug fits? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${PROJECT}/scripts/luka_meta_guard.py" submit --spec "${SPEC}" \
    --execute --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_META_GUARD_DEBUG_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nStudy: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
printf 'Wait for all three exact IDs in the completed submission ledger; then Slurm owns the jobs.\n'
printf 'No target rebuild or predictor calibration. Borrowed parent artifacts must remain intact.\n'

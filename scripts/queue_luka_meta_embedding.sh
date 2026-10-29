#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Stopped; preserve outputs and journals. Do not blindly retry." >&2' ERR
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
ROOT="${MAIN}/checkpoints/luka_meta_embedding_${COMMIT:0:8}_debug_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
REFERENCE="${MAIN}/checkpoints/luka_embedding_sweep_007885f7_tier3_r1/science_debug/campaign_spec.json"
test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
git -C "${PROJECT}" fetch origin main
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
test ! -e "${ROOT}"
test -s "${PARENT}"
test -s "${REFERENCE}"
test -s "$(dirname -- "${PARENT}")/train_OFFLINE/selected.pt"
test -s "$(dirname -- "${REFERENCE}")/w5/fit/selected.pt"
df -h /home/ryreu
read -r -p 'Personally available persistent GiB reserved (at least 4; registered budget is 4 GiB): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT}/scripts/luka_meta_embedding.py" create --root "${ROOT}" \
    --parent "${PARENT}" --reference "${REFERENCE}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
SPEC="${ROOT}/campaign_spec.json"
python -u -B -s "${PROJECT}/scripts/luka_meta_embedding.py" submit --spec "${SPEC}"
read -r -p 'Paste the printed 64-character PLAN_HASH value: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue shared setup and BOTH fixed/meta debug continuations? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup python -u -B -s "${PROJECT}/scripts/luka_meta_embedding.py" submit --spec "${SPEC}" \
    --execute --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_META_EMBEDDING_DEBUG_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nSubmission PID: %s\nStudy: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
printf 'Wait for the three exact job IDs in the completed ledger. Then Slurm owns all jobs and logout is safe.\n'
printf 'No manual second submission is needed; both fits depend on verified setup success.\n'

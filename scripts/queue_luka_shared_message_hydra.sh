#!/usr/bin/env bash
# Called from a clean, pushed detached worktree on Brown CS Hydra.
set -euo pipefail
trap 'echo "Stopped; preserve specs, outputs and journals. Do not blindly retry." >&2' ERR
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
BASE="$(cd "${HOME}" && pwd -P)/hlt_u100_pairs"
COMMIT="$(git -C "${PROJECT}" rev-parse HEAD)"
PARENT="${BASE}/runs/cross_message_199928da_3693bcda_a6000_r1/campaign_spec.json"
ENV_DIR="${BASE}/environment_pair_hydra_v1/venv"
ROOT="${BASE}/runs/shared_message_${COMMIT:0:8}_a6000_r1"
[[ -f "${PARENT}" && -x "${ENV_DIR}/bin/python" && ! -e "${ROOT}" ]]
git -C "${PROJECT}" fetch origin main
git -C "${PROJECT}" merge-base --is-ancestor "${COMMIT}" origin/main
[[ -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)" ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src" VIRTUAL_ENV="${ENV_DIR}"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
df -h "${BASE}"
read -r -p 'Personally available GiB reserved for new outputs (at least 4): ' QUOTA </dev/tty
CLI="${PROJECT}/scripts/luka_shared_message_hydra.py"
"${ENV_DIR}/bin/python" -B -s "${CLI}" create --root "${ROOT}" --parent-spec "${PARENT}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}"
SPEC="${ROOT}/campaign_spec.json"
"${ENV_DIR}/bin/python" -B -s "${CLI}" review --spec "${SPEC}"
PLAN_HASH="$("${ENV_DIR}/bin/python" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission_bootstrap/command_plan.json")"
printf '\nReview hash: %s\nTwelve independent A6000 fits after warmed setup; Slurm CPU release survives logout.\n' "${PLAN_HASH}"
read -r -p 'Authorize setup, durable release, twelve measured fits and report? Type YES: ' ANSWER </dev/tty
[[ "${ANSWER}" == YES ]]
"${ENV_DIR}/bin/python" -B -s "${CLI}" submit --spec "${SPEC}" --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_SHARED_MESSAGE_HYDRA_V1
printf '\nCampaign: %s\nSafe to log off after both bootstrap job IDs are printed.\n' "${ROOT}"

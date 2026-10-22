#!/usr/bin/env bash
# Run from the new clean, detached, pushed source worktree on sporcsubmit.
set -euo pipefail
trap 'echo "Stopped; preserve logs/journals and inspect. Do not blindly rerun." >&2' ERR
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COMMIT="$(git -C "${PROJECT_DIR}" rev-parse HEAD)"
MAIN=/home/ryreu/atlas/HLT_Classification
SITE_NAME="${SITE_NAME:-sporc_a100_debug}"
ORDER_PENDING="${ORDER_PENDING:-0}"
case "${SITE_NAME}" in
    sporc_a100_debug) PARTITION=debug ;;
    sporc_a100) PARTITION=tier3 ;;
    *) echo 'STOP: SITE_NAME must be sporc_a100_debug or sporc_a100.' >&2; exit 1 ;;
esac
[[ "${ORDER_PENDING}" == 0 || "${ORDER_PENDING}" == 1 ]]
if [[ "${ORDER_PENDING}" == 1 && "${PARTITION}" != debug ]]; then
    echo 'STOP: tier3 must leave the existing debug holds untouched.' >&2
    exit 1
fi
# Explicit replacement guard: no cancellation is performed by this helper.
if [[ -n "${REPLACES_PREFLIGHT_JOB:-}" ]]; then
    [[ "${PARTITION}" == tier3 && "${REPLACES_PREFLIGHT_JOB}" == 21944417 ]]
    OLD_ACCOUNTING="$(sacct -X -n -P -j "${REPLACES_PREFLIGHT_JOB}" \
        -o JobIDRaw,JobName%100,User%100,Account%100,Partition%100,State%40,ElapsedRaw)"
    [[ -n "${OLD_ACCOUNTING}" && "${OLD_ACCOUNTING}" != *$'\n'* ]]
    IFS='|' read -r OLD_ID OLD_NAME OLD_USER OLD_ACCOUNT OLD_PARTITION OLD_STATE OLD_ELAPSED EXTRA <<<"${OLD_ACCOUNTING}"
    [[ "${OLD_ID}" == 21944417 && "${OLD_NAME}" == lukad6e_preflight && "${OLD_USER}" == ryreu ]]
    [[ "${OLD_ACCOUNT}" == reu-aisocial && "${OLD_PARTITION}" == debug && "${OLD_ELAPSED}" == 0 && -z "${EXTRA}" ]]
    [[ "${OLD_STATE}" == CANCELLED || "${OLD_STATE}" =~ ^CANCELLED\ by\ [0-9]+$ ]]
    echo 'Verified pending-only cancellation of21944417; its old controller cannot advance to science.'
fi
# End replacement guard.
ROOT="${MAIN}/checkpoints/luka_d066_embedding_${COMMIT:0:8}_${PARTITION}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
TEACHER="${MAIN}/checkpoints/luka_attention_94f56025_r1/science/campaign_spec.json"
REFERENCE="${MAIN}/checkpoints/luka_embedding_sweep_007885f7_tier3_r1/science_debug/campaign_spec.json"
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -z "$(git -C "${PROJECT_DIR}" status --porcelain --untracked-files=all)"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
test ! -e "${ROOT}"
test -s "${TEACHER%/campaign_spec.json}/train_LOGIT_D066/selected.pt"
test -s "${REFERENCE%/campaign_spec.json}/w5/receipt.json"
df -h /home/ryreu
read -r -p 'Personally available persistent GiB reserved for this run (at least 2): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT_DIR}/scripts/luka_d066_embedding.py"
python -u -B -s "${CLI}" create --root "${ROOT}" --parent "${PARENT}" --teacher "${TEACHER}" \
    --reference "${REFERENCE}" --expected-commit "${COMMIT}" --available-quota-gib "${QUOTA}" --site "${SITE_NAME}"
ORDER_ARGS=()
if [[ "${ORDER_PENDING}" == 1 ]]; then ORDER_ARGS+=(--order-pending); fi
python -u -B -s "${CLI}" auto --spec "${ROOT}/gate_spec.json" "${ORDER_ARGS[@]}"
read -r -p 'Paste the printed 64-character FOLLOWUP_PLAN_HASH value: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p "Authorize one ${PARTITION} preflight and one fresh fit after verified success? Type YES: " CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
if [[ "${ORDER_PENDING}" == 1 ]]; then
    read -r -p 'Hold ONLY pending 21940433..21940437 until the new fit finishes? Type HOLD FIVE: ' CONFIRM </dev/tty
    [[ "${CONFIRM}" == 'HOLD FIVE' ]]
    ORDER_ARGS+=(--priority-authorization HOLD_ONLY_REVIEWED_PENDING_SWEEP_JOBS_V1)
else
    echo 'Existing jobs and manual holds will not be changed or released by this controller.'
fi
LOG="$(mktemp "${ROOT}/auto-followup.XXXXXX.log")"
nohup python -u -B -s "${CLI}" auto --spec "${ROOT}/gate_spec.json" "${ORDER_ARGS[@]}" \
    --execute --plan-hash "${PLAN_HASH}" --authorization AUTHORIZE_LUKA_D066_EMBEDDING_FOLLOWUP_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf '\nController PID: %s\nStudy: %s\nLog: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}" "${LOG}"
printf 'Confirm ARMED in the log before leaving. No manual science submission is needed.\n'

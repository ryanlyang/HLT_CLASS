# HCWDL-MHPE Refined Continuation Runbook

This creates and submits the additive 300k/60-pass continuation from the
completed C25P75 source campaign. It performs no final-test access.

```bash
set -euo pipefail
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export SOURCE_SPEC="${MAIN_REPO}/checkpoints/hcwdl_mhpe_c25p75_300k60_7810cc28_r1/campaign_spec.json"
cd "${MAIN_REPO}"
git fetch origin main
export REFINED_COMMIT="$(git rev-parse origin/main)"
export REFINED_SHORT="${REFINED_COMMIT:0:8}"
export REFINED_WORKTREE="/home/ryreu/atlas/HLT_Classification_mhpe_refined_${REFINED_SHORT}"
export REFINED_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_refined_${REFINED_SHORT}_r1"

test -f "${SOURCE_SPEC}"
test ! -e "${REFINED_ROOT}"
if [ -e "${REFINED_WORKTREE}" ]; then
  test "$(git -C "${REFINED_WORKTREE}" rev-parse HEAD)" = "${REFINED_COMMIT}"
  test -z "$(git -C "${REFINED_WORKTREE}" status --porcelain)"
else
  git worktree add --detach "${REFINED_WORKTREE}" "${REFINED_COMMIT}"
fi
export PYTHONPATH="${REFINED_WORKTREE}/src"

python -s "${REFINED_WORKTREE}/scripts/create_hcwdl_mhpe_refined_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${REFINED_ROOT}" \
  --project-dir "${REFINED_WORKTREE}" \
  --source-commit "${REFINED_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL MHPE REFINED CONTINUATION 300K60 EXACT SPEC"

python -s "${REFINED_WORKTREE}/scripts/submit_hcwdl_mhpe_refined_campaign.py" \
  --spec "${REFINED_ROOT}/campaign_spec.json" \
  --output "${REFINED_ROOT}/dry_run_submission_ledger.json"

python -m json.tool "${REFINED_ROOT}/command_plan.json"

python -s "${REFINED_WORKTREE}/scripts/submit_hcwdl_mhpe_refined_campaign.py" \
  --spec "${REFINED_ROOT}/campaign_spec.json" \
  --output "${REFINED_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL MHPE REFINED CONTINUATION 300K60 EXACT LEDGER"

python -m json.tool "${REFINED_ROOT}/submission_ledger.json"
squeue --me -o "%.18i %.60j %.2t %.10M %R" | grep -E 'JOBID|hcwmhper_'
```

Monitor with:

```bash
python -s "${REFINED_WORKTREE}/scripts/monitor_hcwdl_mhpe_refined.py" \
  --spec "${REFINED_ROOT}/campaign_spec.json" \
  --submission-ledger "${REFINED_ROOT}/submission_ledger.json" \
  --output "${REFINED_ROOT}/monitor.json"
```

Only exact campaign-bound job IDs from the ledger may be cancelled.

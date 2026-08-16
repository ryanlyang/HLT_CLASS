# HCWDL-MHPE dense anchor-50 300k/60-pass runbook

This launches the additive `C10P90_DENSE_ANCHOR50_300K60` profile. It reuses
a completed authenticated unified-balanced 300k foundation and does not
submit another smoke. Local implementation work never submits jobs.

## Build and dry-run

Use a clean detached Tigris worktree at the exact pushed commit, then run:

```bash
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export PROJECT_DIR=/home/ryreu/atlas/HLT_Classification_mhpe_dense_<short>
export FOUNDATION_LOCK=/absolute/completed/300k/foundation/locks/foundation.json
export CAMPAIGN_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_mhpe_dense_<short>_r1
export SOURCE_COMMIT=<full-40-character-commit>

python -s "${PROJECT_DIR}/scripts/create_hcwdl_mhpe_campaign.py" \
  --foundation-lock "${FOUNDATION_LOCK}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SOURCE_COMMIT}" \
  --recipe-profile C10P90_DENSE_ANCHOR50_300K60 \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL MHPE DENSE ANCHOR50 300K60 EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_mhpe_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/dry_run_submission_ledger.json"
```

The plan must contain exactly 38 jobs: 29 fits, six reducers, aggregate,
finalist lock, and completion. GPU jobs request 8 CPUs, 96G, six hours, and
one GH200.

## Live submission

After reviewing the immutable spec and dry-run ledger:

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_mhpe_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL MHPE DENSE ANCHOR50 300K60 EXACT LEDGER"
```

Use the existing MHPE exact-ID monitor and recovery CLIs. Never cancel by a
broad name match. Scientific metrics do not gate descendants; only execution,
data, numerical, or lineage failures fail closed.

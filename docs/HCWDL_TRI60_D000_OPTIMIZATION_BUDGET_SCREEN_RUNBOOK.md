# TRI60 D000 Optimization-Budget Screen Runbook

This is a standalone full-data, 17-fit screen.  It does not cancel, reprioritize,
or depend on the live TRI60/DX ladders.

## Local validation

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_hcwdl_tri60_d000_budget_screen.py `
  tests/test_hcwdl_tri60_d000_long180.py `
  tests/test_hcwdl_mhpe_tri60.py
```

## Commit and push

Stage only the files named in the implementation handoff; do not use
`git add -A` in a dirty worktree.  After pushing, record the exact 40-character
commit.

## Tigris creation, dry run, and live submission

```bash
set -euo pipefail

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export SOURCE_SPEC="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"

cd "${MAIN_REPO}"
git fetch origin main
export DOPT_COMMIT="$(git rev-parse origin/main)"
export DOPT_SHORT="${DOPT_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_d000_opt_${DOPT_SHORT}"
export DOPT_ROOT="${MAIN_REPO}/checkpoints/hcwdl_tri60_d000_opt_${DOPT_SHORT}_r1"

test -f "${SOURCE_SPEC}"
test ! -e "${DOPT_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${DOPT_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach "${PROJECT_DIR}" "${DOPT_COMMIT}"
fi

export PYTHONPATH="${PROJECT_DIR}/src"
bash -n "${PROJECT_DIR}/sbatch/run_hcwdl_tri60_d000_budget_screen_task.sh"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri60_d000_budget_screen_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${DOPT_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${DOPT_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI60 D000 OPTIMIZATION BUDGET SCREEN EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_d000_budget_screen_campaign.py" \
  --spec "${DOPT_ROOT}/campaign_spec.json" \
  --output "${DOPT_ROOT}/dry_run_submission_ledger.json"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_d000_budget_screen_campaign.py" \
  --spec "${DOPT_ROOT}/campaign_spec.json" \
  --output "${DOPT_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI60 D000 OPTIMIZATION BUDGET SCREEN EXACT LEDGER"

squeue --me -o "%.18i %.58j %.2t %.10M %R" | grep -E 'JOBID|hcwdopt' || true
```

The two CPU gates run first.  The 17 `hcwdopt_train_*` jobs then become eligible
together.  `aggregate` and `campaign_complete` run only after all fits complete.
No job in this command references a live Slurm job from another campaign.

## Print completed results

```bash
python -s "${PROJECT_DIR}/scripts/print_hcwdl_tri60_d000_budget_screen_results.py" \
  --campaign-root "${DOPT_ROOT}"
```

The printer ranks the imported original result and all 17 fits, shows selected
pass, accuracy, AUC, linear macro R50, and AUC/R50 deltas from the original
60-pass `D000<-D033E` fit.

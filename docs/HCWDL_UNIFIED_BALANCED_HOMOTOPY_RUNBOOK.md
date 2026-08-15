# HCWDL Unified-Balanced 300k Runbook

This runbook creates one shared foundation and, after its lock exists, submits
six independent arms. It does not run another smoke and it never submits jobs
unless `--execute` and the exact authorization phrase are both supplied.

## 1. Create a clean source-pinned worktree

On Tigris, after the completed implementation commit is pushed:

```bash
set -euo pipefail
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
git -C "${MAIN_REPO}" fetch origin main
export UB_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export UB_SHORT="${UB_COMMIT:0:8}"
export UB_WORKTREE="/home/ryreu/atlas/HLT_Classification_hcwdl_ub_${UB_SHORT}"
test ! -e "${UB_WORKTREE}"
git -C "${MAIN_REPO}" worktree add --detach "${UB_WORKTREE}" "${UB_COMMIT}"
test -z "$(git -C "${UB_WORKTREE}" status --porcelain)"

export PYTHONPATH="${UB_WORKTREE}/src"
export UB_FOUNDATION_ROOT="${MAIN_REPO}/checkpoints/hcwdl_ub_foundation_${UB_SHORT}_r1"
export UB_ARMS_ROOT="${MAIN_REPO}/checkpoints/hcwdl_ub_arms_${UB_SHORT}_r1"
export UB_LEDGER_ROOT="${MAIN_REPO}/checkpoints/hcwdl_ub_ledgers_${UB_SHORT}_r1"
```

Set these three authenticated inputs explicitly. The U/J parent needs its
completed fixed-preprocessing prefix through `locks/graph_recipe_lock.json`;
its scientifically unrelated training descendants and aggregate do not need
to be complete. Do not select inputs merely by newest modification time:

```bash
export PARENT_UJ_ROOT=/absolute/path/to/completed/300k/hcwdl_uj_campaign
export PARENT_UJ_SPEC="${PARENT_UJ_ROOT}/campaign_spec.json"
export FACTORIAL_ROOT=/absolute/path/to/completed/300k/architecture_input_factorial
export FACTORIAL_SPEC="${FACTORIAL_ROOT}/campaign_spec.json"
export PRIOR_UJ_SMOKE_COMPLETE=/absolute/path/to/completed/v1/smoke/reports/campaign_complete.json
test -f "${PARENT_UJ_SPEC}"
test -f "${PARENT_UJ_ROOT}/locks/coupling_lock.json"
test -f "${PARENT_UJ_ROOT}/locks/endpoint_equality_lock.json"
test -f "${PARENT_UJ_ROOT}/locks/graph_recipe_lock.json"
test -f "${FACTORIAL_ROOT}/reports/campaign_complete.json"
test -f "${FACTORIAL_SPEC}"
test -f "${PRIOR_UJ_SMOKE_COMPLETE}"
```

## 2. Build the exact no-new-smoke waiver

This is an evidence lock, not a claim that HCWDL-UB itself passed a smoke.

```bash
export UB_WAIVER="${MAIN_REPO}/checkpoints/hcwdl_ub_waiver_${UB_SHORT}.json"
python -s "${UB_WORKTREE}/scripts/build_hcwdl_unified_balanced_waiver.py" \
  --source-commit "${UB_COMMIT}" \
  --parent-preparation-lock "${PARENT_UJ_ROOT}/locks/graph_recipe_lock.json" \
  --parent-homotopy-spec "${PARENT_UJ_SPEC}" \
  --prior-smoke-completion "${PRIOR_UJ_SMOKE_COMPLETE}" \
  --performance-guide "${UB_WORKTREE}/docs/HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md" \
  --readiness-evidence "${UB_WORKTREE}/docs/HANDOFF.md" \
  --project-dir "${UB_WORKTREE}" \
  --authorization-phrase "AUTHORIZE HCWDL UB 300K NO NEW SMOKE EXACT EVIDENCE" \
  --output "${UB_WAIVER}"
```

## 3. Create, dry-run, and submit the shared foundation

```bash
python -s "${UB_WORKTREE}/scripts/create_hcwdl_unified_balanced_foundation.py" \
  --parent-homotopy-spec "${PARENT_UJ_SPEC}" \
  --factorial-campaign-spec "${FACTORIAL_SPEC}" \
  --campaign-root "${UB_FOUNDATION_ROOT}" \
  --project-dir "${UB_WORKTREE}" \
  --source-commit "${UB_COMMIT}" \
  --operational-waiver "${UB_WAIVER}" \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL UB 300K FOUNDATION EXACT SPEC"

python -s "${UB_WORKTREE}/scripts/submit_hcwdl_unified_balanced.py" \
  --spec "${UB_FOUNDATION_ROOT}/foundation_spec.json" \
  --output "${UB_LEDGER_ROOT}/foundation_dry/submission_ledger.json"

python -s "${UB_WORKTREE}/scripts/submit_hcwdl_unified_balanced.py" \
  --spec "${UB_FOUNDATION_ROOT}/foundation_spec.json" \
  --output "${UB_LEDGER_ROOT}/foundation_live/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL UB 300K FOUNDATION EXACT IDS"
```

Training jobs request exactly 8 CPUs, 96 GiB RAM, six hours, and one GH200.
Sidecar construction is CPU-only and may use its separately locked CPU class.

Monitor without changing scheduler state:

```bash
python -s "${UB_WORKTREE}/scripts/monitor_hcwdl_unified_balanced.py" \
  --spec "${UB_FOUNDATION_ROOT}/foundation_spec.json" \
  --submission-ledger "${UB_LEDGER_ROOT}/foundation_live/submission_ledger.json" \
  --output "${UB_LEDGER_ROOT}/foundation_live/monitor.json"
```

Proceed only when this exact file exists and validates:

```bash
export UB_FOUNDATION_LOCK="${UB_FOUNDATION_ROOT}/locks/foundation.json"
test -f "${UB_FOUNDATION_LOCK}"
```

## 4. Create and dry-run all six arms

```bash
python -s "${UB_WORKTREE}/scripts/create_hcwdl_unified_balanced_arms.py" \
  --foundation-lock "${UB_FOUNDATION_LOCK}" \
  --arms-root "${UB_ARMS_ROOT}" \
  --project-dir "${UB_WORKTREE}" \
  --source-commit "${UB_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL UB SIX ARM 300K EXACT SPECS"

python -s "${UB_WORKTREE}/scripts/submit_hcwdl_unified_balanced_arms.py" \
  --recipe-sweep "${UB_ARMS_ROOT}/recipe_sweep.json" \
  --output-root "${UB_LEDGER_ROOT}/arms_dry"
```

The dry run authenticates six canonical specs and command plans. It does not
submit any job.

## 5. Submit the six independent arms

This is the only six-arm scheduler mutation command:

```bash
python -s "${UB_WORKTREE}/scripts/submit_hcwdl_unified_balanced_arms.py" \
  --recipe-sweep "${UB_ARMS_ROOT}/recipe_sweep.json" \
  --output-root "${UB_LEDGER_ROOT}/arms_live" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL UB SIX ARMS PARALLEL EXACT LEDGERS"
```

The wrapper preflights all six arms before submitting the first one. It writes
six distinct ledgers and introduces no cross-arm dependencies or array cap.
Submission calls are sequential, but all six DAG frontiers are immediately
eligible to run in parallel.

Show the complete queue names:

```bash
squeue --me -o "%.18i %.60j %.2t %.10M %R"
```

Monitor one arm by replacing `ARM` below:

```bash
export ARM=C10P75G15
python -s "${UB_WORKTREE}/scripts/monitor_hcwdl_unified_balanced.py" \
  --spec "${UB_ARMS_ROOT}/${ARM}/arm_spec.json" \
  --submission-ledger "${UB_LEDGER_ROOT}/arms_live/${ARM}/submission_ledger.json" \
  --output "${UB_LEDGER_ROOT}/arms_live/${ARM}/monitor.json"
```

## 6. Exact cancellation and recovery

Preview exact IDs before cancellation:

```bash
python -s "${UB_WORKTREE}/scripts/cancel_hcwdl_unified_balanced.py" \
  --submission-ledger /exact/scope/submission_ledger.json
```

Execution additionally requires:

```text
--execute --authorization-phrase "CANCEL HCWDL UB EXACT IDS"
```

After publishing a monitor, create either a source-pinned recovery using a new
execution-only commit, or a resource recovery using a JSON increase such as:

```text
--resource-overrides-json '{"gpu_training":{"walltime":"12:00:00"}}'
```

Source and resource changes cannot be combined. Recovery schedules only the
authenticated failed/downstream closure and preserves completed outputs.

## 7. Validation aggregation and sealed test

After all six completion artifacts exist, build the read-only aggregate and
finalist lock with `aggregate_hcwdl_unified_balanced_sweep.py` and
`lock_hcwdl_unified_balanced_finalists.py`. No final-test access occurs.

Final test requires a separate explicit execution lock with phrase:

```text
AUTHORIZE HCWDL UB SEALED FINAL TEST EXACT FINALISTS
```

and a separate submission phrase:

```text
SUBMIT HCWDL UB SEALED FINAL TEST EXACT LOCK
```

The sealed job evaluates only exact-HLT models. It is intentionally outside
the foundation and six-arm submissions.

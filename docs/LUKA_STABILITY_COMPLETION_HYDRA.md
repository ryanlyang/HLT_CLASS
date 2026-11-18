# Complete the outstanding stability results

This execution repair targets the original 42-fit stability campaign and its
existing two recovery roots. It preserves their immutable specs, source,
checkpoints, predictions, jobs and scientific meaning.

## What the fresh diagnostic established

Four more original jobs completed 30 epochs and exported their fit receipts,
then failed the final predictive storage admission check:

| Reuse after full validation | Original job |
| --- | --- |
| DEC2_s7 | 180553 |
| PROJ_s7 | 180548 |
| PROJ_s8 | 180554 |
| R512_s8 | 180558 |

These are not retrained and no old task receipts are fabricated. The new
qualification record binds their exact failed IDs, logs, startup miniatures,
claims and completed fit receipts. Original validators authenticate all 30
epochs, exported models, prediction identities and metrics before reuse.

The first storage recovery (`794b8f80`) remains in place. In the supplied
snapshot R512_s9 and R_L50_s9 were complete; DEC2_s8, DEC2_s9 and PROJ_s9 were
still running. This repair never submits or cancels those tasks. It includes
their results as their existing receipts become available.

The numerical-v2 SOFT setup (`183850`, source `d28c4033`) completed its genuine
four-GPU qualification, replay, four discarded TRAIN passes, evolved solver
stress and all three production miniatures. It saved `resource_diagnostic.json`
then failed because measured walltime exceeded the original 48-hour ceiling.
Its dependent release `183851` was cancelled and no SOFT science was submitted.

The recorded timings were about 11.04 and 12.60 seconds per soft update and
0.621 seconds per hard update. The original projection already accounts for
six soft passes and 24 hard passes. It estimates about 59.8 hours including
validation/startup, and approximately **120.15 hours requested per fit** after
the unchanged factor-of-two margin and 30-minute allowance. This is an estimate,
not a promised completion time. It excludes queue waiting. Three simultaneous
fits use twelve GTX1080Ti cards.

## Recovery execution

The new `luka_stability_completion_hydra` driver authenticates the completed
GPU qualification directly. It does not turn a failed Slurm job into a successful
one. Every original qualification, numerical tolerance, memory envelope, evolved
cost, timing and miniature check still applies. The sole resource-validation
override is an explicit seven-day upper bound for these three SOFT executions;
their actual requests are calculated from the saved measurements, not set to
seven days. Registration records this under
`measured_soft_walltime_2x_plus_30min_bounded_7days/v1`.

No new four-hour discarded setup is needed. After registration and a full Slurm
dry review, one command submits **SOFT_s7, SOFT_s8, SOFT_s9 and an automatic CPU
report**. Each fit requests four GTX1080Ti GPUs, 12 CPUs and 64000 MiB, and runs
its original discarded production miniature before 30 scientific epochs. The
CPU report has `afterany` dependencies on those three fits, so failures remain
visible rather than suppressing the report. It also reads the original and
existing storage recovery; any still-incomplete fits remain explicitly listed.

The original SOFT-v2 training code, solver, calibration, seeds, initialization,
batch size, HLT-only deployment, CE+U100-w5 objective, reconstruction schedule,
four readouts and checkpoint selection remain unchanged. The old SOFT spec
continues to identify the science; the new execution spec identifies where
replacement outputs live and which jobs produced them. There is no epoch resume
from selected weights. All three SOFT fits start fresh.

Available disk space is advisory, as already authorized for recovery. Actual
I/O failures, corruption, nonfinite values and lineage mismatches remain fatal.
The saved SOFT setup estimates about 0.93 GiB for three scientific outputs;
no checkpoints are copied merely for reporting.

## Contracts and reporting

New artifacts use `LUKA_STABILITY_COMPLETION_<KIND>/v1`: SPEC, QUALIFICATION,
PLAN, SUBMIT_CLAIM, RUN_CLAIM, TASK, RESULTS and PROGRESS. They bind the original,
storage and numerical-v2 specs, exact pushed execution source, reused evidence
and measured walltimes. The temporary resource-validator time override is
explicit in QUALIFICATION; it is never written into an original scientific
spec or published as an original ACCEPTANCE.

A common submission claim prevents another recovery directory from duplicating
these three fits. Ambiguous partial submission stops for journal inspection.
The scheduler's `--test-only` review must accept the approximately five-day
requests before any job is submitted; an unlimited partition maximum alone
does not establish account/QOS eligibility.

Combined results cover all 42 original scientific slots, across original tasks,
seven authenticated storage salvages, existing storage replacements and the
three numerical-v2 SOFT replacements. RAW remains primary. RAW_BN, EMA05 and
EMA10 are separate readouts. A native/predictor/RNG mismatch with a control
withholds that paired delta while preserving the individually valid result.
Recovery uses the historical anchors; this does not create an EMA-matched
CE/OFFLINE baseline or an independent test.

## Commit and run

Commit the new package, CLI, two wrappers, test and this runbook; the shared plan,
handoff and source map contain unrelated changes, so stage this recovery's
additions by hunk rather than adding those entire files. Push the new commit,
then run the Brown block below. Keep existing pinned checkouts intact.

In local PowerShell, stage the dedicated implementation:

```powershell
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Review the existing staged work before adding this recovery' }
git add -- src/hlt_classification/luka_stability_completion_hydra `
    scripts/luka_stability_completion_hydra.py `
    scripts/queue_luka_stability_completion_hydra.sh `
    sbatch/run_luka_stability_completion_hydra.sh `
    tests/test_luka_stability_completion_hydra.py `
    docs/LUKA_STABILITY_COMPLETION_HYDRA.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git add -p -- docs/plans/LUKA_RECONSTRUCTION_STABILITY_IDEAS.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
```

Select the new completion sections, leaving unrelated hunks unstaged. Then:

```powershell
& {
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged checks failed' }
    git diff --cached --stat
    git commit -m "Recover completed stability fits and qualify longer SOFT replacements"
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
    git rev-parse HEAD
}
```

```bash
bash <<'BASH'
set -euo pipefail
kinit rlyang </dev/tty
read -r -p 'New pushed completion commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
STORAGE="$BASE/runs/stability_storage_794b8f80_r1/recovery_spec.json"
SOFT="$BASE/runs/stability_soft_recovery_d28c4033_4x1080ti_r1/campaign_spec.json"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SOFT")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SOFT")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_stability_completion_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="$BASE/runs/stability_completion_${COMMIT:0:8}_r1"
test ! -e "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
LOG="$(mktemp "$BASE/runs/stability_completion_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Authentication log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_stability_completion_hydra.py" create \
    --root "$ROOT" --storage-spec "$STORAGE" --soft-spec "$SOFT" --expected-commit "$COMMIT" >"$LOG" 2>&1; then
    tail -n 100 -- "$LOG"
    exit 1
fi
tail -n 8 -- "$LOG"
printf 'Recovery spec: %s\n' "$ROOT/recovery_spec.json"
bash "$PROJECT/scripts/queue_luka_stability_completion_hydra.sh" "$ROOT/recovery_spec.json"
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

Expect `lukarsc_SOFT_s7`, `lukarsc_SOFT_s8`, `lukarsc_SOFT_s9`, and
`lukarsc_report_pascal`. Registration and review authenticate saved checkpoints
and may take several minutes. The four additional salvages are available in the
combined results immediately after registration; no GPU job is needed for them.

If submission is interrupted, retain the new root and its exact journal. Do not
choose a different root to retry. If the scheduler rejects the resource request,
no job is submitted; preserve its error for a resource-plan correction.

## View results

Set SPEC to the exact printed path. Use `progress` instead of `results` for
epoch-by-epoch values, and `--readout all` for all four readouts.

```bash
SPEC="/absolute/path/to/the/printed/recovery_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_stability_completion_hydra.py" \
    results --spec "$SPEC" --readout all
```

## Validation and donors

The combined local recovery validation passed **69 checks across two runs**:
66 passed initially; three installed-Weaver checks initially could not import
the dependency. Installing the cached Weaver 0.5.3 wheel into a temporary local
validation directory allowed all three to pass in 80.20s. These exercise real
ParT CUDA miniature exports, including all seven unchanged storage restart
paths and the numerical-v2 SOFT worker. The passes also include the actual
four-process Gloo 30-epoch export/report test, CUDA numerical-v2 tests,
exact measured time-cap reproduction, retained memory
and convergence gates, four-fit salvage, active/ambiguous-job rejection,
submission idempotence, cross-root duplicate prevention and paired-report rules.
Exact pinned-source imports match the Brown CLI layout. CLI help and both Bash
wrappers pass syntax checks. New-file AST/whitespace and tracked diff checks
pass. No remote submission has been made by this session. Real four-card Hydra
setup evidence is reauthenticated at registration; no new Tigris miniature is
claimed for this execution-only repair.

Execution scaffolding and combined-report semantics derive from
`794b8f8016f9b47a3368bc223cecdf01d4d345f4`,
`luka_stability_storage_recovery/{core,runtime,submission,reporting}.py` and its
CLI/wrappers/test. The pinned numerical-v2 scientific worker and qualification
are loaded from `d28c403360f7b4662f679f5564529a048d8db5e2`,
`luka_reconstruction_stability_hydra/`. Original fit validation is loaded from
`3ccc35ae709c63d8833cb0388c618b53dcde5b9e`; shared physics/model/input code stays
at `58d0c0edf5f41cff2398fb168fc16fc92e169294`. No model or loss is migrated.

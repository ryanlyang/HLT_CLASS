# Stability storage-guard recovery

Approved 2026-11-17: remove predictive free-space admission for this recovery.
The original `3ccc35ae` campaign required its entire remaining output allowance
plus 1 GiB to remain available, including after a fit had exported all results.
All ten supplied failures report that guard, rather than an actual write error.
The user's subsequent `df` showed about 18 GiB available. That is a snapshot,
not a reservation or a guarantee of future free space.

This execution-only repair uses the unchanged original training code, scientific
specification, seeds, inputs, four-GTX1080Ti allocation and measured walltimes.
No scientific loss, readout, initialization or selection semantics change.
Available space and the estimate for seven new fits are informational. Neither
the old whole-campaign allowance nor a replacement numeric threshold rejects
a recovery job. Actual file access/write errors, corruption, invalid data,
nonfinite values and provenance mismatches still fail normally.

## Exact scope

| Action | Task | Original job |
| --- | --- | --- |
| Verify and reuse all 30 epochs and exported readouts | R_L25_s9 | 180544 |
| Verify and reuse all 30 epochs and exported readouts | R256_s9 | 180546 |
| Verify and reuse all 30 epochs and exported readouts | C512_s8 | 180557 |
| Restart in a fresh output root | DEC2_s8 | 180559 |
| Restart in a fresh output root | PROJ_s9 | 180560 |
| Restart in a fresh output root | C_L50_s9 | 180561 |
| Restart in a fresh output root | R_L50_s9 | 180562 |
| Restart in a fresh output root | C512_s9 | 180563 |
| Restart in a fresh output root | R512_s9 | 180564 |
| Restart in a fresh output root | DEC2_s9 | 180565 |

Registration requires these exact jobs to be terminal `FAILED|1:0` in the
original submission journal, with only the recognized storage exception in
their logs. Each salvage must pass the original full fit validator: all epoch
records, native checkpoints, aligned prediction arrays, selected metrics and
fit receipt. Its discarded startup miniature and original claim also validate.
The other seven must have no original task directory or claim. Unexpected
partial training or an additional failure cause stops this recovery.

The three complete fits are not copied, retrained or marked Slurm COMPLETED.
Their original missing task receipts are not fabricated. A new recovery spec
authenticates their original fit receipts and labels them explicitly in combined
reporting. Original campaign files, jobs and source checkouts are preserved.
The seven new fits parent the original scientific spec and receive separate
execution receipts bound to this recovery, its new source and exact new job IDs.

The original successful GPU setup is fully validated and reused. Each new fit
still runs the original discarded production miniature and memory checks before
science. There is no new setup job: submission queues seven independent GPU
fits, then one CPU report with `afterany` on those seven. A common immutable
submission claim prevents another recovery root from duplicating these jobs.
An ambiguous partial submission stops; there is no blind automatic retry.

The combined report includes authenticated original completed fits, the three
salvages, and completed replacements. RAW remains primary; RAW_BN, EMA05 and
EMA10 remain separate. Native/predictor/RNG mismatches withhold paired deltas
while retaining individually valid results. The earlier three SOFT failures
and their separate numerical recovery are outside this repair.

## Versioned execution contracts

New records use `LUKA_STABILITY_STORAGE_<KIND>/v1`, explicit content/parent
hashes and the original exploratory-validation flags. `SPEC` binds the exact
original spec, failure logs, complete-fit receipts, qualified acceptance,
new pushed source and policy `available_space_advisory_actual_io_errors_fatal/v1`.
`PLAN`/`SUBMIT_CLAIM` bind eight exact commands and their journal. `RUN_CLAIM`/
`TASK` bind replacement execution; `RESULTS`/`PROGRESS` distinguish old,
salvaged and replacement evidence. Old scientific records retain their original
contracts. There is no final-test inference or new scientific quality gate.

## Commit and queue

Commit and push only the new recovery package, its CLI/two wrappers, test and
this runbook. Shared handoff/source-map/plan links can be staged separately by
hunk, since they may contain unrelated work. The live original checkout must
remain unchanged. After pushing, run on a Brown login node:

```bash
bash <<'BASH'
set -euo pipefail
kinit rlyang </dev/tty
read -r -p 'New pushed storage-recovery commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
PARENT="${BASE}/runs/reconstruction_stability_3ccc35ae_4x1080ti_r1/campaign_spec.json"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$PARENT")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="${BASE}/code_stability_storage_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="${BASE}/runs/stability_storage_${COMMIT:0:8}_r1"
test ! -e "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
LOG="$(mktemp "${BASE}/runs/stability_storage_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Authentication log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_stability_storage_recovery.py" create \
    --root "$ROOT" --parent-spec "$PARENT" --expected-commit "$COMMIT" >"$LOG" 2>&1; then
    tail -n 100 -- "$LOG"
    exit 1
fi
printf 'Recovery spec: %s\n' "$ROOT/recovery_spec.json"
tail -n 8 -- "$LOG"
bash "$PROJECT/scripts/queue_luka_stability_storage_recovery.sh" "$ROOT/recovery_spec.json"
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

Expect seven `lukarsf_<task>` jobs and `lukarsf_report_pascal`. The report is
automatic. Registration and review authenticate large saved artifacts and can
take several minutes. Do not create a second recovery root if submission is
interrupted; retain the exact journal and inspect it.

To view results immediately, including the three salvages, use the exact printed
recovery spec path for `SPEC` below. These commands work on a login node with
valid filesystem credentials and do not require new free-space reservations.

```bash
SPEC="/absolute/path/to/the/printed/recovery_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_stability_storage_recovery.py" \
    results --spec "$SPEC" --readout all
```

Replace `results` with `progress` for epoch-by-epoch saved values. Actual
filesystem exhaustion or expired credentials can still prevent reading/writing;
this repair does not alter authentication, delete files or claim extra quota.

## Local evidence and donors

Focused recovery validation: **23 passed, no skips, in 47.08s**. This includes
zero-free-space advisory behavior, propagation of actual write errors, exact
failure/job authentication, corrupt-artifact rejection, seven-fit/eight-job
submission with no duplicate retries, and the unchanged site allocation checks.
Actual four-process Gloo tests run 30 miniature epochs and combine original
salvage and replacement exports across all four readouts. Installed-Weaver CUDA
miniatures exercise all seven restart tasks and validate their exported models.
These synthetic local runs do not substitute for four-card Hydra qualification;
that existing successful original setup is authenticated and reused.

Before implementation, focused original resource/registration/submission/report
checks passed (12 passed, 49 deselected, 30.21s). Recovery imports also pass with
the exact `3ccc35ae` package and `58d0c0ed` dependencies. Both Bash wrappers pass
syntax checks. No remote jobs were submitted by this implementation session.

The original serialized inventory, evaluated locally for 298 output slots and
the original population sizes, estimates **2.358 GiB** for seven replacement
fits plus report allowance. The original whole-campaign budget is 29 GiB.
The estimate is informational; registration records the actual original spec's
inventory and current filesystem observation. Actual output size/free space can
differ. There is no space threshold to override or quota number to invent.

Original scientific donor: `58d0c0edf5f41cff2398fb168fc16fc92e169294`.
Original stability worker: `3ccc35ae709c63d8833cb0388c618b53dcde5b9e`.
The recovery dynamically calls the latter's campaign, submission, worker,
training, resources, kernel, identity, contracts and reporting modules;
their source is never edited. Execution wrapper/test scaffolding follows
the corresponding stability files at local baseline
`8a25f12653d0bcfefe170273d3f3e0748b4b508f`. New execution contracts are described
above; no numerical/model implementation is migrated.

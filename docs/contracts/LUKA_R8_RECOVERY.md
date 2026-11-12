# R8 pre-claim recovery v1

This execution-only recovery covers original C3 job168726 and R14 job168727
from source58d0c0edf5f41cff2398fb168fc16fc92e169294. Both failed at
`scontrol show job` before claims or training output existed. Original setup
168256 and release168257 completed successfully. The controller later purged
the failed job records. No scientific result or seed is being rejected.

New `LUKA_R8_RECOVERY_{SPEC,PLAN,SUBMIT_CLAIM,RUN_CLAIM,TASK,RESULTS}/v1`
artifacts bind original spec/setup acceptance, setup receipt, failure logs,
original immutable ledger, new pinned recovery source and replacement job IDs.
Only C3/R14 may be retried, only after exact FAILED/1:0 accounting and absence
of any original claim/output. New attempts write to a separate recovery root.
Original source, campaign specs, setup, ledgers, report and other jobs are never
rewritten. Claims prevent duplicate execution; ambiguous submission journals
stop instead of resubmitting. Repeating a fully completed submission is idempotent.

The CLI loads the recovery driver from its new worktree but imports the entire
scientific package from the original immutable worktree. The original source
validator, authenticated measured setup, native training and export are reused.
The donor training function receives the unmodified original spec and original
target paths. C3/R14 retain seeds, losses, 30 epochs, batch128, one A6000,
native BF16 and decoder FP32. No setup or teacher fit is repeated.

Only failed read-only `scontrol show job/config` calls are retried, at most
three attempts with5/10-second delays. Each failure prints captured stdout and
stderr. Allocation mismatches, nonfinite training, source drift and other
exceptions are not retried or bypassed. These limits bound attempt count;
individual Slurm RPCs still use the site's timeout. Fit requests preserve
measured minutes plus15 minutes for startup/retries within the original48-hour
ceiling. No automatic training restart is enabled.

One reviewed submission queues two independent fits and a CPU afterany report.
The recovery report merges authenticated completed original and replacement
results, explicitly marks replacement provenance and recomputes R14-C3.
It may be partial if other original fits are still running; rerun `results`
later. Original report168742 remains unchanged. All original input/setup files
and both worktrees must remain available. No final-test access is introduced.

The original real setup qualifies the unchanged scientific execution. Local
failure injection and production-fit tests qualify the new orchestration;
remote replacement-job completion remains to be established.

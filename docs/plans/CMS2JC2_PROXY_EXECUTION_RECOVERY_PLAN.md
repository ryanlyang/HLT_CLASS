# Source-pinned production scheduler recovery

## Authorized scope

Ryan authorized implementation on 2026-09-30 following failure of array
element 209946_1. Keep the original dataset root, immutable study, successful
preflight, original attempts, reservations and committed 4,432-jet shard.
Add a separately authenticated execution repair; never rewrite historical
source hashes. No cancellation, deletion, fitting, new split or benchmark.
This amends only the execution-source boundary of the production plan.

The repair is specifically for a pre-bulk campaign with a completed preflight,
one completed TRAIN pilot and one missing TRAIN pilot. All original jobs must
be terminal, exactly journal-bound, and absent from the queue. Unknown or
ambiguous submissions stop. Existing incomplete files/reservations stay and
count against the same storage budget.

## Source and artifact boundaries

The original worktree must still match its frozen source. A new clean pushed
worktree pins the replacement source and queue helper. Every old source file
must match except explicitly enumerated execution functions and production
documentation. For modified Python modules compare their ASTs
after removing only the enumerated operational functions. In particular,
`worker.generate`, population/identity construction, storage, physical array
encoding, response model, random streams and numerical environment may not
change. Additional files are restricted to the new recovery modules/helper
and this plan/contract. This is an execution-equivalence check, not a general
permission to migrate scientific code.

An EXECUTION_REPAIR/v1 record pins both sources, the original study/preflight,
original attempt and ledger references, and retained shard receipt hashes.
New attempts and shards have distinct ATTEMPT_EXECUTION_REPAIR/v1 and
SHARD_EXECUTION_REPAIR/v1 contracts with a reference to that repair. New shard
`source` records the actual replacement source and `scientific_source` the
original source; old receipts retain their original meaning and bytes.
The complete manifest uses a distinct DATASET[_REDUCED_CONFIRMATION]_EXECUTION_REPAIR/v1
kind, carrying both sources and the unchanged confirmation limitations.
Readers validate the new lineage explicitly; old schemas are not relabeled.

## Review and automatic continuation

Preparation is dry: authenticate, freeze the repair, write one missing-pilot
attempt and an immutable RECOVERY_PLAN/v1. No job is submitted. The plan binds
the exact initial command plan and the automatic continuation policy: unchanged
resource formula, two retained pilot receipts, 36 CPUs per task, at most 16
concurrent tasks, 32/64/128 GiB, 1--8h, original total storage budget, all remaining
frozen shards, and one afterany finalizer. The initial retry reuses the successful
preflight rather than scheduling another historical audit.

Executing the reviewed recovery-plan hash plus explicit sealed-test
materialization permission authorizes the bounded continuation as well as the
initial retry. The disconnect-safe helper holds a process-released exclusive
controller lock, submits the missing pilot once, waits for exact successful
accounting and authenticated receipt, then calculates and validates the bulk
plan and submits generation plus finalizer. At most one attempt may be active.
Every scheduler submission still has its exact plan hash, authorization phrase,
durable intent and response. Do not submit a new attempt automatically after
any failure. Missing accounting may be polled for up to 14 days; failed jobs,
wrong identities, corruption or unsafe resource/storage envelopes stop.

Restarting the controller may follow its already journaled jobs or return an
already submitted bulk ledger, but never repeat an ambiguous sbatch. A crash
between creating a continuation attempt and recording its reference stops for
inspection. No automatic deletion or silent adoption of orphan attempts.
The helper ends once bulk and its finalizer are queued; that is NOT a complete
dataset claim. The final manifest remains the completion marker.

The original test seal is unchanged: materialization only after the explicit
build lock; no classifier evaluation, tuning, label inputs or native HLT reads.
Poor confirmation quality remains a recorded limitation, not an execution gate.

## Verification

Test source drift and forbidden scientific edits, lineage/schema confusion,
successful shard preservation, retry exclusion, CPU/resource bounds, dry-run
non-submission, worker receipt publication, terminal/no-overlap guards,
controller restart and ambiguous-submission refusal, pilot failure stopping
bulk, and final manifest/reader behavior. Existing genuine completed Tigris
pilot is retained; the retried pilot validates the corrected worker before
bulk release. Local tests do not claim that remote retry has already passed.

## Operator entry point

After committing/pushing, use a clean detached Tigris worktree at that exact
commit. The original worktree and dataset root must remain in place. From the
new worktree run:

```bash
bash scripts/queue_cms2jc2_proxy_recovery.sh OLD_ROOT/study_spec.json EXACT_NEW_COMMIT
```

This authenticates and prints the initial retry commands, preserved jet count,
source commit and bounded continuation policy, then the recovery plan hash.
After reviewing, execute:

```bash
bash scripts/queue_cms2jc2_proxy_recovery.sh OLD_ROOT/study_spec.json EXACT_NEW_COMMIT \
  --execute REVIEWED_RECOVERY_PLAN_HASH
```

Execution automatically detaches with `nohup`, prints a PID/log path, and records
`RECOVERY_HELPER_EXIT` in that log. There is no further operator handoff between
pilot and bulk. A zero exit plus `recovery/handoff.json` means generation and
finalizer were queued, not that all 2.25M jets are complete. All further failures
are preserved for inspection; no cancel/requeue loop is started.

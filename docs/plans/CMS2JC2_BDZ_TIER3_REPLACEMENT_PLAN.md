# B_DZ comparison: tier3 execution replacement

## Authority and frozen science

Ryan authorized moving the pending B_DZ comparison to tier3 on 2026-09-25.
This execution-only extension supersedes the debug-only/no-cancellation rules
of `CMS2JC2_BDZ_TRACKING_TUNING_PLAN.md` only for the five ledger-bound pending
comparison tasks. It does not authorize changing any other campaign or
repeating calibration. The original plan's scientific meaning remains binding.

Use a fresh disjoint root and clean pushed source. Authenticate the original
`BDZ_STAGE/v1` comparison, its complete live submission ledger and journals,
the completed gate acceptance and 4,000-jet calibration receipts, and all
earlier frozen inputs. Reuse the exact calibration registry/map, response,
data identities, four development shards, replicas, diagnostics and selection.
Require the same numerical environment and byte-identical scientific source,
including `bdz_campaign.py`, `bdz_worker.py`, `bdz_maps.py`, `bdz_metrics.py`.
Only the pre-existing execution-only source allowlist may differ.

The unchanged scientific worker publishes the same BDZ_SHARD/SELECTION kinds
under the new stage hash. No original file is overwritten, copied as a new
scientific result or deleted. Confirmation/test and JetClass2 particles remain
untouched. Calibration quality is not a gate and no new parameter is tuned.

## Exact execution

New stage `bdz_compare_tier3_r1`, contract `BDZ_TIER3/v1`:

- `bz_eval_0..3`: one node/task, 36 CPUs, 128 GiB, 8 hours, no GPU.
- `bz_select`: one node/task, 1 CPU, 32 GiB, 4 hours, afterok all four NEW IDs.
- Partition tier3, account reu-aisocial, QOS qos_tier3, atlas_kd_sporc.
- Same bounded spawn pools, numerical thread limits and progress logging.
- No gate jobs or dependency on old scheduler IDs in the replacement DAG.

The observed hypothetical tier3 start was earlier than debug, not a guarantee.
Reuse the successful real SPORC serial/process and resource acceptance: this
is a partition-only move, not a claim of measured tier3 completion/runtime.

## Safe replacement protocol

`create-bdz-tier3` creates metadata only. It refuses a wrong-family subject,
an overlapping root, or any original comparison output receipt OR worker
claim. `dry-run` reviews exactly five CPU-only commands. `retire-bdz-tier3`
without execute previews the original exact job states and required phrase.

Live retirement requires the reviewed plan hash and
`AUTHORIZE CMS2JC2 BDZ TIER3 REPLACEMENT`. It first checks tier3 admission with
test-only requests. All five old jobs must be pending or non-success terminal;
RUNNING, COMPLETING, COMPLETED, SUSPENDED and UNKNOWN cause a stop before any
cancellation. Authenticate each pending job's owner, name, stage comment,
worktree, script, account, QOS, partition and CPU allocation against the old
acknowledged journals. Cancel only those exact IDs with controller-side
PENDING/debug/account filters. A pending-to-running race must not cancel the
running job. Recheck scheduler states and old claims/receipts after cancellation.
Accounting lag stops the operation without a retirement receipt; inspect and
explicitly retry, never loop cancellation automatically.

If any old work has started or completed, preserve it and stop for a revised
reuse plan. This version deliberately does not silently discard partial or
completed shards. All five tasks were pending in the user's snapshot; that
snapshot is not treated as live authority.

Immutable retirement evidence binds the old ledger, new stage/plan and exact
terminal states. Live submit requires it and a fresh state check, then the
existing `AUTHORIZE CMS2JC2 BDZ_COMPARE EXACT PLAN` phrase and plan hash.
Workers verify retirement evidence and reused artifacts before claiming work.
The submission journal prevents duplicate jobs; ambiguous intent requires
explicit reconciliation, not deletion or blind retries. Creating the new
study does not itself authorize or perform a Git push or remote mutation.

Use `bdz-results --spec <new stage>` after completion. It authenticates the
selection and shard hashes and reports scores/figures. Stop after this same
development screen; no automatic follow-up or production qualification.

## Verification

Test original gate/map reuse, identical tiny ROOT numerical results, unchanged
donor bytes, exact resources/dependencies, wrong-family/source/map rejection,
separate authorization, target partition admission before cancellation,
pending-to-running/completed races, accounting lag, original claims/receipts,
retirement-before-submit, idempotent acknowledged submission and unrelated-job
isolation. Run B_DZ and existing migration regressions. Weaver is inapplicable.

# Frozen C observed-topology diagnostic: debug replacement

The user authorized moving the pending C topology jobs to debug on 2026-09-25.
This execution-only extension supersedes the tier3/no-cancellation rules of
`CMS2JC2_C_OBSERVED_TOPOLOGY_DIAGNOSTIC_PLAN.md` only for the five authenticated
unfinished evaluation/report tasks of the supplied original C stage. B's debug
jobs and all classifier jobs are out of scope. No automatic Git push or live
submission is authorized by merely creating this specification.

## Unchanged science

Reuse the completed original `DEV_C_TOPOLOGY/v1` acceptance and its original
frozen C_L comparison, model, samples, ranges, association policy and numerical
environment. Require exact scientific source bytes, including the C topology
generator/worker/results, as well as original receipt and artifact hashes.
Only the already registered execution-only source surface may differ.

Retain all 10,000 development evaluation jets, four 2,500-jet shards, replicas
0/1/2 and the six original free/fixed views. Evaluation associations ARE still
recomputed by the unchanged worker. There is no refit, new matching policy,
regeneration of fitting targets, altered RNG, JC2 read or final-test access.
Observed topology remains privileged diagnostic information, not a deployable
response. Resolution and prediction quality remain results, never failure gates.

Use a clean source-pinned worktree and fresh disjoint root. Preserve original C
and B roots/worktrees. Reuse completed acceptance; do not rerun or republish it.
If an old evaluation/report has a receipt or has started/completed, stop and
reassess reuse rather than discarding its work.

## Execution

New stage `observed_topology_debug_r1`, contract `DEV_C_TOPOLOGY_DEBUG/v1`:

- `ct_eval_0..3`: each debug, one node/task, 36 CPUs, 128 GiB, **8 hours**, no GPU.
- `ct_report`: debug, one node/task, 1 CPU, 32 GiB, 4 hours, afterok all four NEW
  evaluation IDs. No dependency on the historical acceptance scheduler ID.
- Account `reu-aisocial`, QOS `qos_tier3`, existing atlas_kd_sporc worker and
  numerical environment. Existing bounded pools, eight-jet chunks, progress
  reporting, storage envelopes, claims and output receipts are unchanged.

Eight hours is an unmeasured trial limit; timeout would require a separate
restart. Debug shares two 36-core nodes with B in the user's site snapshot, so
this cannot promise an immediate start or four simultaneous shards.

## Authenticated replacement protocol

`create-c-topology-debug --parent-spec <OLD C topology stage> --project-dir
<clean checkout> --source-commit <pushed full hash> --root <fresh root>` imports
the original six-job acknowledged ledger/journals and completed acceptance.
Creation and normal `dry-run` never cancel or submit jobs. Review all five
new commands and their exact old task-to-job mapping before live actions.

`retire-c-topology --spec <NEW debug stage>` prints a read-only state preview.
Live retirement additionally requires `--execute`, the registered plan hash,
and `AUTHORIZE CMS2JC2 C TOPOLOGY DEBUG REPLACEMENT`. First prove debug admission
with test-only Slurm requests; require original acceptance COMPLETED/0:0 and
validate its content/lineage/codec evidence. Check exact original pending job
names, owner, comment, worktree, script and one-node CPU-only tier3 allocation.

Refuse active, successful or unknown original targets before ANY cancellation.
Use one controller-filtered `scancel --ctld --state=PENDING` restricted to the
exact original C target IDs, account and partition. Never cancel by name alone.
Recheck accounting and receipts after cancellation; if accounting lags or a
target starts/completes, stop without publishing retirement evidence. The user
may explicitly retry retirement after inspection; there is no automatic retry.

Immutable retirement evidence binds the new stage/plan and old acknowledged
ledger. Every live `submit` requires that receipt AND a fresh check that all
old targets are non-success terminal. Submission retains the original CTOPO
phrase `AUTHORIZE CMS2JC2 CPU DEVELOPMENT CTOPO EXACT PLAN`, plan review, site
checks and intent/receipt journal. Workers verify retirement evidence and reused
acceptance before claiming work; they need not poll historical Slurm jobs.

Use `c-topology-results --spec <NEW debug stage>` for unchanged authenticated
reporting. Partial/ambiguous submission uses existing reconciliation; never
delete an existing root, blindly resubmit, or modify old jobs/artifact files.

## Validation and readiness

Test exact five-job resources/dependencies, C acceptance/source/ledger reuse,
unchanged tiny ROOT evaluation/report and donor bytes, admission before cancel,
pending/running races and accounting lag, separate retirement authorization,
retirement-before-submit, wrong-family rejection and B isolation. Rerun existing
B debug and C topology regressions. Local results cannot establish remote queue
wait/runtime; real reuse remains conditional on authenticated original SPORC
acceptance and unchanged environment/source. GPU/Weaver parity is inapplicable.

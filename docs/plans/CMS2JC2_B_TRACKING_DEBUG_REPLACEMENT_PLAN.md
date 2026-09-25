# Frozen B tracking: debug replacement

Authorized 2026-09-25 by the user to move the pending B audit to debug, with
exact-ID cancellation/replacement if needed. This is an execution-only extension
of `CMS2JC2_FROZEN_B_TRACKING_AUDIT_PLAN.md`, not a new response experiment.
It supersedes that plan's tier3-only/no-cancellation rule ONLY for the five
authenticated unfinished tasks of the explicitly supplied B subject. C and all
other campaigns remain untouched. No implicit live submission or Git push.

## Frozen science and reuse

Import the original tier3 `DEV_B_TRACKING/v1` stage, its full acknowledged
submission ledger/journals, and its completed authenticated `bt_acceptance`
result/receipt. Require exact frozen response, samples, ranges, variants, keys,
numerical environment and all original scientific source bytes, including B's
generator, metrics, worker and result reader. Only the established execution-only
dispatch/submission files may differ. Reuse the original B_L comparison through
its existing receipt authentication. Do not refit, rematch, rerun acceptance,
read JetClass2, access final test, or regenerate a completed B evaluation.

Use a fresh disjoint source-pinned worktree/study. Retain the old root and all
its products. Reject this replacement if any evaluation or report has already
published a receipt; do not discard that work. If an old task starts or completes
during migration, stop for an explicit revised plan rather than cancelling it.

## Exact execution

Stage `frozen_b_tracking_debug_r1`, new `DEV_B_TRACKING_DEBUG/v1` contract.
All five jobs use debug / reu-aisocial / qos_tier3, one node/task, zero GPUs.
Four independent `bt_eval_0..3` jobs request 36 CPUs, 128 GiB, **8 hours** each.
The imported acceptance is validated before submission and at worker startup;
no stale scheduler dependency on its historical ID is needed. `bt_report`
requests 1 CPU, 32 GiB, 4 hours and afterok all four NEW evaluation IDs.
Every shard retains all 2,500 jets, replicas 0/1/2, and all four interventions.
Pools, chunks, progress, counters and outputs are unchanged. Eight hours is an
unmeasured trial cap: a timeout requires restarting the affected shard. Debug's
two nodes can run at most two 36-CPU evaluations at once; no queue-time promise.

## Migration protocol

1. Create and review the full five-command dry plan before touching old jobs.
2. A separate `retire-b-tracking --spec <new debug spec>` is read-only by default.
   It prints exactly the five original B evaluation/report IDs and their states.
3. `--execute`, the exact reviewed plan hash, and
   `AUTHORIZE CMS2JC2 B TRACKING DEBUG REPLACEMENT` are required to retire them.
   First verify debug resource admission with test-only Slurm requests. Require
   old acceptance COMPLETED/0:0. Authenticate pending jobs against their original
   names, owner, stage comment, worktree, script and CPU-only tier3 allocation.
   Refuse RUNNING, COMPLETING, unknown or completed target states before ANY
   cancellation. Issue one `scancel --ctld --state=PENDING` with exact bound IDs
   only. Controller-side filtering protects a pending-to-running race. Recheck all states;
   if accounting lags or a job started, publish no retirement receipt and stop.
   A later explicit retry may confirm terminal states; it cannot cancel running
   work. Never modify/delete the original metadata or cancel by name/user alone.
4. Publish authenticated `DEV_B_TRACKING_DEBUG_RETIREMENT/v1` evidence only when
   all five targets are non-success terminal and acceptance succeeded. It binds
   the new stage/plan and original acknowledged ledger. No automatic retry loop.
5. Normal live `submit` additionally requires this receipt AND a fresh scheduler
   check that no original B target can run. Standard BTRACK authorization,
   reviewed plan, site admission and submission intent/receipt journal remain.
   Thus ordinary submit cannot bypass retirement. Workers validate retirement
   evidence without polling Slurm for old jobs on every task.

Failed/ambiguous submissions retain their root and require the existing explicit
reconciliation process. No duplicate audit, source edits in pinned worktrees,
old acceptance re-publication, or mixed old/new output tree.

## Validation

Test unchanged legacy tier3 registration, five-job debug DAG/resources, complete
mock submission and idempotence, acceptance/source/ledger corruption, output
preservation, pending-only exact cancellation, refusal of running/completed or
unknown subjects, accounting lag/race, retirement-before-submit, and unchanged
worker evaluation/report behavior. The real SPORC B acceptance may be reused
only with the source/environment/receipt checks above. This does not establish
a measured debug runtime or scientific qualification. No GPU/Weaver test applies.

## Operator entry points

After a scoped push, use a clean detached worktree at that exact commit and the
unchanged `atlas_kd_sporc` environment. `--parent-spec` below is the OLD six-task
frozen-B audit's `stage_spec.json`, not the original A/B/C comparison. Locate it
through its authenticated live ledger, requiring the exact expected B job map;
do not select the newest directory or use a broad name filter for cancellation.

```bash
python -s "${PROJECT_DIR}/scripts/cms2jc2_response_dev.py" create-b-tracking-debug \
  --parent-spec "${OLD_B_SPEC}" --project-dir "${PROJECT_DIR}" \
  --source-commit "${PINNED_COMMIT}" --root "${DEBUG_ROOT}"

SPEC="${DEBUG_ROOT}/stages/frozen_b_tracking_debug_r1/stage_spec.json"
CLI="${PROJECT_DIR}/scripts/cms2jc2_response_dev.py"
python -s "${CLI}" dry-run --spec "${SPEC}"
python -s "${CLI}" retire-b-tracking --spec "${SPEC}"
```

Review all five commands and old task IDs. Only then set `PLAN_HASH` to that
registered command plan's content hash and explicitly run:

```bash
python -s "${CLI}" retire-b-tracking --spec "${SPEC}" --execute \
  --reviewed-plan-hash "${PLAN_HASH}" \
  --authorization-phrase "AUTHORIZE CMS2JC2 B TRACKING DEBUG REPLACEMENT"

python -s "${CLI}" submit --spec "${SPEC}" --execute \
  --reviewed-plan-hash "${PLAN_HASH}" \
  --authorization-phrase "AUTHORIZE CMS2JC2 CPU DEVELOPMENT BTRACK EXACT PLAN"
```

Stop on any error; do not continue to submit after unsuccessful retirement.
If accounting has not caught up after cancellation, retain the created study,
inspect its exact old IDs and explicitly retry only retirement, not creation.
If any old task ran/completed, reassess reuse rather than cancelling it.
Results use `b-tracking-results --spec "${SPEC}"`; monitoring uses
`monitor --spec "${SPEC}"`. Both retain the usual authenticated readers.

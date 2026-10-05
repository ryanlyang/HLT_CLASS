# JetClass2 dzfix fusion: final-direct node-failure recovery

Authority: the 2026-10-05 amendment in
`docs/plans/JETCLASS2_DZFIX_FUSION_CHAIN_500K_PLAN.md`.

## Scope and source

This is a narrow, explicitly authorized restart-zero adapter, not a general
retry policy. The only supported scientific donor commit is
`fd1c05786287c57b080b2d229dd64664f830a571`. Only these tasks may be replaced:

- `train_FINAL_DIRECT_D000`;
- `aggregate`;
- `complete`.

The original `train_FUSION_D000_D000`, `reduce_FUSION_D000_D000`, and
`train_FINAL_BRIDGE_D000` jobs and artifacts remain in place. No other campaign
is in scope. A terminal unsuccessful final-direct accounting record and its
node-failure log are required. An expired Slurm controller record is allowed;
`sacct` supplies the failed job's exact-ID state. No selected checkpoint or
completed training report may be silently overwritten.

## Artifacts and authentication

New artifacts have the `JETCLASS2_DELPHES_DZFIX_FUSION_RECOVERY_` prefix and `/v1`
version: `SPEC`, `PLAN`, `RESERVATION`, `INTENT`, `EXECUTION`, and `RELEASED`.
Each carries a content hash and `final_test_accessed: false`. The recovery spec
binds the original campaign and live-ledger hashes, six exact original IDs,
accounting evidence, node-failure log bytes, completed task hashes, the entire
partial output inventory, external dependencies, helper commit and bytes.
The canonical recovery root is `<original>/recoveries/<new-name>`.

The CLI puts the original checkout first on the import path before importing
any `hlt_classification` module, then loads the new adapter by filename. Both
checkouts must be clean and pinned. The original scientific validator, genuine
execution acceptance, completed teacher reducers, fit implementation, and
publication path are reused unchanged. There is no claim of a new local GPU
acceptance and no weakened scientific gate.

The new worker authenticates its real Slurm ID against its separate live ledger
using the original resource/site/environment validator. It additionally checks
the new wrapper path, original working directory, and exact `jc2fcr_` job name.
A scoped process-local dispatch adapter permits only the already-authenticated
task; it never fakes an environment job ID or edits the original ledger.

## Submission and dependencies

Default operation creates an immutable three-task dry run. Live submission
requires the exact phrase:

```text
AUTHORIZE DZFIX FUSION FINAL DIRECT NODE FAILURE RECOVERY
```

The original scheduler/resource requests are preserved, including tier3,
`qos_tier3`, one A100 and the original training walltime. The replacement fit
is initially held. The new aggregate depends on it and, if unfinished, the
original final-bridge job. The new complete depends on the new aggregate.
Completed prerequisites are verified through artifacts; the failed direct ID
is never used as an `afterok` dependency.

Durable submission intents/journals distinguish acknowledged prefixes from
ambiguous scheduler acknowledgements. The latter fail closed, never silently
resubmit. Exclusive claims and a campaign-level task reservation prevent
competing recovery roots. Only after all three replacement IDs are published
may the exact old pending aggregate/complete jobs be canceled, after owner,
account, working-directory and job-name checks. The fit is released only after
those summaries are terminal. Delayed cancellation accounting leaves the new
fit held; rerun submission after accounting catches up. An interrupted worker
requires operator review, not automatic retry.

## Outputs and operation

Before fitting, archive the byte-verified partial final-direct output tree
under the recovery root. Paths must remain inside the registered campaign and
must not traverse symlinks. No files are deleted. Scientific outputs and task
receipts are then published at their original canonical locations, with new
companion execution receipts binding the helper, recovery spec, actual job ID,
and scientific receipt. Existing results commands remain valid.

From the clean, pushed recovery checkout on SPORC:

```bash
bash scripts/queue_jetclass2_dzfix_fusion_recovery.sh --dry-run
bash scripts/queue_jetclass2_dzfix_fusion_recovery.sh --execute
```

The helper defaults to the known `jc2_dzfix_fusion_tier3_coarsefd1c0578_r1`
campaign. `SOURCE_SPEC` and `RECOVERY_ROOT` may specify explicit paths subject
to the same validation. It does not fetch or modify either checkout. Repeating
an acknowledged successful submission returns its ledger without duplicates.

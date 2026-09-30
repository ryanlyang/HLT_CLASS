# CMS2JC2 proxy execution repair / v1

Authority: [execution recovery plan](../plans/CMS2JC2_PROXY_EXECUTION_RECOVERY_PLAN.md).

New CMS2JC2_PROXY kinds (schema version 1): EXECUTION_REPAIR, RECOVERY_PLAN,
RECOVERY_CONTINUATION, RECOVERY_HANDOFF, ATTEMPT_EXECUTION_REPAIR,
SHARD_EXECUTION_REPAIR, DATASET_EXECUTION_REPAIR and
DATASET_REDUCED_CONFIRMATION_EXECUTION_REPAIR. All use canonical content hashes,
exact parent hashes, immutable atomic publication and original test flags.

The repair lives at `ROOT/recovery/repair_spec.json`; it is unique per dataset.
It references the original unchanged study, preflight, original attempts and
submission ledgers, retained receipts, and the replacement source snapshot and
worktree. Reference validation precedes reuse. Replacement attempts explicitly
reference this record. No implicit source override through path existence.

Repaired shards pin their attempt and repair, actual execution source hash,
original scientific source hash, unchanged environment and population. Old
SHARD/v1 receipts remain valid only with the original source and without repair
fields. The final repaired manifest carries the repair and `execution_source`;
its `source` remains the original scientific source. Sealed-test data are
materialized, not evaluated; the ordinary reader remains train/validation only.

One reviewed RECOVERY_PLAN authorizes the exact missing-pilot plan and the
bounded measured continuation policy. Automatic continuation publishes the exact
bulk command plan before submitting it. Submission journals, no-overlap checks,
test build lock and conservative disk/resource admission are not bypassed.
No requeue, cancellation, cleanup or scientific refit is authorized here.

# CMS2JC2 proxy dataset production / v1

See the active [implementation plan](../plans/CMS2JC2_PROXY_DATASET_PRODUCTION_PLAN.md).

All new objects use `CMS2JC2_PROXY_<KIND>/v1`, schema version 1, canonical
content hashes, parent hashes, and immutable publication. Old
`CMS2JC2_RESPONSE_*` objects retain their original scope and semantics.

Reduced confirmation is separately defined in
[CMS2JC2_REDUCED_CONFIRMATION.md](CMS2JC2_REDUCED_CONFIRMATION.md).
Only the explicit opt-in route emits
`CMS2JC2_PROXY_STUDY_REDUCED_CONFIRMATION/v1` and
`CMS2JC2_PROXY_DATASET_REDUCED_CONFIRMATION/v1`; neither is interchangeable
with the original STUDY/DATASET contracts. Both preserve the reduced coverage,
subset result, and incomplete full-population status. Output fields, population,
storage admission, generation kernel, and final-test restrictions are unchanged.

Dataset: one frozen JOINT replica-zero proxy per selected dzfix offline jet.
Counts: TRAIN 1,000,000; validation 250,000; sealed final test 1,000,000.
An authenticated same-registry TRAIN_1M profile defines the outer roles.
Validation subset selection is label-independent, defined by the plan.

Block fields are exactly `offsets` (int64), `jet_identity` (uint8 [jets,32]),
`p4` (float64 [particles,4], px/py/pz/E GeV), `charge` (int8), `category`
(int8, six response PIDs), `tracking` (float64 [particles,4], d0/dz/d0err/dzerr
in the frozen bridge's mm/sign convention), `valid` (bool [particles,4]).
No labels, original offline values, construction keys, or mapping indices are
deployable feature fields. Offsets permit empty jets. Identity is metadata,
not a model feature. Invalid tracking values are zeroed; valid errors are
positive. Physical invariants follow the unchanged response `Particles` schema.

Final-test **materialization** is a distinct permission from evaluation:
test shard receipts and the complete manifest truthfully set
`final_test_accessed=true`, `final_test_materialized=true`, and
`final_test_evaluated=false`. Metadata-only setup and pilot receipts set
accessed/materialized false. The production reader exposes only train and
validation and rejects final-test reads; a later explicitly authorized
inference contract is needed. Raw native HLT is never read.

The complete manifest is the commit marker. It binds source, environment,
frozen response, reviewed CMS confirmation, explicit test build lock,
population, each shard receipt, and all block byte hashes. It declares
`dataset_kind=controlled_CMS_calibrated_proxy` and
`physics_production_qualified=false`. Successful assembly does not relabel
the proxy as real HLT. Source limitations and confirmation outcomes remain
in the durable provenance directory.

Submission uses exact reviewed plan hash and an explicit stage-specific
authorization phrase. CPU-only Tigris arrays request 36 CPUs per element and
at most 16 simultaneous elements. Exact scheduler IDs, immutable intents and
submission receipts provide the recovery boundary. Ambiguous submission is
not automatically retried. Existing live campaigns are never mutated.

Scheduler authentication clarification (2026-09-30, no schema/physics change):
generation addresses the exact `ArrayJobId_ArrayTaskId` selector, then checks
the returned raw `JobId` against the worker environment. `scontrol -o` output
must contain exactly one record with no repeated keys. A parent-wide query
must never be flattened into one allocation. Existing source pinning remains
mandatory; this correction alone does not migrate an old execution spec.
Explicit source-pinned recovery is a separate
[execution-repair contract](CMS2JC2_PROXY_EXECUTION_RECOVERY.md), with distinct
attempt/shard/manifest kinds. Original v1 artifacts keep their existing meaning.

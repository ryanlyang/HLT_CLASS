# HCWDL Unified-Balanced Full-Data Coarse Contracts

The authoritative science is frozen in the
[coarse three-arm plan](../plans/HCWDL_UNIFIED_BALANCED_FULL_DATA_COARSE_THREE_ARM_PLAN.md).
This document records reusable versioned identities.

The v1 family is:

- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_GRAPH/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_FOUNDATION_REUSE_LOCK/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_RECIPE/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_SPEC/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_COMMAND_PLAN/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_SWEEP/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_TRAINING_REPORT/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_RUNTIME/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_AGGREGATE/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_ARM_COMPLETE/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_RECOVERY_SPEC/v1`;
- `HCWDL_UNIFIED_BALANCED_FULL_COARSE_RECOVERY_COMMAND_PLAN/v1`.

The graph contains exactly 36 fresh fits and no M1. Coordinates are stored as
integer numerator/denominator pairs plus exact float hex encodings. A semantic
change to rung coordinates, edge routing, loss weights, temperatures, seeds,
population, schedule, checkpoint selection, views, or endpoints requires a
new contract version.

The foundation reuse lock is an immutable authorization overlay. It never
rewrites an imported FULL3 artifact. It proves that the model/view/training
core is byte-identical, binds every imported hash, and names every node that
directly consumes U000 logits: `U033`/`J017` in all three arms plus
`U067`/`J033` as grandparent consumers in `C10P75G15`. Path existence is not
reuse evidence.

Each arm has a separate immutable spec, command plan, submission ledger,
monitor report, reports, and completion record. Recovery is limited to the
authenticated failed/downstream closure with identical semantic source and
nondecreasing resources. Completed outputs are preserved. Scheduler mutation,
exact-ID cancellation, and recovery each require their exact authorization
phrase. Every contract records `final_test_accessed: false`.

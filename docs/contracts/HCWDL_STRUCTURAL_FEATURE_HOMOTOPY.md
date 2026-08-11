# HCWDL Structural-Feature Homotopy Contracts

This is the operational contract index for the validation-only structural-
feature homotopy defined by
`docs/plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md`. The plan
is the scientific authority. These contracts make its artifacts immutable and
fail closed; they do not redefine it.

## Scope and data boundary

- Smoke uses exactly 4,096 train and 4,096 validation jets from an
  authenticated HCWDL smoke.
- Pilot uses exactly 300,000 train and 100,000 validation jets from an
  authenticated unweighted HCWDL pilot.
- Both modes register zero final-test rows and no final-test task.
- Offline information appears only in authorized P0/U/J/D views and the
  frozen TOFF oracle/target cache. Every deployable endpoint is exact HLT.
- Labels are forbidden from residual coupling and switch calibration.

## Immutable artifact families

`hcwdl_homotopy_contracts.py` freezes these v1 families:

- residual-shell coupling config, calibrations, base shards/manifests, switch
  sidecars, role manifests, full-role audit, and coupling lock;
- shared FP32 TOFF target shards, manifest, and lock;
- coordinate table, 80-node graph, recipe overlay, endpoint equality lock,
  graph/recipe lock, campaign spec, and command plan;
- node wrapper and runtime reports, validation aggregate, resource profile,
  and validation-only campaign completion;
- distinct bounded-cache, TOFF-target, and completed-smoke resource
  measurements, so incompatible evidence payloads cannot share a contract
  identity;
- source-pinned and resource-only recovery specs and command plans.

The exact reusable contract identities are:

```text
HCWDL_RESIDUAL_SHELL_COUPLING_CONFIG/v1
HCWDL_RESIDUAL_SHELL_SCALE_CALIBRATION/v1
HCWDL_RESIDUAL_SHELL_SWITCH_CALIBRATION/v1
HCWDL_RESIDUAL_SHELL_BASE_SHARD/v1
HCWDL_RESIDUAL_SHELL_BASE_MANIFEST/v1
HCWDL_RESIDUAL_SHELL_SWITCH_SIDECAR/v1
HCWDL_RESIDUAL_SHELL_COUPLING_MANIFEST/v1
HCWDL_RESIDUAL_SHELL_COUPLING_AUDIT/v1
HCWDL_RESIDUAL_SHELL_COUPLING_LOCK/v1
HCWDL_TOFF_TARGET_SHARD/v1
HCWDL_TOFF_TARGET_MANIFEST/v1
HCWDL_TOFF_TARGET_LOCK/v1
HCWDL_STRUCTURAL_FEATURE_COORDINATE/v1
HCWDL_STRUCTURAL_FEATURE_NODE_SPEC/v1
HCWDL_STRUCTURAL_FEATURE_GRAPH/v1
HCWDL_STRUCTURAL_FEATURE_RECIPE/v1
HCWDL_STRUCTURAL_FEATURE_ENDPOINT_EQUALITY_LOCK/v1
HCWDL_STRUCTURAL_FEATURE_GRAPH_RECIPE_LOCK/v1
HCWDL_STRUCTURAL_FEATURE_PILOT_SPEC/v1
HCWDL_STRUCTURAL_FEATURE_COMMAND_PLAN/v1
HCWDL_STRUCTURAL_FEATURE_TRAINING_REPORT/v1
HCWDL_STRUCTURAL_FEATURE_NODE_RUNTIME/v1
HCWDL_STRUCTURAL_FEATURE_AGGREGATE/v1
HCWDL_STRUCTURAL_FEATURE_RESOURCE_PROFILE/v1
HCWDL_STRUCTURAL_FEATURE_CACHE_RESOURCE_MEASUREMENT/v1
HCWDL_STRUCTURAL_FEATURE_CACHE_MINIATURE/v1
HCWDL_STRUCTURAL_FEATURE_TARGET_RESOURCE_MEASUREMENT/v1
HCWDL_STRUCTURAL_FEATURE_SMOKE_RESOURCE_MEASUREMENT/v1
HCWDL_STRUCTURAL_FEATURE_WEAVER_PARITY/v1
HCWDL_STRUCTURAL_FEATURE_CAMPAIGN_COMPLETE/v1
HCWDL_STRUCTURAL_FEATURE_SMOKE_SELECTION/v1
HCWDL_STRUCTURAL_FEATURE_SELECTION/v1
HCWDL_STRUCTURAL_FEATURE_RECOVERY_SPEC/v1
HCWDL_STRUCTURAL_FEATURE_RECOVERY_COMMAND_PLAN/v1
HCWDL_STRUCTURAL_FEATURE_RESOURCE_RECOVERY_SPEC/v1
HCWDL_STRUCTURAL_FEATURE_RESOURCE_RECOVERY_COMMAND_PLAN/v1
```

Every reusable JSON artifact carries a contract, schema version, canonical
content hash, exact parents, and `final_test_accessed: false`. Array artifacts
also bind immutable NPZ bytes and logical array hashes. Path existence alone
never authorizes reuse.

## Coupling and endpoint authorization

The coupling lock requires complete selected train/validation coverage, exact
row conservation, zero invariant counters, deterministic independent
ROOT/assignment recomputation, solver-optimum rechecks, and exact P0/D100/HLT
endpoint audits. The endpoint lock additionally binds the coordinate table,
cache miniature, projection implementation, and Shell Exact semantics.
`V(1,0)` is byte-exact D100; `V(1,1)` and factorized D0 are byte-exact HLT.

The graph/recipe lock binds both endpoint and TOFF-target locks, the exact
80-fit graph, parent unweighted recipe, per-node overlay, coordinate table,
command plan, and source commit. Training cannot start without it.

## Training and reporting

Every pilot fit is cold-started, runs 60 passes, validates every pass, uses
macro-AUC/CE/log-rejection/earliest-update checkpoint selection, and obtains
its explicit unweighted CE/KD loss from the graph overlay. Smoke uses the same
production worker for two bounded updates. Each fit publishes the generic
PMARD report, an HCWDL-UJ lineage wrapper, selected/final checkpoints, and an
immutable runtime report with measured GPU-hours and peak memory.

The aggregate contains all 80 fits, imported and frozen contextual controls,
transition diagnostics, recovery, teacher retention, specified comparisons,
and total measured GPU-hours. Poor finite scientific performance completes.

## Recovery and runtime acceptance

Recovery consumes an immutable ledger and monitor and submits only the exact
failed/downstream closure. Source recovery changes only the clean pinned
worktree/commit. Resource-only recovery requires OOM/timeout evidence,
identical source/science/output, and monotonic CPU, memory, or walltime
increases. Cancellation uses only exact campaign-bound job IDs.

Local tests and dry runs establish implementation readiness, not Tigris
acceptance. A genuine production-worker smoke must measure coupling, caches,
targets, training, aggregation, storage, preemption, and recovery. Its signed
resource profile is mandatory before creating the 300k pilot.

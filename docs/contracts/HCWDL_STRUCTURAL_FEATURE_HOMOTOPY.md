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

`hcwdl_homotopy_contracts.py` preserves the v1 coupling/runtime families and
freezes new v2 identities for every artifact whose graph meaning changed:

- residual-shell coupling config, calibrations, base shards/manifests, switch
  sidecars, role manifests, full-role audit, and coupling lock;
- shared FP32 TOFF target shards, manifest, and lock;
- v2 coordinate table, 45-node graph, recipe overlay, graph/recipe lock,
  campaign spec, and command plan, with the unchanged endpoint-equality claim;
- node wrapper and runtime reports, validation aggregate, resource profile,
  and validation-only campaign completion;
- distinct bounded-cache, TOFF-target, completed-smoke resource, and exact
  USR1/resume
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
HCWDL_STRUCTURAL_FEATURE_COORDINATE/v2
HCWDL_STRUCTURAL_FEATURE_NODE_SPEC/v2
HCWDL_STRUCTURAL_FEATURE_GRAPH/v2
HCWDL_STRUCTURAL_FEATURE_RECIPE/v2
HCWDL_STRUCTURAL_FEATURE_ENDPOINT_EQUALITY_LOCK/v1
HCWDL_STRUCTURAL_FEATURE_GRAPH_RECIPE_LOCK/v2
HCWDL_STRUCTURAL_FEATURE_PILOT_SPEC/v2
HCWDL_STRUCTURAL_FEATURE_COMMAND_PLAN/v2
HCWDL_STRUCTURAL_FEATURE_TRAINING_REPORT/v1
HCWDL_STRUCTURAL_FEATURE_NODE_RUNTIME/v1
HCWDL_STRUCTURAL_FEATURE_AGGREGATE/v2
HCWDL_STRUCTURAL_FEATURE_RESOURCE_PROFILE/v1
HCWDL_STRUCTURAL_FEATURE_OPERATIONAL_EVIDENCE_WAIVER/v1
HCWDL_STRUCTURAL_FEATURE_CACHE_RESOURCE_MEASUREMENT/v1
HCWDL_STRUCTURAL_FEATURE_CACHE_MINIATURE/v1
HCWDL_STRUCTURAL_FEATURE_TARGET_RESOURCE_MEASUREMENT/v1
HCWDL_STRUCTURAL_FEATURE_SMOKE_RESOURCE_MEASUREMENT/v1
HCWDL_STRUCTURAL_FEATURE_RESUME_EVIDENCE/v1
HCWDL_STRUCTURAL_FEATURE_WEAVER_PARITY/v1
HCWDL_STRUCTURAL_FEATURE_CAMPAIGN_COMPLETE/v2
HCWDL_STRUCTURAL_FEATURE_SMOKE_SELECTION/v1
HCWDL_STRUCTURAL_FEATURE_SELECTION/v1
HCWDL_STRUCTURAL_FEATURE_RECOVERY_SPEC/v2
HCWDL_STRUCTURAL_FEATURE_RECOVERY_COMMAND_PLAN/v2
HCWDL_STRUCTURAL_FEATURE_RESOURCE_RECOVERY_SPEC/v2
HCWDL_STRUCTURAL_FEATURE_RESOURCE_RECOVERY_COMMAND_PLAN/v2
HCWDL_STRUCTURAL_FEATURE_EXECUTION_RESOURCE_RECOVERY_SPEC/v1
HCWDL_STRUCTURAL_FEATURE_EXECUTION_RESOURCE_RECOVERY_COMMAND_PLAN/v1
```

Every reusable JSON artifact carries a contract, schema version, canonical
content hash, exact parents, and `final_test_accessed: false`. Array artifacts
also bind immutable NPZ bytes and logical array hashes. Path existence alone
never authorizes reuse.

The exhaustive coupling audit may execute one process task per authenticated
train or validation source unit. Its reducers are exact integers, while
endpoint and solver hashes are framed per source and combined in canonical
train-then-validation source order. Completion order and worker count are
operational details, not scientific identity; the audit records this reduction
scheme and validators fail closed on an unknown scheme. Parallel execution
does not weaken the every-row endpoint, conservation, or solver-optimum checks.

For the graph-thinned v2 pilot, an explicit operational-evidence waiver may
carry the completed 80-fit v1 production-worker smoke together with completed
v2 parity, coupling, endpoint, target, and graph-lock evidence. It records the
human decision and exact fixed resource requests. It is not a v2 smoke
completion and cannot change the scientific graph, loss, data, or test
boundary.

## Coupling and endpoint authorization

The coupling lock requires complete selected train/validation coverage, exact
row conservation, zero invariant counters, deterministic independent
ROOT/assignment recomputation, solver-optimum rechecks, and exact P0/D100/HLT
endpoint audits. The endpoint lock additionally binds the coordinate table,
cache miniature, projection implementation, and Shell Exact semantics.
`V(1,0)` is byte-exact D100; `V(1,1)` and factorized D0 are byte-exact HLT.

The graph/recipe lock binds both endpoint and TOFF-target locks, the exact
45-fit v2 graph, parent unweighted recipe, per-node overlay, coordinate table,
command plan, source commit, and installed-Weaver CUDA parity for both the
unified and native-offline factories. Training cannot start without it.

## Training and reporting

Every pilot fit is cold-started, runs 60 passes, validates every pass, uses
macro-AUC/CE/log-rejection/earliest-update checkpoint selection, and obtains
its explicit unweighted CE/KD loss from the graph overlay. Smoke uses the same
production worker for two bounded updates. Each fit publishes the generic
PMARD report, an HCWDL-UJ lineage wrapper, selected/final checkpoints, and an
immutable runtime report with measured GPU-hours and peak memory.
The initial pilot `gpu_training` request is exactly 8 CPUs, 96 GiB, six hours,
and one GH200; campaign creation rejects a profile with a different row.

The aggregate contains all 45 fits, imported and frozen contextual controls,
transition diagnostics, recovery, teacher retention, specified comparisons,
and total measured GPU-hours. Poor finite scientific performance completes.

## Recovery and runtime acceptance

Recovery consumes an immutable ledger and monitor and submits only the exact
failed/downstream closure. Source recovery changes only the clean pinned
worktree/commit. It retains the campaign's original scientific-source map in
the scientific identity while separately binding the complete corrected
execution-source map, every changed semantic-file hash pair, and the explicit
`execution_only_human_authorized_v1` classification. Source-recovery workers
must match that corrected map exactly; they may not pretend the corrected
bytes equal the original campaign bytes. Resource-only recovery requires
OOM/timeout evidence, identical source/science/output, and monotonic CPU,
memory, or walltime increases. Cancellation uses only exact campaign-bound
job IDs.

The separately versioned execution-and-resource recovery is used only when a
single failed closure needs both a reviewed execution-only source correction
and a monotonic resource increase. It binds the same original scientific
identity, complete old/new semantic-source maps, exact prior ledger and
monitor, old and replacement resource envelopes, and distinct human
authorization/submission phrases. It never permits a GPU-type change, a
resource decrease, a graph/data/loss change, or a task outside the exact
failed/downstream closure. This combined contract does not broaden either
legacy v2 recovery identity.

Local tests and dry runs establish implementation readiness, not Tigris
acceptance. A genuine production-worker smoke must measure coupling, caches,
targets, training, aggregation, storage, preemption, and recovery. Its signed
resource profile is mandatory before creating the 300k pilot. The profile
requires at least 25% measured headroom for walltime, process RAM, GH200 GPU
memory, and an explicitly declared durable campaign-root storage budget. It
cannot be published without an immutable `SIGUSR1` event whose rolling-state
hash is the exact checkpoint named by a later completed engine report's resume
provenance; the interrupted and resumed Slurm job IDs must both exist and must
be distinct.

# HCWDL Unified-Root Balanced Homotopy Contract

## Status and scope

This contract implements the validation-first 300k campaign defined by
[`HCWDL_UNIFIED_BALANCED_HOMOTOPY_IMPLEMENTATION_PLAN.md`](../plans/HCWDL_UNIFIED_BALANCED_HOMOTOPY_IMPLEMENTATION_PLAN.md).
It is additive. It does not mutate, relabel, or resume an HCWDL-UJ campaign.

The population is exactly 300,000 training jets, 100,000 validation jets, and
100,000 sealed final-test jets imported from the authenticated unweighted
HCWDL parent. Ordinary foundation and arm work may read only train and
validation. Deployable and final-evaluated models consume exact HLT inputs.

## Scientific coordinates

`U000` is a freshly initialized, unweighted-CE unified 21-channel ParT trained
on the exact bounded projected-offline P0 multiset. It is not native TOFF's
two-stream adapter. `M0paired` is the paired unweighted-CE exact-HLT root.

The balanced structural coordinate reuses the immutable residual coupling
base. It publishes a separate sidecar. Each atomic edit is stratified by edit
kind, six-state source and target categories, charged applicability, and
validity-group changes. Within each jet and stratum, positive integer edit
mass is placed on a deterministically rotated circular mass arc. Switches are
stored as exact rounded uint16 coordinates. The same sidecar supports every
rung grid; no switch is redrawn by epoch, worker, batch, resume, or campaign
arm. The endpoints explicitly apply none or all of the edits.

Uniform Shell Exact uses one exact rational offline strength for every matched
slot. Continuous quantities interpolate with that strength. PID, identity,
quality, lost-hit, applicability, and validity groups use immutable
identity/HLT-slot/group hash variates and the exact rational threshold. Match
confidence is retained only as provenance and is forbidden from the new D
coordinate. The legacy confidence-warped behavior remains a paired control in
the reference arm.

Default paths are:

```text
shared/U000
  -> U020 -> U040 -> U060 -> U080 -> U100
  -> D80F -> D60F -> D40F -> D20F -> D0F -> M1F

shared/U000
  -> J010 -> J020 -> J030 -> J040 -> J050
  -> J060 -> J070 -> J080 -> J090 -> J100 -> M1J
```

The arrows are KD edges. Every model is cold-started. U/D/J nodes use
temperature 2. M1 nodes use fixed `0.25 CE + 0.75 parent KD` at temperature 1.
All fits use unweighted per-jet CE, 60 natural-population passes, validation
after every pass, the locked common child learning rate, and macro-AUC-first
checkpoint selection.

## Shared foundation and six arms

One foundation owns exactly two fits, `shared/U000` and `shared/M0paired`, plus
balanced train/validation sidecars, endpoint/resource gates, and an
identity-ordered FP32 U000 target cache. A completed immutable foundation lock
is the only training input shared by the six arms.

The six arm recipes are:

| Arm | CE | parent KD | grandparent KD |
|---|---:|---:|---:|
| `C25P75` | 0.25 | 0.75 | 0.00 |
| `C10P90` | 0.10 | 0.90 | 0.00 |
| `C05P95` | 0.05 | 0.95 | 0.00 |
| `C10P75G15` | 0.10 | 0.75 | 0.15 |
| `C05P80G15` | 0.05 | 0.80 | 0.15 |
| `C00P100` | 0.00 | 1.00 | 0.00 |

When no grandparent exists, its registered weight is transferred to the
parent. Zero-weight terms remain serialized. Each arm has a distinct spec,
root, command plan, submission ledger, monitor, cancellation surface,
recovery closure, aggregate, and completion artifact. Arms have no cross-arm
teachers or scheduler dependencies. `C25P75` alone owns 11 paired legacy
controls. The exact registry is 2 shared fits + 34 reference-arm fits + five
times 23 ordinary-arm fits = 151 fits.

## Persistence and lineage

Durable particle-view datasets are forbidden. Workers reconstruct one
train/validation view in RAM per job using the corrected prepared-endpoint
path, then replay it for all epochs. Durable data are limited to authenticated
assignments, residual bases, switch sidecars, compact FP32 target caches where
multiple consumers justify them, checkpoints, reports, locks, task
attestations, monitors, and ledgers.

Every reusable artifact binds source/split/row-selection identities, exact
parent hashes, producer commit, and logical and byte hashes as appropriate.
Foundation-spec v2 and operational-waiver v2 bind the completed U/J
fixed-preprocessing prefix through its graph/recipe lock, endpoint-equality
lock, coupling lock, and manifests. They deliberately do not require the
unrelated U/J training graph or validation aggregate to finish before the
new campaign can run in parallel.
The foundation lock binds both shared engine-report hashes, both selected
checkpoint hashes, the U000 target manifest, and the separately validated
target lock. Every arm validator reauthenticates that foundation lock and its
exact arm-recipe artifact before a dry run or live submission is accepted.
The operational-evidence waiver truthfully records that no new HCWDL-UB smoke
was run; it binds the prior production-worker completion and Weaver parity,
the corrected preprocessing guide, the pushed source, semantic source hashes,
readiness evidence, and the locked resource envelope. The frozen semantic
source map covers the model, inputs, training engine, checkpoint/resume logic,
view/target caches, homotopy builders, runner/workflow, reporting, and primary
Slurm worker surface.

The implementation registers the plan's contract family plus the following
mechanical children required to authenticate shared and arm-local compact
targets and the separately authorized final-test execution:

```text
HCWDL_UNIFIED_BALANCED_TARGET_SHARD/v1
HCWDL_UNIFIED_BALANCED_TARGET_MANIFEST/v1
HCWDL_UNIFIED_BALANCED_TARGET_LOCK/v1
HCWDL_UNIFIED_BALANCED_TARGET_DIGEST_SHADOW_EVIDENCE/v1
HCWDL_UNIFIED_BALANCED_EXECUTION_REPAIR_RECOVERY_SPEC/v1
HCWDL_UNIFIED_BALANCED_EXECUTION_LOCK/v1
```

These children add no scientific arm or view semantics.

## Failure, resume, and recovery

Finite poor science completes normally and never prunes a row or descendant.
Missing/corrupt lineage, forbidden role access, nonfinite required values, or
source drift fails closed. GPU jobs receive `USR1` 120 seconds before their
limit and use exact rolling checkpoints.

Recovery begins from one immutable scope spec, submission ledger, and monitor.
It schedules exactly the failed/downstream closure. A source recovery may
change execution code only while all frozen semantic source hashes remain
identical. A resource recovery may only increase CPUs, RAM, or walltime and
may not change the GPU class. Completed outputs are preserved. Cancellation
uses only exact IDs from one bound ledger.

One separately versioned execution-repair recovery is permitted for the
historical U000 target-manifest digest-shadow defect. The affected validator
authenticated the manifest correctly but returned its final parent digest
(the U000 report hash) instead of the manifest content hash. Consequently the
already-published target and foundation locks recorded that report hash in
their manifest fields. The repair does not rewrite or relabel either lock. It
must instead bind the exact independently authenticated manifest, report,
checkpoint, target lock, and foundation lock in
`HCWDL_UNIFIED_BALANCED_TARGET_DIGEST_SHADOW_EVIDENCE/v1`; change exactly
`hcwdl_unified_balanced_targets.py` and
`hcwdl_unified_balanced_runner.py` in the frozen semantic-source map; use the
explicit repair authorization phrase; and run only through
`HCWDL_UNIFIED_BALANCED_EXECUTION_REPAIR_RECOVERY_SPEC/v1`. Ordinary campaign,
source-recovery, and resource-recovery paths continue to reject the legacy
lock mismatch. Any other digest or lineage mismatch fails closed. The repair
also preflights shared U000 lineage before building an arm's RAM views.

## Reporting and final-test boundary

Every node reports CE, parent/grandparent KD, zero-weight terms, teacher
agreement, the idealized U000 ancestry, full validation history, selected
checkpoint, and runtime. The read-only sweep aggregate ranks all six arms by
the predeclared macro-AUC-first comparison.

Only `D0F`, `J100`, `M1F`, and `M1J` from the top two validation arms become
finalists. A finalist lock and a separate human-authorized execution lock must
both exist before test rows are opened. The one sealed evaluation reads exact
HLT only and evaluates those eight models plus `M0paired`. Test metrics never
select a model.

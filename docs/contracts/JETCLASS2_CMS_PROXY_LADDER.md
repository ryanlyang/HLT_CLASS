# JETCLASS2_CMS_PROXY_LADDER contracts

The `JETCLASS2_CMS_PROXY_LADDER_*/v1` family defines the 200k-train,
100k-validation CMS-calibrated proxy-HLT benchmark described by
`docs/plans/JETCLASS2_CMS_PROXY_TIGRIS_200K_THREE_SPINE_IMPLEMENTATION_PLAN.md`.

All JSON artifacts are canonical-content-hashed, immutable after publication,
and carry `final_test_accessed: false`.  Parent hashes are mandatory.  File
references carry an exact byte count and SHA-256 digest.

## Release request and release

`RELEASE_REQUEST/v1` binds the proxy study root, exact offline ROOT root,
ordinary-role counts, and the label-blind selection domain.

`RELEASE/v1` binds:

- the authenticated proxy study;
- the exact committed train/validation receipt snapshot;
- exact proxy NPZ block hashes and offline ROOT file hashes;
- 200,000 train and 100,000 validation row identities;
- an immutable compact locator bank.

The locator bank contains only identity and source-location metadata.  It may
not contain classifier inputs, labels, final-test rows, or materialized views.
If the full proxy manifest does not yet exist, the release declares itself a
`committed_ordinary_receipt_snapshot`; later receipts cannot silently change
that population.

## Views, inputs, and foundation

`VIEWS/v1` freezes the persistent proxy-HLT skeleton, offline tail, transferred
`SALIENCE_PT_LINEAR` full-cardinality matcher, U/D interpolation, categorical
switches, and validity-aware tracking semantics.  `Dxxx` names the retained
offline-feature fraction (`D066` is two-thirds offline/one-third proxy and
`D000` is exact proxy-HLT).

`INPUTS/v1` freezes the 17 Particle Transformer inputs.  Unknown particle
category produces five zero PID flags.  Invalid tracking values/errors produce
zeros and never an invented significance.  Capacity overflow fails; truncation
is forbidden.

`FOUNDATION/v1` binds the release, matcher, views, inputs, identity digests, and
one compact proxy-slot-to-native-offline-index assignment bank.  It records
observed maximum proxy/offline/view cardinalities and endpoint audit evidence.
Full materialized views are forbidden.

## Gate

`SOURCE/v1` binds the exact pushed commit and runtime-semantic file hashes.
`GATE_SPEC/v1` defines three ordered tasks:

1. authenticate and freeze the ordinary release;
2. compute/audit the new endpoint-specific assignment foundation;
3. perform a genuine GH200 full-selected-population preflight.

`RUNTIME_PROFILE/v1` is eligible only when it records the exact Tigris site,
installed Weaver/numerical environment, full-population RAM caches, a real
one-pass fit, reducer inference, GPU peak, and measured walltime envelopes.

`GATE_SPEC/v2` is the explicit SPORC recovery gate. It imports the immutable
`RELEASE/v1` from a completed `GATE_SPEC/v1`, then binds SPORC-debug foundation
construction and preflight resources. `RUNTIME_PROFILE/v2` records an A100
measurement on `sporc_a100_debug` and the sole authorized transfer to
`sporc_a100` tier3 under
`sporc_debug_to_tier3_same_a100_environment_resources_v1`. Debug is never the
scientific execution site.

`GATE_SPEC/v3` is a preflight-only recovery. It imports and validates the exact
completed `FOUNDATION/v1` and `RELEASE/v1` from its v2 parent gate, binds the
measured 16-worker cache bounds, and requests exactly 16 CPUs, 160000 MiB, one
A100, and eight hours on SPORC debug. `RUNTIME_PROFILE/v3` carries the same
explicit debug-to-tier3 transfer and makes those right-sized resources the
only eligible science allocation.

Oscar `GATE_SPEC/v4`/`RUNTIME_PROFILE/v4` and the dual-slot operational
replacement `GATE_SPEC/v5`/`RUNTIME_PROFILE/v5` are defined by
`JETCLASS2_CMS_PROXY_OSCAR_PORTABILITY.md`. Both consume the same exact
portable release and foundation; v5 changes only measured CPU-worker and
memory allocation.

`GATE_COMPLETE/v1` authenticates the gate, foundation, and runtime profile.  It
does not itself authorize scientific submission.

## Science campaign

`SCIENTIFIC_PLAN/v1` contains exactly 17 fresh fits:

- `M0HLT`, `OFFLINE`, and `U000` controls;
- 1 DIRECT student;
- 5 COARSE students;
- 8 DENSE students.

It contains exactly 12 probability publications.  Recovery is defined with
`M0HLT` as zero and pure `OFFLINE` as one hundred percent; `U000` remains a
reported control, not the recovery oracle.

### Oscar OFFLINE ECC recovery

`JETCLASS2_CMS_PROXY_LADDER_CAMPAIGN_ECC_RECOVERY/v1` is the registered
recovery for verified uncorrectable-ECC roots in `CAMPAIGN_SPEC/v2`. It derives
the retry set as the exact registered downstream closure, reuses authenticated
successful task outputs, and excludes every evidenced failed node from all
replacement GPU jobs. Reused parent job IDs remain recorded as immutable
provenance in the recovery specification. They are not emitted as Slurm
dependencies because their outputs are authenticated before the recovery is
created and historical jobs may age out of the Slurm controller.
`JETCLASS2_CMS_PROXY_LADDER_CAMPAIGN_ECC_RECOVERY_COMMAND_PLAN/v2` emits exact
`afterok` dependencies only between jobs in the replacement closure. The
recovery fails closed for non-ECC roots, unrelated terminal tasks, missing
success artifacts, an unauthenticated reused parent, or a retry set that
differs from the graph closure.

`JETCLASS2_CMS_PROXY_LADDER_CAMPAIGN_ECC_RECOVERY_SUBMISSION_REPAIR/v1`
authenticates an interrupted legacy v1 submission whose immutable journal is a
valid prefix of the corrected plan. It binds the legacy recovery, v1 command
plan, canonical dry-run ledger, every journal-event byte, exact imported job
IDs and their accounting snapshot, and a fresh source lock. Its corresponding
`CAMPAIGN_ECC_RECOVERY_SUBMISSION_REPAIR_COMMAND_PLAN/v1` removes only
historical completed-parent Slurm dependencies. It imports, but never
resubmits, the accepted prefix. Imported jobs are also omitted from new Slurm
dependencies because they can age out before continuation; registered
descendant workers retain the scientific dependency and fail closed unless the
imported task's immutable completed output authenticates. Separate command-plan,
dry-run, live-ledger and journal filenames preserve all legacy evidence without
overwrite.

`CAMPAIGN_SPEC/v1` binds the completed gate, foundation, runtime profile,
scientific plan, task graph, model factory, exact source, and separate fresh
campaign root.

`TASK_REPORT/v1` is an immutable task completion pointer whose output inventory
is checked byte-for-byte.  Scientific metric quality never controls task
success.  Invalid lineage, nonfinite required values, missing parents, stale
source, capacity overflow, forbidden data access, and corrupt artifacts fail
closed.

`COMMAND_PLAN/v1` and the exact submission ledger bind every Slurm command and
dependency by exact task/job identity.  Live gate and science submission use
different explicit authorization phrases.

## Nested 100k/50k DIRECT+COARSE variant

`POPULATION_SELECTION/v1` derives exactly 100,000 train and 50,000 validation
rows from the authenticated 200k/100k foundation. It ranks canonical row
identities using SHA-256 over a versioned domain, parent foundation hash, role,
and identity, selects the lowest ranks, then preserves foundation order. It
records exact ordered identity digests. Labels, particle features, and final
test are not read during selection; the existing assignment bank is reused
without recomputation.

Oscar `GATE_SPEC/v6` and `RUNTIME_PROFILE/v6` bind that population and are
defined operationally by `JETCLASS2_CMS_PROXY_OSCAR_PORTABILITY.md`. A genuine
full-selected-population v6 preflight is required even though the immutable
portable materialization and full-cardinality matches are reused.

`SCIENTIFIC_PLAN/v2` and `CAMPAIGN_SPEC/v2` register only:

- `M0HLT`, pure `OFFLINE`, and `U000` controls;
- one DIRECT student;
- five COARSE students.

They contain exactly 9 fresh fits, 5 probability reducers, and 16 total Slurm
tasks including aggregate and completion. DENSE is absent from the registered
plan and submission ledger. Recovery remains `M0HLT` to pure `OFFLINE`.

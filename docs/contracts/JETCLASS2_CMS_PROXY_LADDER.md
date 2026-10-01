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

# CONTEXT_V2: one frozen stronger recipe and automatic Oscar follow-up

## Authority and scope

The user requests a small strengthening, no additional tuning pilots, full
2.25M generation on Oscar, then the same 100k/50k DIRECT + COARSE comparison
without another manual submission between stages. This plan supersedes the
earlier proposal for a separate diagnostic/strength-selection pilot only.
Existing CONTEXT_V1 data, source checkouts, jobs and results remain immutable.
The desired 1.5--2 percentage-point offline/HLT accuracy gap is a motivation,
not a guaranteed outcome or an acceptance threshold. Do not tune to ladder wins.

## Frozen single change

CONTEXT_V2 retains three triangular coupling layers and changes the shear
amplitude from 0.65 to **1.0**. Phases, visible-p4 context, error-map coefficients,
input encoding, missing-track semantics and all stochastic mechanisms remain
unchanged. In particular no new noise draws, drops, merges, PID conversions,
p4 changes, class-dependent decisions or count calibration are introduced.

Build from the authenticated, lossless CONTEXT_V1 banks already on Oscar:
apply the frozen V1 inverse, then the V2 forward map. This is a re-encoding of
the same low-noise endpoint, not a second degradation. Floating-point inverse
roundoff is recorded and bounded; it is not advertised as bitwise recovery of
the historical pre-map tracking values. P4, counts, charge, PID, masks and row
identities must remain byte-identical. Complete tracking only is transformed.
New error columns remain synthetic encoded scales, not calibrated uncertainties.
Classifier inputs never receive the inverse, context keys or construction data.

## Population and test boundary

Exactly 1M train / 250k validation / 1M final test, with existing membership,
physical bank ordering and canonical inventory/file/tree/raw-entry identities.
Test processing is explicitly authorized **materialization only**. The ordinary
reader and classifier reject test requests before reading test receipts/banks.
No test histograms, labels, model predictions or recipe selection are permitted.

The new classifier release copies the authenticated V1 100k/50k release index,
rebinds its proxy block checksums and verifies its canonical joins. It must not
re-rank with a new study hash: that would silently choose different jets.
Recompute particle assignments on V2, without generator ancestry. Preserve the
same unclipped 17-input asinh/log1p interface, model, seeds, optimizer, pass
budgets, C25/P75 T2 loss and validation selection. Nine fresh fits, five
probability reducers, aggregate and completion (16 jobs), no dense branch.
M0HLT, OFFLINE and U000 are fresh controls; recovery uses M0HLT=0/OFFLINE=100.

## One reviewed workflow, no tuning pilot

Initial static DAG: generation array -> finalizer -> training preparation ->
GPU preflight -> authorized science launcher. The launcher authenticates the
completed gate, creates the exact measured 16-job science plan, verifies it
against the preauthorized policy, writes a dry ledger, and submits it once.
All initial commands and the bounded downstream policy are shown before the
user authorizes the workflow. No metric or requested accuracy gap gates jobs.

There is no separate CPU pilot or CE strength screen. Each production shard
checks serial/process replay of its first block and reversible-map integrity
inside its real generation job, with lossless write/readback on every block.
The ordinary GPU preflight retains installed-Weaver parity, CE/KD execution
and full-selected-population resource measurement. This is runtime acceptance,
not another scientific pilot; no acceptance model initializes a science fit.
This Oscar-native acceptance replaces the generic Tigris requirement here.

Generation: CPU-only Oscar batch, default account, 6 CPUs, 32,000 MiB, 2h, at most
8 array elements. Finalizer: 1 CPU/16,000 MiB/2h. Preparation: 6 CPUs/90,000 MiB/8h.
Preflight: gpu/norm-gpu, one L40S, 6 CPUs/90,000 MiB/12h. Launcher:
batch, 1 CPU/8,000 MiB/2h. Science uses the existing measured Oscar shape, bounded
by 48h per fit and 24h per reducer; 4 CPUs/32,000 MiB/1h per reporting task.
Scheduler feasibility is checked before live submission. No GPU is requested
by generation/preparation/launcher. No claim is made about current queue speed.

## Persistence, provenance and recovery

Require a fresh persistent dataset root and separate workflow root, neither
overlapping sources/checkouts. Default data budget 20 GiB including 2 GiB headroom;
operator must attest available quota for data plus training/cache outputs.
Bounded per-shard reservations include partial output; no automatic deletion.
Publish content-hashed receipts last, after physical validation. Finalize only
after all exact shards pass and counts match. An incomplete dataset cannot
start matching or training. Source, input manifest, original release, fixed
recipe and scientific policy are all hash-bound.

Live submission has one durable exclusive claim and exact-ID journal. Workers
authenticate their own submission and never accept an unregistered array task.
Partial/ambiguous submissions fail closed; preserve journals for explicit
recovery. Completed initial submission ledgers can be inspected idempotently;
launcher reruns stop if science already exists and require explicit recovery.
Never duplicate a partially submitted graph. No cancellation or existing-job
mutation is authorized by this workflow.

## Acceptance

Local tests: recipe isolation, invertibility/numerical limits, serial/process
parity, role-preserving all-role synthetic generation, corruption and storage
failures, exact reused selection, end-to-end fresh foundation/cache, true CPU
training-kernel smoke where practical, versioned dispatch and old regressions,
full workflow dependencies, authorization/source/job-binding failures and
duplicate/ambiguous submission behavior. Genuine Oscar data/GPU acceptance
occurs in the queued workers; local fixtures cannot establish performance.

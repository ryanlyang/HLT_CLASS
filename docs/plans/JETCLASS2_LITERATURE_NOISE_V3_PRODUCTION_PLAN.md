# Frozen NOISE_V3: full JetClass2 synthetic dataset

## Scientific decision and scope

The user approved generating 2,250,000 paired jets after Tigris pilot 218339
completed on 2026-10-03. The authoritative recipe is the committed
`literature_proxy_v3.kernel.recipe()` and the **full-precision calibration
artifact** inherited by the completed v3 pilot, not rounded console rates.
The pilot used 20,000 training jets: 40.9229 offline particles/jet and 38.0021
synthetic particles/jet, exact process replay, and exactly unchanged v2
particle topology. This is a controlled randomized benchmark, **not CMS HLT**
and not evidence of improved classification or distillation.

This production plan supersedes the pilot's training-only execution scope,
not its physics. No fitting, optimization, new random seed, replica, CMS
response model, or validation/test calibration is permitted. In particular,
the output mean is allowed to differ from 38 on other populations.

## Population and pairing

Use the same authenticated dzfix offline snapshot and split registry as the
pilot. Require the existing `TRAIN_1M` profile with the same registry and
unchanged outer validation/test file memberships. Produce exactly 1,000,000
training, 250,000 validation, and 1,000,000 final-test jets. Reuse the existing
CMS-proxy **metadata-only** population builder, including its label-blind
validation hash-subset domain, so these memberships match the previous
2.25M dataset. Do not reuse that dataset's physical outputs or learned mapping.

Canonical jet identity is the inventory hash, relative source file, ROOT tree
key/cycle, and original entry. Original particle keys are exactly
`part:{index:08d}` as in the literature pilot. Raw offline columns use GeV and
native Delphes mm; do not apply the CMS feature-interface conversion. Read
only `jet_nparticles` and the fourteen offline particle columns used by the
pilot. No labels or native-HLT particle branches are read. Native-HLT
eligibility already present in the split metadata remains inherited.

The pilot's scientific implementation file hashes must match the production
source; a recipe dictionary alone is not sufficient evidence of unchanged code.
The same frozen identity-keyed streams are used regardless of role, shard,
chunk, process count, or retry. Pair by canonical identity/original entry,
never by compressed row number or particle index across offline and proxy.
Physical banks contain offsets, jet identities and physical particle fields;
construction ancestry and particle keys are not model inputs and are not
exported into the banks.

## Single submission, bounded execution

Submit the complete DAG once: scheduled CPU preflight -> generation array ->
finalizer. No second human submission is required after successful preflight.
Tigris account `reu-aisocial`, partition `tigris`, no GPUs. Preflight and each
generation element request 36 CPUs, 64 GiB, two hours. At most sixteen
generation elements run simultaneously (576 allocated CPUs). Finalizer uses
one CPU, 16 GiB, two hours. File-local shards contain at most 25,000 jets;
offline windows are at most 512 entries; output blocks at most 1,000 jets.
Process work is bounded and ordered, with native numerical threads limited
to one per process. Queue wait is not included in runtime estimates.

The preflight authenticates the completed pilot chain, requires matching
Python/architecture/NumPy, replays up to 64 saved pilot training jets through
the production raw reader, serial/process generator and physical writer,
and demands exact saved-v3 physical parity. It records wall time/RSS/bytes
and checks conservative production time, memory and storage projections.
Only operational validity can fail the gate, not classifier performance.
The array depends on preflight success. The finalizer uses `afterany` and
will report missing receipts without publishing a complete manifest.

## Storage, provenance and recovery

The output root must be fresh, persistent, and outside all source datasets,
parent campaigns and execution worktrees. The operator attests currently
available quota and persistence; filesystem free space is also checked.
Default output budget is 20 GiB plus 2 GiB free headroom. Preflight checks
measured output projections with a factor-two margin. Each shard reserves
its bounded share; partial failed attempts consume budget and are retained.
No existing dataset or artifact is overwritten or deleted.

All metadata is versioned and content-addressed, all physical blocks have
byte hashes, and publication is atomic. Source is a clean, exact, pushed
commit. Submission uses an exact reviewed plan hash and authorization,
durable intent/response journals, exact array-element Slurm identities,
and no automatic retry of ambiguous submissions. Recovery at the same
commit creates a new attempt for only missing authenticated receipts, after
all prior jobs are demonstrably terminal; no overlapping writers or broad
cancellation. Source-changing repair requires a separately reviewed plan.

## Final-test boundary and consumption

The user's authorization includes deterministic final-test **materialization**.
An immutable test-build lock grants only that operation. Test shard receipts
and the full manifest truthfully mark test materialization/access; they never
claim test evaluation. No test distributions, model inference, scores, tuning
or selection are performed. Train/validation can be released independently
once all receipts for that role authenticate, without reading test banks.
The complete manifest requires every shard and the exact 2.25M role totals.
Consumers must explicitly reject final-test reads until their own finalist
and evaluation locks exist; this package's reader exposes train/validation only.

## Verification and handoff

Focused tests cover actual synthetic ROOT ingestion, exact v3 saved-pilot
parity, serial/process/shard replay, population identity, sealed test access,
bank corruption, incomplete publication, quota reservations, exact-plan DAG
submission and array identity/recovery. Local tests are not remote evidence.
The completed v3 pilot is real recipe evidence; the first production job
provides the genuine production-worker miniature before bulk work starts.

# JetClass2 dzfix fixed-slot K=2 concatenation campaign

Scientific authority: [the active implementation plan](../plans/JETCLASS2_DZFIX_FIXED_SLOT_CONCAT_K2_LADDER_PLAN.md).
Family: `JETCLASS2_DELPHES_CONCAT_K2_*/v1`. Historical U/D, full-cardinality
and fusion-chain artifacts retain their old meaning; none is modified.

## Population, source and isolation

Use the exact dzfix `TRAIN_500K` selection from the authenticated debug
salience screen: 500,000 train / 1,000,000 validation / 1,000,000 sealed test.
The source is the screen's selected **salience formula**, inventory and split
membership. Authenticate the eight-task live ledger, completed task receipts,
selection lock and selected foundation. Do not reuse its one-to-one maps,
models, probability banks or GPU acceptance as new K2 evidence.

The contextual bottleneck control is not a salience candidate. The consumer
must never choose a different formula after viewing its own report results.
No job reads final-test particle arrays. Inventory byte authentication is not
test inference. Existing production remains on its registered site; every new
`jc2k2_*` job, including launchers and CPU preparation, requests SPORC `debug`.

New launch/campaign roots must be separate and outside protected source roots.
Source checkout must be clean, exact-commit and known on an origin ref. No
operation updates an existing campaign, its dependencies or its jobs.

## Exact retention and capacity-two objective

For native counts `n=N_HLT>0`, `m=N_offline>0`, allocate `n` original HLT slots
and `2*n` rich destinations. Native offline, not persistent U000, supplies the
rich pool. Compute the selected formula's integer salience on each **complete
native endpoint before either cropping or duplication**.

Keep `min(m,2*n)` offline particles with largest frozen salience, breaking
ties by ascending native offline index. This retention precedes geometry.
Each HLT owner supplies two exchangeable capacity-one destinations with its
same frozen salience and raw features. Solve one global exact integer
assignment (not greedy nearest-neighbour), matching every retained offline
particle exactly once. The ordered objectives are:

1. Maximum total salience-weighted bounded angular utility.
2. Minimum total uncapped quantized delta-R.
3. Maximum total salience of occupied HLT destinations, counting each slot.
4. Minimum total quantized absolute log-pT response.
5. Minimum category mismatches.
6. Minimum valid-charge mismatches.
7. Lexicographically smallest native offline index per ordered destination;
   unmatched is after every real index.

The formula, quantization and exact mixed-radix solver are reused from
`hcwdl_fullcard_salience_contracts` and `hcwdl_fullcard_salience_matcher`;
the new duplicated-destination problem and retention policy have their own
view hash. Destinations are owner order, then rich slot 1/2. The last objective
canonicalizes two interchangeable rich slots. No salience is renormalized
on retained particles or duplicated endpoints.

Compact maps are `int32[n,2]` native offline indices, with `-1` for fillers.
Store row IDs, HLT offsets and native offline counts. Coverage, uniqueness,
canonical ordering, schema, content hashes, parents, source bytes and exact
row joins are checked on load. Exhaustive-reference acceptance is bounded
to expanded sides of at most eight, never production combinatorics.

## Views and deployable endpoints

Interleave each original HLT particle with its two rich destinations. Fill
unoccupied rich slots with active copies of their HLT owner. Thus **every
rung has exactly `3*n` active particles**, including every original HLT.

`D100` contains retained native offline + native HLT + required HLT fillers.
`D075`, `D050`, `D025` use offline fractions 3/4, 1/2, 1/4. Four-vectors and
jointly applicable numeric measurements interpolate linearly. PID/charge are
atomic; charge applicability changes carry their measurements atomically;
uncertainty validity changes switch the value/error pair atomically. Fixed
identity/owner/slot/group hashes give deterministic nested switches. Labels
never enter these decisions. `D000` short-circuits to native HLT repeated
three times without accessing identity, offline particles, maps or salience.

Recompute all 17 model features and pair vectors from the **entire resulting
set** at every rung. No source flag, ownership index, positional/slot embedding,
confidence feature, or extra model channel is supplied. Duplication therefore
also affects jet-normalized features; it is not equivalent to appending old
precomputed HLT features. All fillers have true masks.

Capacity is `ceil16(max(max_native_offline,3*max_native_HLT))`, using the
authenticated inventory; never the previous 320 constant. Native capacity
is recorded separately. The archived 186-file SPORC inventory has maxima
HLT=277/offline=306, implying 832 padded slots. Runtime rederives this value.
No constituent trimming or silent truncation is permitted.

`deployment_inputs(native_hlt, copies=1 or 3, capacity=...)` is standalone.
HLT-only selected models publish hashed deployment manifests; both x3 and
compressed x1 require only native HLT at inference.

## Registered fits and controls

All models are fresh single-encoder 17-input/11-class ParTs, not learned
two-branch fusion. The ordered ladder is:

```text
CONCAT_K2_D100 CE -> D075 KD -> D050 KD -> D025 KD -> D000 KD
                                                      -> HLT_X1_COMPRESSED KD
```

Four additional comparisons: `HLT_X1_CE`, `HLT_X3_CE`, `OFFLINE_CE`,
`DIRECT_HLT_X3_KD` taught directly by D100. Total: ten fits, five teacher
reducers, aggregate and complete = 17 science tasks. No implicit ensemble,
warm start, K=3 run or extra seed sweep is included.

Pair initialization and sampler seeds across all fits using separate
`JC2/CONCAT_K2/v1` domains. Reuse the registered training kernel:
batch 256, AdamW, peak 3e-4, floor 1.5e-5; three-pass warmup, hold through
45, cosine to floor at 60, then constant floor through maximum 100.
Minimum 60, patience clock begins at 60, patience 15, significant AUC delta
5e-5: earliest early stop 75; restore best checkpoint using the existing
AUC/CE/R50/update ordering. CE controls have no KD. Students use C25P75,
T=2 including T-squared KL scaling and exact identity-joined train banks.
Precision is BF16 forward / FP32 loss. No gradient accumulation, altered
microbatch, automatic LR scaling or optimizer resume is registered.

Validation is deterministically class-stratified 50/25/25 into checkpoint,
diagnostic and report roles. Select checkpoints on checkpoint rows only;
report metrics/recovery on common report rows. The parent salience screen
has already used this validation reservoir, so report is not advertised as
a pristine final test. Final test remains sealed. Recovery uses fresh
HLT_X1_CE=0%, OFFLINE_CE=100%; rejection recovery uses linear R50.
Poor performance is a result and never fails a job or prunes descendants.

## Preparation diagnostics

All ordinary-role assignment rows check support, original-HLT equality,
mixed-type validity and rebuilt features at all five coordinates. Each shard
reports native counts, overflow/cropped/filler counts, owner occupancy, exact
joint multiplicity histogram, maximum lengths, salience-weighted delta-R,
delta-R bins of width .01 (unbounded tail), exact quantized counts above
delta-R .1/.2/.3/.5/1.0 and the maximum, per-jet lost-pT/lost-salience
fraction bins of width .001, and per-class counts and pT/salience totals.
The foundation lock aggregates population/per-role/per-class counts and
loss totals. Class-specific cropping is reported, not used to drop jets.

Dense costs and materialized particle views are never persisted. Store only
compact maps, diagnostics, selected weights, reports and class-probability
banks. RAM cache construction has bounded process and pending-result counts.

## Debug-only workflow and genuine acceptance

```text
authenticated parent screen complete
  -> after_matching launcher (creates full canonical dry run)
  -> authenticate -> matcher_acceptance -> per-file assignment jobs
  -> foundation_lock -> partition_validation + audit_storage -> preflight
  -> after_gate launcher -> 17 science tasks
```

The screen completion ID comes from its exact live ledger (currently
21748725), not a hardcoded dependency. If completion is durable, fully
authenticate it instead of requiring an expired scheduler job. Only `afterok`
edges are permitted. Manual live submission of `stage=full` is forbidden.

Resources: metadata 1 CPU/8192 MiB/4h; assignment 1 CPU/8192 MiB/12h;
partition 4 CPUs/320000 MiB/6h; preflight and fits 4 CPUs/320000 MiB/24h/one
A100; reducers 4 CPUs/320000 MiB/6h/one A100. All request account reu-aisocial,
partition debug and qos_tier3, with no requeue. CPU workers limit nested
numerical threads. The existing absolute-path SPORC helper sets the conda
environment, PYTHONNOUSERSITE and LD_LIBRARY_PATH.

These are **requested bounds, not measured acceptance**. New preflight builds
full ordinary-population caches, exercises CE and KD kernels at expanded
coordinates and both HLT endpoints, performs actual installed-Weaver FP32
forward/input/parameter-gradient parity including duplicate-p4 cases, tests
selected-state and T=2 bank round trips, and stresses full batch 256 with
the longest real rows padded to registered capacity and three optimizer steps.
It also checks validation inference and records peak allocated/reserved CUDA
and CPU RSS. Technical acceptance fits cannot be used as science teachers.

CPU RSS must stay below 80% of the request, CUDA allocation below 90% of the
actual A100 capacity, and conservative 100-pass runtime projection below 23h
(1.30 margin plus preparation). Full cache bounds must fit 65% of requested
RAM, leaving worker/optimizer headroom; at least 32 GiB durable space is
required. Genuine acceptance is bound to campaign, environment, hardware,
batch, expanded capacity and exact resources, then rechecked by science workers.
OOM/time failures stop release; no automatic smaller batch or clipping is
allowed. A hardware failure requires a reviewed new resource/recipe decision.

Submission uses immutable canonical plans, per-job intents, receipts and live
ledgers. Retries reuse acknowledged IDs; ambiguous acknowledgements fail
closed. There is no broad cancellation, automatic scheduler migration, or
checkpoint resume. Failed execution outputs are retained for diagnosis;
do not delete partial directories or blindly requeue against them. A reviewed
restart-zero attempt must use exact old IDs and a fresh isolated root.

## Queue interface and current evidence boundary

From a clean, pushed, pinned checkout on sporcsubmit:

```bash
bash scripts/queue_jetclass2_concat_k2.sh
bash scripts/queue_jetclass2_concat_k2.sh --execute
```

First call is dry only. Second explicitly authorizes the whole registered
staged workflow using `AUTHORIZE JETCLASS2 DZFIX CONCAT K2 DEBUG 500K EXACT SPEC`.
`SCREEN_SPEC`, `INVENTORY`, `LAUNCH_ROOT`, `CAMPAIGN_ROOT` can select explicit
paths; source/parent/root checks remain mandatory. Inspect with the thin
`scripts/jetclass2_concat_k2.py` CLI (`gate`, `results --per-class`, read-only
`monitor`, and `audit` for completed matching/cropping diagnostics).

Local synthetic/CPU tests do not certify installed-Weaver or A100 memory.
No real K2 SPORC acceptance, trained classifier, or final-test result is
claimed merely because the queue tooling is implemented.

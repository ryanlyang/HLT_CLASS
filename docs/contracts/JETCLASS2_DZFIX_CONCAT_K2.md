# JetClass2 dzfix fixed-slot K=2 concatenation campaign

Scientific authority: [the active implementation plan](../plans/JETCLASS2_DZFIX_FIXED_SLOT_CONCAT_K2_LADDER_PLAN.md).
Family: `JETCLASS2_DELPHES_CONCAT_K2_*`. Launch and campaign specs are `/v6`,
GPU acceptance `/v5`, training reports and batch probes `/v2` for the
2026-09-23 explicitly authorized batch-128 recipe. These retain v5 strict
parity, v4 preparation reuse and v2 portable scheduling. Matching,
views, seeds and remaining artifacts keep `/v1`. New
`EXECUTION_POLICY/v1` and `EXECUTION_RECORD/v1` bind allowed and actual sites.
Historical v1 debug-only executions and other campaigns are not modified.

## v6 physical batch 128 after measured batch-256 OOM

This section supersedes historical 256-only / 128-before-256 wording in the
v3-v5 sections below. Source `91be01940f814e1ea7a8a460b9694f0148f0d024`,
job 21768860: early D100/D000 and population D100 parity passed. D100 batch
128 completed three training steps plus validation at 21.16 GiB peak tensor
allocation (about 33.57 GiB reserved). Batch 256 OOMed requesting 2.64 GiB
with 2.37 GiB free on the 39.52 GiB A100. No science acceptance was published.

Every new K2 fit uses physical batch **128**, including all controls and x1
compression. Inference batch is also 128 for checkpoint/report validation,
teacher-bank generation and checkpoint/bank round trips. No accumulation,
LR rescaling, particle clipping, row loss or automatic fallback. Last partial
batches run normally. Model, seeds, matching, dataset and pass-based LR/early
stopping stay fixed. Update counts and BatchNorm populations change with the
physical batch; this is not an assertion of identical optimization/results.

`salience_learned_training.train_kernel` accepts explicit positive-integer
training/inference batch overrides without mutating shared globals. Unchanged
callers keep their historical 256 defaults and report layout. K2 opts in on
every fit/preflight call and checks the kernel's reported `batching` record
(training 128, inference 128, accumulation 1). K2 training reports bind the
full registered recipe and inference batch; acceptance checks kernel evidence
instead of accepting a spec-only batch-size change.

Probe policy `k2_registered_batch_128_v1` requires exactly four batch-128
longest-population probes (D100/D075/D000/HLT-x1), three optimizer steps and
validation each. Never retry 256 in this new gate. All early/native/population
parity, CE/KD mini-fits, round trips, 90% GPU / 80% CPU headroom and 23-hour
runtime checks remain mandatory. Partial D100 evidence from 21768860 does not
authorize the other views or science. Historical acceptance is rejected.

Use new source-pinned launch/campaign roots and the original authenticated
`1f930650` K2 preparation donor. No assignment jobs on the verified-import
path. Existing artifacts/jobs are unchanged; pending-job partition-only
debug/tier3 moves remain allowed. Fresh SPORC batch-128 acceptance is required.

## v5 strict parity and early failure detection

Remote 21767292 failed FP32 parameter-gradient comparison, with max absolute
difference `1.5348196e-5` in `pair_embed.embed.0.weight`, after full D100 caching.
CUDA peak was about 0.65 GiB at failure; neither batch probe ran. Equal seeds
are insufficient: local Weaver 0.5.3 / PyTorch 2.5.1 CUDA experiments also
reproduced a native-versus-native strict-gradient mismatch without offloading.

`concat_k2_model.parity_backend` requires deterministic algorithms with
`warn_only=False`, deterministic cuDNN, no cuDNN benchmark and no cuDNN/matmul
TF32. All flags are restored in `finally`; production training is unchanged.
Only the preflight worker exports `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
Python starts. The helper rejects missing/incorrect cuBLAS setup, rather than
claiming a late edit can configure existing CUDA handles. The workspace setting
lasts for that preflight process; strict PyTorch flags are scoped to parity.

Pair-storage policy `k2_pair_saved_tensors_cpu_v2` preserves shape, strides and
dtype. Pinned buffers hold the physical storage span, including gaps, and
backward recreates the original view with a rebased zero storage offset.
This handles transposed, gapped, expanded, overlapping, empty and scalar views
without writing through overlapping destinations. Packed CUDA values retain
no CUDA tensor reference. Native computation and checkpoint keys stay intact.

New `concat_k2_parity.py` reads four distinct registered **training** jets via
bounded ROOT iteration and authenticates their existing K2 assignments and
identity/count joins. No matching is rerun and no final-test rows are opened.
Before any full expanded cache is built, run native-wrapper parity and
FP32/BF16 three-update storage parity on D100 and D000. D000 view construction
receives an offline-free jet. Publish four `EARLY_PARITY/v1` artifacts with
campaign lineage, sample identities/file indices, native report and storage
evidence. Failure leaves prior completed reports but cannot publish acceptance.

The gate requires all four early records plus the existing population-derived
parity and eight ordered probes. Backend policy, pair-storage policy and the
original tolerances are checked exactly, not trusted as unchecked report text:
FP32 `rtol=2e-5, atol=2e-6`; BF16 `rtol=.01, atol=5e-4`. Comparisons still include
all gradients, optimizer state, BatchNorm buffers and three updated checkpoints.
Corrupt/missing/CPU-only records, non-restoring hooks, policy drift and sample
role/identity mismatches fail closed. Real SPORC A100 acceptance is still needed.

Use a new pushed checkout and fresh launch/campaign roots, with
`K2_PARTITION=debug` and the original completed `1f930650` preparation donor.
Do not reuse the failed nested-import `49516092` campaign as the donor.
No old jobs or immutable roots are modified. Scientific batch stays 256;
128 remains a diagnostic, not an automatic fallback. Resource limits remain
90% GPU / 80% CPU / 23-hour projected fit, with measured full-cache execution.

## v4 optional verified K2 preparation import

Set `K2_REUSE_SPEC` to an explicit completed K2 `campaign_spec.json`, or pass
`--reuse-preparation-spec` to `create-launch`. Omit it to compute fresh matches.
The donor must be an original freshly computed K2 campaign (v1 through v6),
never a nested import or the one-to-one salience foundation. A failed GPU
preflight does not invalidate a completed, authenticated matching foundation.

`PREPARATION_IMPORT/v1` pins the donor spec content/file hashes, foundation and
lock hashes, complete preparation receipt closure and matching producer file
hashes. Require identical selected salience source (ignoring only the new
consumer commit), data root, inventory, split membership/counts, capacities,
view contract, K=2 orientation and all-row invariants. Missing receipts,
producer changes, stale sources, conflicting selection, corrupt arrays or
incomplete lock fail closed. No acceptance threshold is weakened.

The CPU importer authenticates every closure payload and rebuilds the donor
lock from validated arrays. It independently copies exact NPZ bytes (no links),
publishes new assignment reports referencing the donor report/foundation and
import hashes, and inventories **every copied file** in its task receipt.
Report `elapsed_seconds` remains the original matching measurement, not import
time; `matching_recomputed=False` makes this explicit. The destination has
`reused_assignments=True` and its own rebuilt lock. Source files are read-only;
keep the pinned donor available for subsequent lineage authentication. Imported
maps still undergo the normal identity join against ROOT jets when views load.

The reuse gate is exactly:

```
authenticate -> import_preparation -> foundation_lock
  -> partition_validation + audit_storage -> preflight
```

There are six gate tasks and the same 17 science tasks (23 in the full dry
run), with no matcher-acceptance or assignment jobs. The import job uses the
CPU-only metadata resource request; it never builds expanded particle caches.
Validation partitioning is rebuilt, not imported. Models, optimizer state,
teacher banks, cached views and GPU acceptance are never imported. The fresh
campaign-bound GPU gate must run the ordered 128/256 probes before automatic
science submission. A CPU import pass is not GPU execution acceptance.

Example opt-in for the completed original donor, in a **new pushed checkout**
and with fresh launch/campaign roots:

```bash
export K2_PARTITION=debug
export K2_REUSE_SPEC=/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_dzfix_concat_k2_1f930650_r1/campaign_spec.json
bash "${PROJECT_DIR}/scripts/queue_jetclass2_concat_k2.sh"
# Inspect the dry run, then use the same command with --execute.
```

Existing launches cannot be retargeted by changing the donor environment
variable. This implementation neither cancels nor changes the already queued
`df29abcc` assignment jobs; any cutover requires separately verified exact IDs.
Partition-only pending-job moves between debug and tier3 remain permitted.

## v3 K2 memory repair and ordered probes

Remote preflight 21757208 failed with a real CUDA OOM in native Weaver
pair embedding (8.22 GiB next allocation, 1.77 GiB free on the 39.52 GiB A100).
This is not resolved by weakening the 90% headroom gate.

`concat_k2_model.K2ParticleTransformer` uses instance-local saved-tensor hooks
inside native `pair_embed` only. Autograd's saved CUDA tensors go to pinned CPU
memory in their original dtype and are restored for backward. Native pair
computations, full BatchNorm population, running buffers, inputs and checkpoint
keys are unchanged. There is no pair chunking, recomputation, gradient
accumulation, precision change or trimming. Eval/no-grad bypasses offload.
The adapter is used only by K2 fits, reducers and K2 acceptance; other campaigns
and the shared native model are unchanged. Policy is hashed into registration.

Fresh acceptance first checks installed-Weaver wrapper parity and storage-only
training parity in FP32 and BF16, on D100 and duplicate-input D000. Three AdamW
updates compare logits/loss, feature and parameter gradients, BatchNorm buffers,
updated weights and optimizer state. Counters must prove real CUDA saves AND
backward retrieval; CPU doubles and no-op hooks are not valid acceptance.

For each D100/D075/D000/HLT-x1 acceptance case, test physical batch **128 first,
then 256**, with a fresh model/optimizer for each batch. Use the longest distinct
real training and validation rows, with train padding to the full registered
capacity; perform three full optimizer updates and validation. Publish each
hashed `batch_probe_ACCEPTANCE_<node>_<size>.json` immediately with status,
allocated/reserved GPU peaks, step/validation timings, throughput and storage
counters. Counters sum saved bytes, not unique live-memory savings.

An OOM at 128 prevents the 256 attempt. An OOM at 256 leaves the successful
128 evidence available and stops before publishing acceptance or releasing
science. Other errors propagate normally. **128 is diagnostic, not an automatic
production fallback.** All science still uses physical batch 256. A production
change to 128 needs an explicit new registration; accumulation is not silently
substituted for the BatchNorm population of 256.

The existing full-cache RAM, 90% GPU/80% CPU headroom, 23-hour projected-fit,
round-trip, source, real-Weaver and sealed-test gates remain. Mini-fit timing
uses the optimized production path. Transfers may slow training; only the real
A100 gate can quantify this. Small parity-fixture timings include diagnostic
copies and are not a throughput benchmark. Historical v1/v2 acceptance cannot
authorize v3 science. Use fresh roots and `K2_PARTITION=debug` for the requested
replacement launch; pending partition-only moves to/from tier3 remain allowed.

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
test inference. Existing production remains on its registered site. New
`jc2k2_*` jobs default to SPORC `tier3`, with an explicit `debug` option.
Every stage, including launchers and CPU preparation, allows pending-job
partition-only moves between these two registered sites.

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
`JC2/CONCAT_K2/v1` domains. Reuse the registered training kernel with explicit
physical batch 128 and inference batch 128, AdamW, peak 3e-4, floor 1.5e-5; three-pass warmup, hold through
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

## Portable SPORC workflow and genuine acceptance

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
the campaign's initial tier3/debug partition and qos_tier3, with no requeue.
Requests keep the <=24-hour common envelope even when submitted to tier3.
CPU workers limit nested
numerical threads. The existing absolute-path SPORC helper sets the conda
environment, PYTHONNOUSERSITE and LD_LIBRARY_PATH.

These are **requested bounds, not measured acceptance**. New preflight builds
full ordinary-population caches, exercises CE and KD kernels at expanded
coordinates and both HLT endpoints, performs actual installed-Weaver FP32
forward/input/parameter-gradient parity including duplicate-p4 cases, tests
selected-state and T=2 bank round trips, and stresses full batch 128 with
the longest real rows padded to registered capacity and three optimizer steps.
It also checks validation inference and records peak allocated/reserved CUDA
and CPU RSS. Technical acceptance fits cannot be used as science teachers.

CPU RSS must stay below 80% of the request, CUDA allocation below 90% of the
actual A100 capacity, and conservative 100-pass runtime projection below 23h
(1.30 margin plus preparation). Full cache bounds must fit 65% of requested
RAM, leaving worker/optimizer headroom; at least 32 GiB durable space is
required. Genuine acceptance is bound to campaign, environment, hardware,
batch, expanded capacity and exact resources, then rechecked by science workers.
The acceptance records both requested and actual execution sites and the
execution-policy hash. It remains valid across allowed partition-only moves;
accepted GPU name, memory, compute capability and installed environment must
still match exactly. Shared execution checks for other campaigns are unchanged.
OOM/time failures stop release; no automatic smaller batch or clipping is
allowed. A hardware failure requires a reviewed new resource/recipe decision.

Submission uses immutable canonical plans, per-job intents, receipts and live
ledgers. Retries reuse acknowledged IDs; ambiguous acknowledgements fail
closed. There is no broad cancellation, automatic scheduler migration, or
checkpoint resume. Failed execution outputs are retained for diagnosis;
do not delete partial directories or blindly requeue against them. A reviewed
restart-zero attempt must use exact old IDs and a fresh isolated root.

### Manual pending-job moves

On v2-v6 portable executions, the scheduler may change `Partition` between `tier3`
and `debug`. The canonical submission commands/ledger remain unchanged as
historical intent. Workers use the actual Slurm partition to activate the
same SPORC environment, independently authenticate it against `scontrol`,
and write a hashed `execution/<task>/<job-id>.json` with requested/actual sites,
policy hash, exact subject/job, node and resources. They reject other
partitions, incorrect job IDs, changed account/QoS, CPUs/RAM/time limits or
inconsistent environment. GPU workers additionally enforce actual A100 and
the exact accepted hardware/software. Monitoring shows both requested and
actual partitions without interpreting the move as source/ledger corruption.

For a **verified pending portable K2 job**, normal Slurm commands are:

```bash
scontrol update JobId=EXACT_PENDING_K2_JOB_ID Partition=debug
# Or, to move it back:
scontrol update JobId=EXACT_PENDING_K2_JOB_ID Partition=tier3
```

These are manual operator actions, not automatic migration or live process
movement. Slurm may reject an update due to site permissions/limits. Do not
change QoS, GRES, memory, CPUs or time as part of the move. Moving a launcher
does not change descendants' initial submission partition; they still use
the immutable campaign choice. Never edit a spec or ledger to match a move.

The previously queued v1 preflight 21757208 and after-gate 21757209 point at
old debug-only source. This patch does not retrofit those jobs. A reviewed
cutover needs a new clean pushed checkout/root and exact old-job accounting;
no cancellation, update, new submission or preparation reuse is performed by
this code change. The unrelated jc2fc/jc2salp campaigns remain out of scope.

## Queue interface and current evidence boundary

From a clean, pushed, pinned checkout on sporcsubmit:

```bash
export K2_PARTITION=debug  # requested replacement; tier3 also supported
bash scripts/queue_jetclass2_concat_k2.sh
bash scripts/queue_jetclass2_concat_k2.sh --execute
```

First call is dry only. Second explicitly authorizes the whole registered
staged workflow using `AUTHORIZE JETCLASS2 DZFIX CONCAT K2 PORTABLE 500K EXACT SPEC`.
`K2_PARTITION=tier3` is the default; `K2_PARTITION=debug` selects debug at
creation. An existing launch refuses a different requested partition on retry;
use its original setting (individual jobs can still be moved with scontrol).
`SCREEN_SPEC`, `INVENTORY`, `LAUNCH_ROOT`, `CAMPAIGN_ROOT` can select explicit
paths; source/parent/root checks remain mandatory. Inspect with the thin
`scripts/jetclass2_concat_k2.py` CLI (`gate`, `results --per-class`, read-only
`monitor`, and `audit` for completed matching/cropping diagnostics).

Local synthetic/CPU tests do not certify installed-Weaver or A100 memory.
No real K2 SPORC acceptance, trained classifier, or final-test result is
claimed merely because the queue tooling is implemented.

# Embedding bridges, responses and regions: recipe v1, execution v8

Scientific authority: [implementation plan](../plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md).
Execution and commands: [runbook](../LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md).
This namespace is distinct from the older embedding/logit ladders.

## Single-A6000 layout (v8)

SPEC/v8 adds EXECUTION_LAYOUT/v1. Setup, input_probe and TP use
`a6000_1_fp32` (one A6000, one process, full local/global128, native BatchNorm).
The29 other fits remain `1080ti_2_fp32` (two processes, local64/global128,
SyncBatchNorm). Both normalize over the complete global batch, including
partial tails. This explicit layout overrides the two default execution fields
in the unchanged historical RECIPE/v1. No claim of bitwise two-rank TP
equivalence is made. Losses, optimizer schedule, epochs, input populations,
crop conditions, checkpointing and HLT-only export are unchanged.

Controller requested/allocated resources, visible devices, torchrun rank
environment, scientific fit receipts, memory records and GPU-hour accounting
must agree with each task's layout. One-rank execution must not initialize
NCCL or convert BatchNorm to SyncBatchNorm. A current single-GPU profile
without its explicit layout is rejected. Historical fit validation defaults
to its original two-rank owner only; it cannot authorize a new worker.

EXECUTION_MATH_POLICY/v2 names the single-A6000 profile with the same strict
deterministic settings. TP_CHECKPOINT/v3 requires all six ordinary-repeat and
checkpoint-update comparisons on that one rank at unchanged tolerances.
PARITY/v2 records fresh single-rank native checks and a separate hash-bound
reference to the pinned v5 PARITY/v1 evidence for the29 Pascal fits. The old
two-rank evidence is read only through explicit historical validation; it is
never relabeled as a current measurement or as single-A6000 TP qualification.
RESOURCES/v5 and ACCEPTANCE/v5 bind one measured A6000 device for TP and the
original two Pascal devices/profile for every other task. Host memory sums
the actual ranks plus launcher; the per-device88% bound is unchanged.

The reviewed source baseline is95e436c6d34885e20a8672d8655343a3b629d458.
The separate R8 recovery additions at that commit do not change consumed
donor runtime. Any subsequent runtime diff must stay inside this campaign's
allowlist. Replace pending bootstrap jobs with a new spec/root, dry review
and exact authorization; changing GRES on an old pinned job is invalid.

## Deterministic A6000 execution (v7)

SPEC/v7 adds EXECUTION_MATH_POLICY/v1. A6000 setup and TP enable deterministic
algorithms with `warn_only=False` for the entire rank. This includes forward,
backward, optimizer, probes and measurement; the parity check cannot use a
different setting from science. Pascal jobs keep deterministic algorithms
disabled as in their saved measurements. Both retain disabled TF32/benchmark,
enabled cuDNN deterministic mode, CUBLAS workspace `:4096:8`, installed Weaver
attention dispatch, FP32, SyncBatchNorm and global128. Missing or altered
registered settings fail closed. No optimizer epsilon or parity limit changes.

TP_CHECKPOINT/v2 records actual numerical settings, policy, dropout-active
ordinary-repeat and ordinary-versus-checkpoint comparisons for all six
conditions on both ranks. Every loss, gradient, parameter, moment, buffer
and RNG comparison must pass the original PROBE_TOLERANCES. RESOURCES/v4 and
ACCEPTANCE/v4 require this evidence. Diagnostic messages identify the failing
comparison and metric. Historical TP_CHECKPOINT/v1 is accepted only in the
explicit v5 resource audit, retaining original spec/hash ownership and limits;
it cannot satisfy current resource admission. RECIPE/v1 and TP_EXECUTION_POLICY/v1
remain unchanged; the new math policy separately versions numerical execution.

## Batched crop frontend and completed-qualification reuse (v6)

SPEC/v6 adds CROP_EXECUTION_POLICY/v1 and required QUALIFICATION_REUSE/v1.
The batched frontend retains the scalar `regions.crops` as its authority:
per-crop contiguous FP64 p4 sums in original order, own-crop frame/normalization,
same17 features, vectors, valid masks, dummy empty tokens, and global128 padding.
Full physical source validation precedes subsetting. No persistent cache or
new scientific inputs are introduced. CROP_FRONTEND/v1 records byte equality
on all-class, long-support and random TRAIN/validation panels, both sources,
all scales. The production native forward, losses and RNG remain unchanged.

Only completed v5 source27a8341210ca686b74d1c538a3ed258bb7fe947d is eligible for
qualification recovery. It must have all30 completed miniatures/profiles,
passing memory/parity/replay checks, only TP over the time ceiling, no setup
acceptance/receipt, and a confirmed FAILED1:0 setup with no active bootstrap
jobs or submitted science. Pin/hash every reused miniature/bank/epoch artifact,
source spec and measurement. Match scientific registration and input-array
bytes. Historical fits and measurements retain original ownership and paths;
new setup records their explicit reuse, not rewritten historical receipts.

RESOURCES/v3 and ACCEPTANCE/v3 require new frontend parity, new TP measurement
and memory/checkpoint proofs, exact preservation of all29 reused profiles,
and the same48h/memory limits. `task_devices` binds each fit to its measured
hardware: TP/setup use two RTX A6000s; the29 other fits use the original two
GTX1080Ti cards. The source profiles and Pascal device records are preserved
exactly. Controller requested/allocated GRES and visible device identities
must match the task; mixed pairs and cross-profile timing reuse are rejected.
World size2, global batch128, FP32 and SyncBatchNorm remain fixed.
Current-runtime distributed parity, original
teacher replay, and CPU qualification still run. TP is measured and admitted
before any unaffected fits; a time failure preserves `resource_TP.json`.
Scientific training always starts fresh and validates its actual parents.

Source admission now uses58d0c0edf5f41cff2398fb168fc16fc92e169294 after review of
the separately committed R8 campaign. Bridge-consumed `storage`,
`physical_columns` and shared `cpu_allocation` function bodies are unchanged;
bridge numerical donors are unchanged. Recipe v1, numerical policy and TP checkpoint policy
remain unchanged. Older v5 resources are readable only through the explicit
historical audit, and cannot serve as v3 acceptance.

## TP memory execution policy (v5)

SPEC/v5 adds the exact TP_EXECUTION_POLICY/v1 declaration. TP alone uses
non-reentrant activation checkpointing on its full jet and every crop while
training, with nested checkpoints around each stateless native transformer
block to bound single-encoder backward memory. Block calls delegate to installed
Weaver; unexpected mutable block buffers fail closed. Replay preserves dropout
RNG and uses temporary cloned
buffers, restoring the original tensor references even on failure. This keeps
normalization updates single-counted without mutating tensors saved for
backward. The registered trim=False factory is required. Parameter names,
native exports, FP32, SyncBatchNorm, global batch128 and RECIPE/v1 are unchanged.

Before replay/export/miniature work, setup publishes TP_CHECKPOINT/v1: all six
conditions, ordinary versus checkpointed updates, dropout active, uneven batch15,
both ranks, existing strict diagnostic-parity tolerances and exact RNG equality.
TP_STRESS/v1 then records real longest-source TRAIN rows for batch128, batch11
and the population remainder, all conditions, with the unchanged memory limits.
Both records are embedded and revalidated in RESOURCES/v2. ACCEPTANCE/v2 cannot
reuse an old profile without this evidence. Later producer miniatures, native
reload, one/two-rank parity and diagnostic-neutrality checks still apply.

Setup164517 passed the existing U100 numerical policy before its TP OOM. That
policy and U100_REPLAY/v3 are unchanged; this is not another tolerance change.

## Population and deployment

Exactly 299234 original TRAIN and 74895 development-validation jets; original
M0HLT/OFFLINE recovery anchors. Original50k overlaps development validation.
The retired historical test was repurposed upstream; no independent-test claim
or final-test reader is introduced. All 22 final classifiers use the same
authenticated D000 input and native 128-wide ParT. Auxiliary heads are removed
by exporting only `mod.*`; native reload reproduces selected probabilities.

There are 30 scientific fits: C0, six intermediate assistants, three D000
endpoints, TR, TP, nine R students and nine P students. R0/P0 alias C0.
`registry.py` fixes every row, coefficient, parent and submission order.
Every final student retains original U100 weight5. New losses ramp linearly
over three epochs. No target enters the classification forward path.

## Target meanings

- Assistant banks contain the **prediction** in original standardized U100
  coordinates. They are never independently standardized or substituted for
  native embeddings. Assistant selection minimizes full-validation original
  U100 prediction error, then favors native AUC and earlier updates.
- TR is a single native encoder/classifier across B/K/I/T/IT/KIT. Only KIT
  receives the weight1 original-U100 coordinate anchor. Six vectors share
  TRAIN population mean/std (floor0.001). Differences are formed against B;
  J=IT-I-T in difference space. Each scalar response RMS has floor0.1;
  J uses IT's scale. R9 predicts absolute vectors with the same scales.
- TP shares its encoder/classifier across HLT/OFFLINE and three regional
  scales. Full CE and CE on the mean of valid crop embeddings each have
  weight1. A crop is encoded independently from its physical particles;
  outside particles and full-jet contextual tokens are unavailable to it.
  TRAIN normalization weights sources/scales/nonempty jets/valid regions
  equally. Whole-OFFLINE P9 has separate TRAIN moments. Empty regions are
  masked targets, with compatible dummy forwards on both DDP ranks.
- The HLT axis and pT90 radius define all regions. Coarse/medium/fine have
  2/4/8 nested regions. No particle matching is used for regional targets.
  P8 changes the target source to HLT; descriptors do not encode source.

Full scientific defaults, mathematical reductions and exact row tables remain
in the implementation plan and hashed RECIPE. Finite poor quality is a result,
never a failed qualification or reason to prune a branch.

## Input provenance and storage

TRANSFER/v1 exports both roles' physical HLT/OFFLINE rows, original assignment
arrays, identities, labels and the original frozen U100 checkpoint. Rebuilding
the assignment NPZ must reproduce its original hash; rematching is forbidden.
The export binds original prepared/source proofs, conventions and summaries.

SPEC/v2 requires `native_input_policy`, `native_root`, `native_hash`, and the
native hash as an explicit parent. The prior repair froze donor runtime
at2fb09f38a3a1da85cdceff9df311963fdb3aa872; v6 uses the reviewed additive baseline
above and allows only additions/modifications
inside the declared campaign runtime. The original scientific donor remains
45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08; imported numerical code is unchanged.

SPEC/v3 retained those inputs and added `u100_extraction_policy` and nullable
`preparation_reuse`. Frozen U100 inference uses original consecutive128-row
contexts, including their padding, with whole batches assigned to ranks.
The random4096 witness evaluates selected rows inside their original batches.
U100_REPLAY/v2 retained atol/rtol1e-4 and added RMS/mean/count diagnostics plus a
non-gating comparison with the old regrouped path. That policy failed163911.

SPEC/v4 adds the complete hashed U100_NUMERICAL_POLICY/v1. Diagnostic164226
showed CPU/GPU RMS5.16e-7 against historical-export RMS6.25e-5, with identical
default/math GPU outputs. The precise historical setting is not established.
U100_REPLAY/v3 admits only all of these fixed bounds:

| Comparison | Maximum absolute | RMS |
| --- | ---: | ---: |
| Historical standardized embeddings | 0.002 | 0.0001 |
| Historical raw embeddings, reconstructed from stored targets/moments | 0.002 | 0.0001 |
| Current CPU math vs GPU default, CPU math vs GPU math, GPU default vs GPU math (standardized) | 0.00001 | 0.000002 |

Historical bounds apply to the4096-row witness AND each path on four evenly
spaced original128-row batches plus the partial tail (610 rows here). Each
comparison preserves original neighbours and padding. The raw comparison
includes stored standardized float32 rounding. A rank-zero CPU copy uses the
same authenticated learned state; it does not initialize a new model or change
the GPU teacher. Thread/backend settings are restored on success or error.
Validators recompute all decisions and reject missing modes, wrong sample
coverage, changed policy, invalid statistics or dishonest pass flags. Legacy
atol/rtol1e-4 counts remain diagnostic only; redundant regrouped inference is
removed in v3. No TRAIN target is replaced. Fresh setup still checks before
full particle preparation; completed preparation still uses explicit recovery.

PREPARATION_REUSE/v1 accepts only completed preparation from source92397736
after its initial U100 witness failure, confirmed FAILED1:0 with no active
bootstrap jobs or science submissions. Original source/spec, completion proof,
all typed manifests/arrays and exact U100 target bytes are authenticated.
Foreign checkout authentication uses the repository-oriented source snapshot
validator with clean-source/exact-commit checks. Its own CLI then enforces the
original imported-module location and publication guards in a separate process.
Hard links preserve payload bytes and avoid duplicate persistent allocation;
new manifests bind the new spec. This credits only absent link destinations
when reserving remaining storage. No old qualification, claim or acceptance
becomes a new execution's evidence. The original source root is retained.

NATIVE_TRANSFER/v1 contains exactly TRAIN D033/D066 and validation
D033/D066/U100 historical NPZ files. It binds original preparation/cache
proofs, exact source snapshot, ordered file inventory, roles and byte count.
Absent source caches may be reconstructed with original assignments on SPORC;
existing corrupt files are fatal. Every emitted NPZ must match the original
byte hash, and all five populations must match their original input digest.
Publication is completion-last and never writes to the historical cache.

Setup copies D000 and TRAIN U100 from the original bundle, OFFLINE from the
parent teacher-sweep delta, and the remaining historical views from the new
native delta. Per-file hashes, identities, labels and particle offsets are
checked before copying; whole-population fingerprints remain exact.
Only D025/D050/D075 are constructed through the original frontend on Brown.
Cross-platform roundoff is never admitted by relaxing historical hash checks.
Field substitutions use the established K/I/T columns and HLT slot identity.
Indices, source keys and matching metadata never enter model tensors.

Original TRAIN U100 arrays and float64 moments are preserved exactly in new
memory-mapped containers. Validation U100 targets come from the original
checkpoint with those TRAIN moments. A fixed4096 TRAIN witness checks the new
extraction against original targets under the declared numerical policy above.
Native D000 FP32 checkpoint replay separately requires AUC difference <=5e-5.

NPY payloads are written privately, flushed/fsynced, exclusively linked to
final names and completed by immutable manifests. Readers validate hashes,
parents, dtype, shape, capability, identities and representation meaning.
Validation target banks are evaluation-only, and technical target banks are
forbidden as scientific parents. They cannot become valid by directory name.
Raw and normalized arrays are processed in bounded chunks or memory maps.

## Artifact inventory

Artifacts use `LUKA_EMBEDDING_BRIDGE_HYDRA_<KIND>/v1`, schema_version1,
except SPEC/v4, schema_version4 and U100_REPLAY/v3, schema_version3. All retain canonical content hashes, explicit
parents and inherited evaluation disclosures. DATA/VIEW/PHYSICAL/U100_BANK
parents include the native delta hash; the changed SPEC hash also separates
every downstream result from the failed v1 execution. Old physical transfers
and the scientific recipe retain their original v1 hashes.

| Kind | Required binding / evidence |
| --- | --- |
| RECIPE / SPEC | Complete task registry, original recipe, source, population, seeds, limits, input transfer and byte inventory |
| TRANSFER | Original source/prepared proofs, assignments, physical shards, U100 checkpoint and each payload hash |
| DATA / VIEW / PHYSICAL / U100_BANK | Spec/transfer/prepared/recipe, role, exact input representation, ordered rows and arrays |
| TARGET_BANK | Producer full-checkpoint hash, task, native/predicted meaning, role, technical flag, indices, TRAIN statistics, validity and file hashes |
| EPOCH / TRAINING_REPORT / FIT | Actual bank lineage, coefficients via recipe/update, selection, all metrics, throughput, memory, target coverage, diagnostics, native reload and output files |
| MINI_POPULATION / PREPARATION | Selected real rows, technical capability, CPU preparation time and peak memory |
| PARITY / REPLAY / U100_REPLAY | Factory/control/distributed/probe neutrality and checkpoint reproduction evidence |
| U100_NUMERICAL_POLICY | Fixed historical/local bounds, sample/mode coverage and diagnostic164226 calibration basis; embedded in SPEC and U100_REPLAY |
| RESOURCES / ACCEPTANCE / CPU_RESOURCES / STARTUP | All GPU/CPU paths, actual timing samples, explicit report extrapolation, memory, per-task walltimes and actual-parent startup checks |
| PLAN / SUBMIT_CLAIM / RUN_CLAIM / TASK | Exact authorized commands, source, accepted job IDs, ownership, measured worker seconds/GPU allocation and complete output inventories |
| RESULTS / PROGRESS | Fresh-C0 comparisons, historical anchors, oracle labels, component metrics, own/ancestor GPU-hours and incomplete states |

Submission also reuses the existing versioned HCWDL exact-DAG ledger/journal
contracts unchanged. Old artifacts do not become this campaign's artifacts.

## Execution and release

GPU jobs: Hydra/gpus/default/normal, one node/task,6CPU,48000MiB.
TP/setup request `gpu:nvidia_rtx_a6000:2`; the other29 fits request
`gpu:geforce_gtx_1080_ti:2`. Both use exactly two torchrun ranks, FP32 and SyncBatchNorm.
Global batch128 is padded before splitting; local means receive
`world_size*local_rows/global_rows` weighting. Keep tail rows and gather unique
evaluation positions in exact original order. TRAIN samplers are seed-bound.

Setup requests12h and remeasures TP on A6000s while authenticating the29
completed Pascal miniatures. It checks all six teacher conditions, source transitions, longest support,
partial batches, native reload, validation, diagnostics and serialized target
exports. Full-fit time is measured/extrapolated with 2x safety +30min, capped
at48h. Each card must use <88% physical memory and aggregate host peaks <85%
of requested RAM. No precision/batch/population/epoch adjustment is automatic.

Original-U100 AdamW parity is exact. One/two-rank comparison tolerances are
loss1e-3, relative-gradient5e-4 and buffers5e-5. Diagnostic neutrality uses
loss1e-6, gradients/parameters/moments5e-6 and buffers1e-6 with exact RNG equality.
These are numerical checks, not accuracy gates.

After setup succeeds, its afterok CPU release submits30 fits and an afterany
report. Release uses1CPU/4000MiB/1h; report2CPU/8000MiB/2h, allowing Slurm CPU
rounding. Only true producer dependencies are present. Producers finish only
after outgoing banks are published. Each scientific fit checks its actual
parents and runs a disposable startup before fresh initialization.

Setup additionally measures CPU authentication and full-size reference metrics
in a disposable process. It validates all 30 real miniature receipts and
extrapolates their history-check cost to 30 epochs. CPU estimates use 2x safety
plus ten minutes and <85% of the corresponding requested host memory. The
setup TASK receipt requires this CPU evidence; GPU acceptance alone cannot
authorize release. Optional S2IT/S6 historical comparisons require an
authenticated selective-campaign spec and frozen prediction/report references;
they never replace fresh C0 or become optimization targets.

Source must be clean, exact and pushed, with only additive runtime in this
namespace since45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08. Submission requires a
full scheduler dry review and one bounded authorization. Claims and journals
are exclusive; ambiguous partial submission never triggers automatic retry.
Failed dependencies become terminal so the report can complete.

Local tests establish readiness to queue setup. Real Hydra qualification is
still required before the release may submit scientific fits.

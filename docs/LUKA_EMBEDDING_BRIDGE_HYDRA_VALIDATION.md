# Embedding bridge / response / region validation

## 2026-11-12 deterministic A6000 repair after setup168529

Remote source6a98c2bda010779b71348ff2cfd07b89f8160fc3 passed all twelve
frontend checks. TP parity had identical loss, normalization buffers and RNG
for all six conditions/ranks, gradient relative errors8.83e-8 to1.31e-7,
moment errors9.14e-8 to1.34e-7, but post-AdamW parameter errors1.13e-5 to1.30e-5
against5e-6. No TP runtime/memory result was produced by that attempt.

Local PyTorch2.5.1 / Weaver0.5.3 fixture / RTX2000Ada reproduction with15
variable-length jets (60..102 particles), independent fine crops and full
eight-plus-two transformer blocks: ordinary-versus-ordinary failed with
parameter relativeL2=1.3112925e-5, gradient=1.5045150e-7; checkpoint-versus-
ordinary gave1.2037169e-5 and1.7256078e-7. Largest differences were in pair-
embedding biases with near-zero gradients. No checkpoint is needed to trigger
the failure. Enabling deterministic algorithms gave zero measured loss,
gradient, parameter, moment and buffer differences for both comparisons.
The math-only SDPA experiment also gave exact checkpoint parity; production
retains the installed fast dispatch rather than forcing math-only attention.

PyTorch2.5.1's [memory-efficient attention backward implementation](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/transformers/cuda/attention_backward.cu#L613-L624)
explicitly disables split-key reduction under deterministic algorithms.
cuDNN deterministic mode alone does not set that switch. This source evidence
and local reproduction identify a credible mechanism; exact A6000 confirmation
still requires Brown's two-rank check on authenticated real jets.

SPEC/v7, EXECUTION_MATH_POLICY/v1, TP_CHECKPOINT/v2 and RESOURCES/ACCEPTANCE v4
bind whole-rank deterministic A6000 execution plus ordinary-repeat controls.
All existing parity tolerances, loss/optimizer settings and Pascal execution
remain unchanged. Historical v5 evidence retains its old schema and audit.
Baseline focused activation/hardware tests:30passed,1two-GPU skip (90.73s).
Post-repair eight-suite run: **171passed,2skipped in188.45s**. Both skips
require two CUDA GPUs/NCCL. New full-depth CUDA regression checks three crop
scales, ordinary repeats, checkpointed updates and diagnostic probes through
complete AdamW updates with unchanged tolerances and exact RNG checks.
Also covers policy tampering, bad repeat measurements, old-v5 recovery and
mixed-hardware admission.34 Python files parse; all three runbook Bash blocks,
both worker/queue shell scripts and scoped whitespace checks pass.
No Brown/Tigris miniature, commit, push or job submission was performed here.

## 2026-11-12 queue instructions after R8 commit58d0c0ed

The separate R8 campaign was committed after the initial repair validation.
Advanced the reviewed source-admission baseline to
58d0c0edf5f41cff2398fb168fc16fc92e169294 after inspecting its runtime diff.
AST comparisons confirm the shared `storage`, `physical_columns` and
`cpu_allocation` helpers consumed by bridge are unchanged. Bridge numerical
donors are unchanged; R8 decoder/model/loss paths are not bridge execution.
Focused source/queue/recovery-credit/hardware tests:6passed in12.00s;
all three runbook Bash blocks and scoped whitespace checks pass.

Runbook recovery now prints exact new free-space demand from the saved
49-array reuse inventory before creating a worktree or submitting jobs.
The known inventory implies about15.88GiB including the1GiB reserve;
20GiB additional free is recommended. The original total-output quota field
is inherited, while actual new disk demand credits hard-linked arrays.
No input copies, remote jobs, commit or push were performed here.

## 2026-11-12 TP runtime repair after setup165857

The user supplied completed resource diagnostics from27a83412. All30
production miniatures and parity checks finished. Only TP exceeded48h:
13311min including2x safety; its components before safety were56.31h training,
47.70h validation and6.66h other. TP memory peaked at7,728,843,264bytes on one
rank and7,090,016,768 on the other; no OOM occurred in this setup.

Local profiling identified repeated scalar crop construction/physical
validation. The batched frontend preserves per-crop contiguous FP64 sum order,
own-frame transforms, final FP32 bytes, masks and global padding. A128-jet
synthetic benchmark with50–150 particles per jet and randomized momenta/tracking
measured coarse0.2505→0.0328s, medium0.4056→0.0384s, fine0.6843→0.0426s.
Every output byte matched. This is a CPU frontend benchmark only; it does not
establish the whole TP speedup or a new walltime on Brown.

Coverage includes empty/single/513-particle crops, neutral categories and
partially valid tracking; corrupted physical fields are still rejected.
All six source/scale conditions retain exact dropout RNG, gradients, buffers
and AdamW updates in local tests. The real prepared-data frontend witness and
historical teacher-bank/new-student consumption run through production code.
Qualification recovery tests retain original ownership, reject modified
artifacts and changed scientific inputs, enforce byte-preserved29 profiles,
reject other historical resource failures, and exercise the setup path that
fits TP alone. An impossible training-only bound stops before full validation.

Local evidence (Python3.10 tagging-hlt, pinned local Weaver fixture, one RTX2000
Ada GPU; no local pair of NCCL GPUs):

- Hardware, qualification recovery, core and activation suites:
  **89passed,2two-GPU skips** in121.16s. Tests reject wrong GRES, mixed GPU
  pairs, cross-profile measurements, changed historical profiles and excess
  resource requests; setup orchestration fits TP alone.
- Native inputs, numerical checks, preparation recovery and crop frontend:
  **76passed**, one fixture missing the new TP profile field. Corrected the
  fixture; its two preparation-order cases then passed in5.57s. Across the
  eight suites, **166distinct tests passed;2 require two CUDA/NCCL GPUs**.
- Includes source/recipe guards, exact crop bytes and optimizer updates, real
  teacher-bank export/consumer integration, one-GPU native Weaver/CUDA checks,
  and corruption/lineage rejection. The local Weaver fixture is not a claim
  of production installed-environment qualification on Brown.
- 35 Python files parsed; all3 current runbook Bash blocks and both worker/
  queue scripts pass `bash -n`; CLI create help and scoped diff checks pass.

No commit, push, SSH, remote inference or Slurm submission was performed here.
SPEC/v6 / RESOURCES-v3 / ACCEPTANCE-v3 are distinct from old execution evidence;
scientific RECIPE/v1 is still exactly equal to the original0a00441f recipe.
The next remote setup must prove crop byte parity, TP runtime/memory, current
distributed update parity and CPU admission. It reuses the29 completed
historical miniatures/profiles instead of fitting them again. The user's latest
instruction moves setup and TP directly to two RTX A6000s. The other29 science
fits remain on two GTX1080Ti cards. Per-task measured device records prevent
cross-profile resource reuse; both profiles retain FP32/global128/two ranks.
The new setup must establish TP's A6000 timing and still fails closed above48h.

## 2026-11-11 TP activation-memory repair after setup164517

User-supplied Brown evidence: U100 numerical qualification passed, followed by
C0, LS_D050 and TR miniatures/exports. TP failed during epoch2 in a crop encoder
with roughly10.45GiB allocated per GTX1080Ti and only5-15MiB free. This is live
activation exhaustion, not the earlier replay check or a scientific-quality
failure. Completed preparation had been reused in388s.

Execution v5 introduces TP whole-encoder plus stateless-block non-reentrant
checkpointing, preserves RNG, isolates recomputation buffer updates, and keeps
all native parameters and scientific settings. SPEC/v5 binds
TP_EXECUTION_POLICY/v1; new TP_CHECKPOINT/TP_STRESS v1 evidence is required by
RESOURCES/ACCEPTANCE v2. Setup now exercises dropout-active update parity and
real longest-support backward passes before replay, miniatures and exports.

Local baseline:49 passed,1 skipped. Final focused five-file suite:
**126 passed,2 skipped in143.41s**, using the installed-Weaver fixture and
PyTorch/CUDA on one RTX2000 Ada laptop GPU. Skips require two CUDA/NCCL ranks.
Tests cover all six TP conditions, uneven batches, repeated autograd.grad
diagnostics, output/gradient/update/moment/buffer/RNG parity, exception restoration,
native reload, schema separation, rejected tampered evidence and early ordering.

Full-depth comparative fixture (batch12,80 particles, eight independent encoder
calls) reduced peak allocated memory from2,064,188,416 to230,850,560 bytes
(1.92GiB to0.215GiB,88.8%). A conservative larger fixture used batch64,
298 particles and eight separate crops each also containing298 particles;
backward completed with finite gradients at9,591,894,528 bytes (8.93GiB), below
the registered88% bound for a10.9GiB card. These synthetic stress fixtures are
not scientific regional samples. Ordinary whole-encoder checkpointing alone
reached10,740,180,480 bytes in that larger fixture; the nested block repair was
added before acceptance. No local timing predicts Pascal performance.

The two initial combined-suite failures were stale SPEC/v4 expectations and a
failure-injection fixture attempting to hash Infinity before validation; both
were corrected, then the complete focused suite passed. Evidence log:
`tmp/embedding_bridge_tp_activation_validation.log`.
Python syntax checks passed for29 files, Bash syntax for5 runbook blocks, and
the scoped repository whitespace check passed.

No new Brown setup or two-GTX1080Ti qualification has run from this source.
No commit, push or submission was performed by the agent. The next step is
the fresh v5 setup using original92397736 preparation; its early real two-rank
parity/memory checks and remaining measured qualification remain mandatory.

## 2026-11-11 bounded historical replay and independent CPU/GPU qualification

Brown diagnostic164226 used the authenticated e0f9d874 source, checkpoint and
original TRAIN bank on610 rows: original batches starting0,99584,199296,299008,
299136 (four128-row batches plus98-row tail). GTX1080Ti,torch2.5.1+cu118,
CUDA11.8,cuDNN90100; inference only. GPU default and math attention were exactly
equal. GPU versus CPU math standardized maximum5.36442e-6, RMS5.16126e-7.
GPU versus historical standardized maximum0.00105965, RMS6.24539e-5; raw
maximum0.000444395, RMS2.90579e-5. CPU versus historical RMS6.24858e-5.
Setup163911's larger4096-row maximum0.00145745 and RMS5.68152e-5 agree with a
small historical-export discrepancy. These observations do not identify an
exact old setting or prove an effect, or absence of effect, on classification.

New SPEC/v4 binds U100_NUMERICAL_POLICY/v1, with U100_REPLAY/v3 evidence.
Historical maximum0.002/RMS0.0001 apply in both raw and standardized spaces;
current CPU/GPU and GPU default/math maximum1e-5/RMS2e-6 remain much tighter.
Both4096-row historical replay and every bounded-probe path must pass. The
legacy elementwise1e-4 criterion is retained as diagnostic evidence. Validators
recompute decisions, validate coverage and reject missing modes/changed limits.
No input/target checksum, source rule or scientific objective is relaxed.
Original TRAIN targets remain byte-identical. Recovery still authenticates
the original92397736 completed preparation and links49 immutable arrays.

Local baseline:34 recovery/native tests passed in42.84s. Initial new-policy
tests:24 passed in9.67s. Combined campaign suites:105 passed,1 skipped in65.70s;
the skip requires two CUDA GPUs/NCCL. Coverage includes sparse large errors,
distributed drift, raw/standardized bounds, all local parity bounds, forged
pass flags, invalid/missing coverage, nonfinite evidence, schema separation,
unchanged recipe and real native CPU/CUDA parity. Final integrated publication
and failure-restoration tests:2 passed in9.49s. Running the numerical suite in
isolation exposed a fixture dependency on earlier tests' disabled-TF32 flags;
the fixture now declares production flags and a fixed initialization seed.
The final isolated numerical suite passed all23 cases in9.80s, including
native witness publication/readback and thread/backend/RNG restoration after
an injected CPU failure. Python syntax25 files and Bash syntax5 runbook blocks
pass; scoped whitespace checks pass. No broad suite repetition was needed
after this test-fixture correction.
No revised production setup, commit, push or submission has been performed by
the agent. Genuine two-GTX1080Ti qualification remains the next acceptance step.

## 2026-11-11 foreign-worktree authentication correction

Registration using4fb32e79 failed before publishing the fresh spec or queueing
jobs: recovery invoked foundation._source on the old92397736 checkout while
that helper was imported from the new checkout. Its module-location guard
correctly rejected the call. The original recovery source test had mocked
that helper, so it missed the cross-worktree failure.

Recovery now validates the original snapshot through the repository-oriented
provenance helper, retaining exact commit and clean-source checks before
executing any old code. The original CLI still checks its own imported path,
published source, immutable registration and exact ledger. No schema or
scientific change; no payload, numerical threshold or qualification bypass.

Evidence: baseline recovery suite21 passed in22.51s; repaired suite27 passed
in26.41s. Six new cases use real Git revisions and detached worktrees, without
mocking source validation or the old CLI subprocess. They reproduce the old
exception, verify successful old-CLI execution, and reject dirty tracked/
untracked source, wrong checkout/commit and tampered snapshot metadata before
scheduler access. Only Slurm responses are simulated. No real repaired Brown
registration or numerical qualification has yet been observed.
The source-runtime allowlist and unchanged recipe/schema checks also passed:
2 tests in6.79s; scoped whitespace checks passed (29 focused passing cases).

## 2026-11-11 original frozen-export batching and completed preparation reuse

User evidence: setup162131, source92397736, completed both data roles and
published PREPARATION/data manifests in8816.77s (host peak15088398336 bytes).
The following U100 replay failed:4096 rows, maximum normalized-coordinate
difference0.0014567375183105469, atol/rtol1e-4. This was not another historical
input hash failure or a storage error. Original saved TRAIN targets remain
unchanged. The original exporter uses consecutive128-row batches; the failed
panel extraction regrouped rows and split each batch across ranks. That
execution difference is established; its responsibility for all remote drift
is not established without a new Brown witness.

Execution v3 preserves original contexts for frozen U100 inference, records
both original-batch and regrouped diagnostics, and retains the strict tolerance.
Fresh setup runs this witness before expensive preparation. Explicit recovery
from the completed92397736 preparation authenticates source/terminal jobs,
completion manifests, all payloads and exact original U100 arrays. It links
49 immutable arrays into a fresh root with new manifest parents; old data,
claims/logs and failed witness remain. No scientific acceptance is reused.

Local evidence (same installed-Weaver environment below):

- Baseline:56 passed,1 skipped in55.84s.
- New recovery/extraction suite:19 passed in24.38s.
- Combined campaign/native/recovery regression:75 passed,1 skipped in66.34s.
- Final actual-CUDA original128/tail replay and early-failure/recovery ordering:
  3 passed in9.55s. These add two cases to the combined suite:77 distinct
  passing cases, one two-GPU/NCCL skip.
- Tests cover nonconsecutive panel order, original batch neighbours, whole-batch
  distribution, tails, duplicate/invalid indices, unchanged tolerance with bad
  targets rejected, payload corruption/partial inventories/lineage rejection,
  terminal accounting, shared payload inodes, and storage-credit accounting.
- Syntax:24 Python files and4 Bash runbook blocks pass.

SPEC/v3 and U100_REPLAY/v2 are versioned; PREPARATION_REUSE/v1 is new.
RECIPE/v1 and both input transfer formats are unchanged. No imported numerical
donor is edited. No repaired two-GTX1080Ti run or Tigris miniature was executed
locally; the numerical replay and remaining full GPU qualification are still
required on Brown. No commit/push or remote mutation/submission was performed
by the agent. Queue readiness here means readiness to run that recovery and
qualification, not a promise that every remaining production check passes.

## 2026-11-11 exact historical-input repair

Remote evidence supplied by the user: setup161306, source0a00441f, failed
before training at the TRAIN D000 fingerprint check. A first-difference scan
found one float32 ULP in relative eta (5.684341886080802e-14). A later full
fingerprint pass found different hashes for D000/U100/D033/D066/OFFLINE, with
all other summary fields equal. Only the first D000 numerical discrepancy
was measured; platform math/layout remains a hypothesis, not a proven cause
or a bound on all differences. Disk reported26GiB free, with no ENOSPC error.

The repair copies authenticated original tensors instead of recomputing
historical views on Brown. A minimal native delta fills five missing
role/coordinate pairs; only the new quarter views use physical reconstruction.
SPEC/v2 binds the delta. Tests explicitly compare original RECIPE/v1 and
TRANSFER/v1 against the original0a00441f implementation: unchanged hashes.

Before queueing, `main` advanced to2fb09f38 with the separate Oscar six-teacher
repair. The embedding-bridge source pin now freezes that exact commit; the
diff contains no changes to imported numerical donors. Its runtime allowlist
remains restricted to this campaign. User cleanup evidence confirms22 failed
partial arrays removed (12.87GiB), with hashes retained, failed root328KiB
and filesystem free space39GiB. This is user-supplied remote evidence, not an
agent-executed deletion or a repaired production acceptance.
After this pin update, the focused source-guard and original-recipe checks
passed:2 tests in6.92s; scoped whitespace checks also passed.

Evidence using the same local environment below:

- Before editing:2 focused preparation/source tests passed.
- New export/preparation/source tests:9 passed in24.03s. Synthetic ROOT
  replay succeeds with all required source cache files removed, reproduces
  original NPZ bytes and leaves source caches absent. Existing corruption
  and injected one-ULP replay changes both stop before completion manifest.
- Full campaign plus native-transfer suite: **56 passed, 1 skipped in50.68s**.
  Includes exact source routing, identity/offset rejection, resealed manifest
  tampering rejection, original U100 preservation, all scientific loss paths,
  real local one-GPU Weaver forward/backward and submission failure handling.
- Final source guard, canonical preparation and recipe checks:3 passed in8.19s.
- Syntax checks:23 Python files,3 Bash runbook blocks; local documentation links
  and CLI help pass. No imported donor runtime changed.

The skipped test needs two CUDA GPUs/NCCL. No real repaired SPORC export or
Hydra/Tigris miniature has run here; local fixture success is not remote
acceptance. The [runbook](LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md) specifies the
fresh v2 execution, reuses the completed physical transfer, and retains
measured two-GPU setup plus automatic release. Output reservation remains
33GiB after the extra input transfer; failed partial arrays are not promoted.

## Original implementation evidence (2026-11-10)

Implementation completed locally on 2026-11-10 for queueing real Hydra setup.
No Brown production setup, remote export, commit, push or research submission
was performed here. No result or runtime improvement is claimed.

Scientific authority: [plan](plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md).
The [runbook](LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md) queues real two-GTX1080Ti
qualification and automatic release; [contract](contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md).
The exact internal donor baseline and reused files are in the
[source map](LEGACY_SOURCE_MAP.md): `45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08`.

## Implemented scope

- All 30 scientific fits and 22 native HLT endpoints; R0/P0 alias fresh C0.
- Three assistant ladders in unchanged U100 axes; shared six-condition TR;
  independently encoded physical-crop TP; all nine R and nine P objectives.
- Original assignments and native frontend; authenticated physical export,
  quarter-view preparation, unchanged TRAIN U100 arrays, evaluation-only
  targets and frozen TRAIN statistics in typed memory-mapped stores.
- Two-rank FP32/SyncBatchNorm execution, true parent dependencies, genuine
  production setup/startup, resource admission and automatic release/report.
- Epoch losses, throughput, memory, target coverage, fixed-panel diagnostics,
  checkpoint selection/replay, stripped native export and read-only contrasts.

## Local evidence

Python: `C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`.
Tests used the repository's existing installed-Weaver fixture at
`tmp/hydra-weaver-fixture-20261102`, one thread per numerical library,
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and CUDA deterministic
workspace configuration `:4096:8`.

1. Baseline donor suite before implementation: **106 passed, 1 skipped**.
2. Combined campaign, two-GPU subclass donor and selective-teacher regression
   suite: **151 passed, 2 skipped in 111.99s**. Both skips require two CUDA
   GPUs/NCCL. This also exercised the donor's real two-process CPU Gloo checks.
3. After final reporting, CPU admission and epoch-observation additions, the
   complete new campaign suite: **48 passed, 1 skipped in 38.18s**. The skip is
   the genuine two-CUDA/NCCL producer pipeline and all-family SyncBN test.
4. After adding measured producer-chain cost reporting: **6 focused tests
   passed in 8.19s**, covering chain accounting, historical references,
   CPU measurements, automatic release and failed-setup handling.

Static checks passed for all 28 campaign files, Python syntax, both shell
scripts, all three Bash runbook blocks, 43 local documentation links and all
58 recorded donor paths at the exact baseline. New files have one final
newline and no trailing whitespace. The CLI help loads successfully.

Reproduction of the final campaign suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -p no:cacheprovider tests/test_luka_embedding_bridge_hydra.py
```

Meaningful campaign checks include:

- Exact original U100 AdamW loss, native state and optimizer-update parity;
  auxiliary heads absent from native reload; every loss reaches the encoder.
- Actual two-epoch native fits for all eight producers, target publication,
  reload and consumption by all final families; full30 C0/R5/P5 fixture fits.
- All original coordinate replays and new quarter views through the real
  physical frontend; changed original assignments rejected despite a newly
  valid outer shard hash.
- Response algebra and scale floors; crop nesting/permutation/isolation;
  valid-region/scale/jet reductions; source-bound targets and controls.
- Technical-bank rejection by scientific readers, corruption/role/hash
  checks, checkpoint selection, exact DAG and duplicate/ambiguous submission
  refusal; CPU/GPU/time/coverage failures preserve diagnostics.
- Full-depth eight-particle/two-class-block **one-GPU** FP32 forward/backward
  for C0, R8, P7, TR and TP with global batch128 and tails98/11, including
  observation gradients. This local card is an RTX2000 Ada laptop GPU,
  not a GTX1080Ti and not a two-rank acceptance run.
- CPU allocation rounding for release/report; measured CPU report kernels
  and admission arithmetic; optional S2IT/S6 ownership/corruption checks;
  missing fresh C0 never replaced by historical BF16 C0; producer-chain
  costs include all assistants and distinguish missing cost evidence.

Synthetic training fixtures reduce depth for fast integration checks; the
separate CUDA cases retain production depth. Neither uses research data.

## Required remote evidence

The runbook pins clean pushed source, exports original physical rows and
U100 weights on SPORC, verifies transfer bytes and available quota on Brown,
then submits setup and durable release. Setup must establish:

- Original view/target/checkpoint reproduction and installed-Weaver parity.
- Genuine two-GTX1080Ti NCCL/SyncBatchNorm loss, gradient, buffer and update
  parity; diagnostic RNG/state neutrality on every scientific family.
- All 30 production miniatures with actual technical parent-bank exports,
  every teacher condition/source/scale, largest support and tail batches.
- Warmed update timings, full validation and bounded serialized export costs,
  summed host/process peaks and per-card peaks under the registered bounds.
- CPU authentication/report measurement and explicit full-campaign estimate.

Each scientific fit additionally authenticates its actual scientific parent
and runs a disposable startup before fresh initialization. Technical weights
and banks cannot be scientific parents. A weak but finite teacher is allowed
to complete and release its dependent students.

Queue readiness is readiness to perform this remote qualification. It does
not mean that its resource estimates, two-rank parity or study outcomes have
already passed on Brown. The next task is the scoped commit/push and runbook;
the implementation changed no existing jobs or donor scientific runtime.

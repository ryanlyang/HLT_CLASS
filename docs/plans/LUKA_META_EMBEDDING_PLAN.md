# Student-benefit OFFLINE embedding adaptation — active v1

This is a new, exploratory fixed30 continuation experiment, not a revision of
the completed w5 campaign. Two HLT-only students start from the exact selected
single-OFFLINE embedding-w5 checkpoint (no logit KD) and the exact OFFLINE teacher
which supplied its targets. No teacher or student is retrained from scratch.

## Registered comparison

- `fixed`: continue CE + 5 mean SmoothL1(beta=1) against original standardized
  OFFLINE embeddings.
- `meta`: the same continuation, but a small TRAIN-only teacher-side adapter
  changes targets to reduce HLT classification loss after five virtual updates.
- Both: 30 additional full passes over the same 95% inner-TRAIN subset, batch128,
  fresh AdamW, 0.1 times the original fixed30 learning-rate curve (3 warmup
  epochs, peak3e-5, floor1.5e-6), native classifier and all encoder weights
  trainable, identical real-update permutations and RNG initialization.
- Remaining 5% of TRAIN is an identity-hash-defined meta-query subset; it is
  excluded from calibration and real/virtual inner updates in BOTH arms.
  The historical initializer already saw these jets: this is NOT independent
  holdout evidence. Validation never enters any loss/gradient/adapter update.
- Full 74,895-jet validation selects each arm's native checkpoint using the
  inherited AUC/CE/rejection/update key. Original50k is report-only and overlaps
  validation. There is NO independent test in this trial ALLJETS scope.

## Shared initialization, not an invented recovered head

The old `selected.pt` saves native weights/buffers only, not the auxiliary head
or optimizer. Authenticate it through the original w5 receipt and original
source fit validator. Restore every native tensor, including the class head and
BatchNorm buffers. Calibrate a fresh deterministic Linear128–GELU–Linear128
predictor for 3 passes on inner-TRAIN with frozen/eval native network, AdamW
lr3e-4, weight_decay0. No validation-based calibration selection. Preserve the
resulting full state as the SINGLE initializer of both continuations.

Authenticate the original OFFLINE teacher; rebuild its compact TRAIN bank once
using the unchanged FP32 extraction and FP64 TRAIN-population mean/std (floor
1e-3). No raw particle cache is persisted. Statistics retain the historical
all-TRAIN convention; they contain no labels or validation data. Teacher stays
frozen; neither its class head nor its encoder is optimized.

## Adapter and objective

On standardized targets z, A(z)=z+0.1*tanh(W2 GELU(W1 z+b1)+b2), with bottleneck32
and W2,b2 initially zero. The bounded residual prevents scale collapse/unbounded
drift. Teacher classifier C is frozen. Before applying C, undo standardization
using the ORIGINAL mean/scale, never refit statistics to adapted targets.

For epochs4–25, every100 real updates within an epoch:

1. Clone current student parameters, buffers and AdamW moments; discard all
   virtual state afterward. Draw five distinct TRAIN-support batches of16 and
   a disjoint meta-query batch of64, deterministically and label-blind.
2. Differentiate through five full-student FP32 AdamW steps on
   CE + 5 SmoothL1(predictor(h), A(z)); targets are NOT detached here.
   Each step uses the next scheduled ordinary learning rate. All parameters
   (including encoder and predictor) participate. No first-order truncation,
   detached backbone, or substitution of SGD.
3. Query the virtual native HLT classifier in eval mode on meta-query HLT only.
   Adapter objective = query CE + 0.1 KL(q_original || q_adapted)
   + 1.0 mean squared standardized residual. Both q distributions come from
   the frozen teacher classifier C applied to unstandardized targets at T=1.
   Preservation is stationary at identity; it does NOT independently reward
   changing targets to improve the teacher's CE. Preservation terms use support
   jets only. This is a short-horizon surrogate, NOT measured recovery/AUC.
4. Update adapter only, Adam lr1e-4, gradient norm cap1. Discard virtual student
   and optimizer. Restore RNG and module modes; no virtual BN state survives.
5. Ordinary real student steps consume detached adapted targets. There is no
   real-student gradient into the adapter and no student logit-KD or
   classifier-on-prediction term. Main classifier remains CE-only.

Epochs1–3 warm the student against identity targets; epochs26–30 freeze adapter.
Virtual attention uses PyTorch math SDPA (supports double backward), FP32 with
TF32 disabled. Real updates retain ordinary BF16/native backend. This precision,
batch-size and eval-query difference is explicit, not exact simulation of a
future batch128 BF16 trajectory. BN buffers are cloned and support-updated;
query uses the resulting virtual buffers. A five-step finite-gradient/parity
check is mandatory. Fixed arm has no meta updates or query-label access.

## Execution and admission

New `luka_meta_embedding` namespace, SPORC A100 debug, 6 CPUs,90,000MiB.
Three jobs: shared `setup`, then independent `fixed` and `meta` with afterok
dependencies. No standalone preflight and no login-node GPU work. Setup is real
useful work (authentication, targets, shared calibration) plus inline acceptance.
Reserve4GiB persistent space (NOT all reported free space); new outputs only.

Setup must measure real installed-Weaver forward/update parity, maximum-support
real and FIVE-step virtual training, a whole inner-TRAIN epoch for each arm
(technical meta probe forces the epoch4 schedule), validation and native saved
checkpoint readback. Admission uses peak memory and a conservative30-epoch
projection against registered12h arm ceilings. Fail closed if unsupported,
nonfinite, too slow or too large; NEVER silently shorten horizon, change batch
sizes/algorithm, train on validation, or bypass authentication. No AUC/accuracy
thresholds. Setup artifacts/receipts and exact Slurm completion gate the fits.

Keep native selected checkpoint, selected full student+adapter training state,
reports and validation predictions. No automatic deletion of old or new assets.
Setup probes are technical and never candidate scientific models. Submission
requires clean exact pushed source, full dry review and explicit plan-hash
authorization. No jobs are submitted by implementation/testing.

## Interpretation

Primary question: does META beat matched FIXED continuation? Historical w5
(44.7% gap recovery) is context, not the sole control. Extra compute and changed
TRAIN allocation must be reported. No promise of improved recovery; poor finite
results are valid and cannot cancel or skip registered fits.

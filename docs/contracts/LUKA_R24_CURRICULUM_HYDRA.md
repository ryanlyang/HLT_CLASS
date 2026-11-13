# R24 curriculum Hydra contract v1

Authority: [active scientific plan](../plans/LUKA_R24_CURRICULUM_HYDRA_PLAN.md).
Namespace `LUKA_R24_CURRICULUM_{RECIPE,SPEC,PLAN,SUBMIT_CLAIM,RUN_CLAIM,PARITY,
REPLAY,RESOURCES,ACCEPTANCE,EPOCH,TRAINING_REPORT,DEPLOYMENT,FIT,STARTUP,TASK,
RESULTS,PROGRESS}/v1`. All records carry version1, content hashes, parent hashes
and the inherited exploratory-development/no-independent-test flags.

SPEC pins three clean sources: the new pushed driver, original R8 science at
`58d0c0edf5f41cff2398fb168fc16fc92e169294`, and completed benchmark at
`1094dea4ccfb36d08e29f740a97d90fc770335ec`. The CLI loads scientific imports
from the original worktree and the additive driver under a separate module
namespace. The benchmark validates its own original-source and receipt chain.
Neither old checkout, spec nor artifact is modified. Interpreter/environment,
prepared population, U100 target hash, fixed reconstruction support and TRAIN
normalization remain bound. New output root cannot overlap protected sources.

RECIPE fixes all twenty cells, seed offset, original optimizer/sampler, global
batch128, 30epochs, CE1/U1005, separate decoder/encoder weight schedules and
deployment. Warm-decoder arms intentionally use two gradient weights, not a
single scalar joint loss. Decoder gradients accumulate at its weight; detached
condition VJPs are scaled to the encoder weight. There is no optimizer reset.
Native-only controls have no decoder. Native architecture does not change.

PLAN uses the reusable HCWDL exact-DAG ledger and write-once journals. Bootstrap
records both setup and CPU afterok release plus the full twenty-fit resource
envelope. Release requires exact successful setup accounting, authentic
four-rank acceptance and the original authorization. Measured commands may
change only bounded walltimes. Fits are independent and ordered as specified;
report waits afterany on all20 exact IDs. Existing complete ledgers return
their IDs without resubmission; partial/ambiguous journals stop for inspection.
No automatic requeue, cancellation or fallback GPU type is permitted.

RUN_CLAIM binds task to accepted ID. Workers cross-check live controller data,
one node/task, 12 CPUs,64000MiB,4 typed GTX1080Ti devices and capability6.1.
The CPU release/report requires one requested CPU (one or two allocated
threads accepted),4000MiB and no GPUs. Only bounded controller reads retry.
Slurm wrappers use absolute source paths, isolated Python, pinned library
directory and deterministic CUBLAS workspace. Four `torchrun` ranks use FP32,
SyncBatchNorm, global padding and sample-weighted SUM gradients. Gradient
presence must agree across ranks. There are no DDP reducer hooks around the
multiple decoder backward passes. Solver threads are three per rank.

PARITY records full-batch versus four-rank loss, gradient, buffers, parameter
updates and AdamW moments for control, both reference seeds, delayed/ramping/
full weights, warm-decoder, cosine, fade and weight1. Exact cases/tolerances
are checked on readback. Native factory parity, unchanged native/AdamW states
during decoder-only preparation, changed decoder parameters and probe RNG/
update neutrality are mandatory. Dropout is disabled only for numerical
parity. Ordinary fits retain independent stochastic streams per rank.

REPLAY checks the authenticated historical native checkpoint on complete
validation in FP32 (AUC absolute tolerance5e-5). This checks execution identity,
not a scientific quality target. Each cell runs a genuine two-epoch technical
fit with real targets and native checkpoint reload. Technical positions7/20
exercise active branches; they are explicitly marked technical and cannot
substitute for a scientific30-epoch fit. Stress includes global128 longest
HLT jets, largest OFFLINE collections, partial11 and the scientific tail98.
Output support is never cropped to fit. Full native validation plus metric
calculation, warmed update samples, input construction, probes, startup and
checkpoint/export costs enter the resource estimate with2x margin and30min
overhead. Host use sums rank high-water RSS and launcher RSS; per-rank GPU
peaks must be below88% and aggregate host below85% of allocation.

ACCEPTANCE is recomputed from all20 measured profiles, and binds every
miniature report. A six-hour setup and48-hour fit cap are fixed before
authorization. Failure preserves diagnostics and blocks release; finite poor
AUC/target quality never does. Technical models/optimizers are discarded.
Every scientific allocation repeats its own two-epoch startup miniature,
then initializes a fresh30-epoch model, optimizer and RNG.

EPOCH logs full validation, original selection-key fields, exact update/LR,
mean CE/embedding/reconstruction losses, separately weighted encoder/decoder
terms, and first-batch per-rank condition probes. The probes use local shuffles;
they are diagnostics, not an original global-shuffle replay. During a delay,
weighted reconstruction condition-gradient norms must be exactly zero.
Validation constructs no OFFLINE targets and calls only features/vectors/mask.
Predictions are gathered exactly once in authenticated global order, including
uneven final batches. The selected checkpoint uses the original AUC, CE,
rejection, earliest-update tie-breaking key.

FIT binds all epoch records, full training state, native-only state, deployment
inventory and saved validation probabilities. Reloaded native predictions
must match the selected model. TASK binds recursively hashed outputs and
accepted job ownership. RESULTS authenticates completed fits; PROGRESS reads
published epochs and marks them provisional. Delta compares fresh D01/D03
with the same seed; historical recovery anchors never replace a missing fresh
control. Neither command performs inference, changes jobs or accesses a test.

No contract in an existing campaign is reinterpreted or versioned in place.

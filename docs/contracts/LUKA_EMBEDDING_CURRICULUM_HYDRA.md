# Embedding curriculum contract v1

Scientific authority is the
[twenty-configuration plan](../plans/LUKA_EMBEDDING_CURRICULUM_HYDRA_PLAN.md).
This is an additive namespace; previous U100 TRAIN targets and all existing
campaign specs retain their meaning. No deployment input or model size changes.

## Registry and loss

`contracts.setting`, `weights`, `learning_rate` and `recipe` define exactly E00
through E19. All use the original initialization and absolute-epoch sampler
seeds, native D000 model, AuxiliaryParT head, AdamW groups, batch 128 and 30 spent
epochs. E00 is the authenticated original C0 same-seed CE1+U100-w5 fit; 19 new
fits are queued. Missing/incompatible C0 is an input error, not permission to
silently substitute a new baseline. Historical SPORC U100 is reported separately.

Targets are original standardized U100 jet embeddings. SmoothL1 uses beta 1
and mean over rows and 128 dimensions, computed in FP32 after frozen TRAIN
normalization. Every finite loss component is logged, including zero-weight
components. Zero-weight terms are excluded from backward: CE-zero classifier
parameters and embedding-zero predictor parameters receive neither gradients
nor AdamW decay. Student forward accepts only the existing HLT batch fields.

## Validation and checkpoint roles

The validation bank is explicitly evaluation-only. Extract from the frozen
original selected COARSE_U100 checkpoint using native pre-classifier features,
FP32, ordered batches of 128, original TRAIN FP64 mean/scale and final FP32
targets. No target statistics are fitted on validation; validation targets
never enter an optimizer step. Both TRAIN and validation inputs are authenticated
and joined by exact ordered identities and labels to their D000 populations.
No final-test inputs are read. Original50k is an overlapping report subset.

Each pretraining epoch records full validation embedding loss. Select its
earliest minimum, except E18 selects the last epoch. E19 has classification
throughout and restarts from its current epoch 10. At transition restore the
selected full model, predictor, buffers and CPU/CUDA RNG. Carry variants restore
the corresponding AdamW moments and counters; reset variants construct fresh
AdamW. Epoch/data-order and schedule clocks count all spent pretraining epochs,
including those after the selected epoch. Never combine earlier weights with
later optimizer state. Final native checkpoint selection excludes embedding-only
pretraining epochs, including E17's weak-CE pretraining window. E00/E19 use all
classification epochs. Native exports omit the predictor; reload and exact
validation metric replay are mandatory.

## Immutable artifacts

All new envelopes use `LUKA_EMBEDDING_CURRICULUM_<KIND>/v1`, schema 1, canonical
content hashes, explicit parent hashes and the repository exploratory-evaluation
flags. Kinds are RECIPE, SPEC, VALIDATION_BANK, BASELINE_PARITY, TRANSITION,
EPOCH, TRAINING_REPORT, FIT, RESOURCE_CHECK, ACCEPTANCE, PLAN, SUBMIT_CLAIM,
RUN_CLAIM, TASK and RESULTS. SPEC binds the clean pushed source, original Group A
spec, C0 reuse proof, teacher transfer, environment, inputs, recipes and resource
limits. VALIDATION_BANK binds teacher, preparation, frozen TRAIN bank and
validation transfer. FIT binds spec, recipe, preparation and both target banks.
TRANSITION includes the complete saved checkpoint's checksum and selected epoch.
TASK receipts bind accepted Slurm IDs to every published output file. Publication
is atomic/immutable; path existence alone never admits reuse.

## Source and execution

Exact internal donor is `b5464e73c94b322a81a33032baea18f1615a5a66`.
Only the new curriculum runtime may be added relative to that donor. Existing
runtime changes require a reviewed compatibility version. Original Group A is
authenticated inside its own pinned worktree, not by importing its foundation
code into the new worktree. Retain original source trees and referenced inputs.

Hydra requests one RTX A6000, 6 CPUs, 32000 MiB, gpus/default/normal. Setup has
12 hours; each measured fit is capped at 48 hours. CPU release uses compute,
1 requested CPU (Hydra may allocate 2), 4000 MiB, 1 hour. Setup checks native
FP32 inference and original U100 one-update model/AdamW parity on installed
Weaver. Each arm gets a discarded warm and measured three-phase miniature
through pre-transition, post-transition and late-stage production paths. These
technical runs do not claim 30 completed epochs: their exact sampled epoch
positions and three completed epochs are recorded. Timing uses the slowest
warmed full batch, non-singleton partial batches, both validation heads, 2x
safety and 900 seconds overhead. Host/GPU peaks must fit 85% of their allocation.
Bad finite AUC, rejection or embedding loss cannot reject a fit or later row.

## Automatic release

Authorization `AUTHORIZE_U100_EMBEDDING_CURRICULUM_V1` binds one exact bootstrap
and the bounded 19-fit plan. Bootstrap queues setup and CPU release with
`afterok` on its exact setup ID. Release requires original authorization,
successful exact-ID accounting, complete setup receipts and measured acceptance;
it persists and dry-reviews every science resource shape before submission.
All 19 fits have no dependencies, arrays, hold flags or concurrency caps.
Queue order is the plan's interleaved curriculum order; scheduler start order
and simultaneous execution are not guaranteed. Complete submission is
idempotent; lost responses or partial journals stop for inspection instead of
retrying unknown jobs. There is no background login process or report job.
`results` and `progress` read saved outputs without inference or mutations.

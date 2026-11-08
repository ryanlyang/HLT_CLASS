# HLT capacity and prediction complementarity

Scientific registration v1. User-authorized implementation: a CPU analysis of
completed S2IT/S4/S6 predictions and twelve fresh student fits. This plan is the
active authority for this separate campaign. It overrides generic Tigris
execution guidance with Hydra RTX A6000; real allocation qualification is
required before automatic science release. No existing execution is modified.

## Fixed questions and comparisons

1. Do equal probability averages of the existing HLT-only S2IT/S4/S6 models
   improve on their individual members? Register the three pairs and the triple,
   with no fitted weights, member search or per-example oracle selection.
2. Does widening the HLT student improve classification, and does it increase
   the benefit of U100 or IT embedding supervision?

Twelve new fits: widths 128 and 256, each with CE alone, CE+5 U100, and
CE+5 U100+5 IT, at two seeds. All are fresh; historical results are context,
not substituted matrix cells. Reuse the completed selective campaign's frozen
IT TRAIN bank and original U100 bank, without new teacher training or exports.

Names are N_CE/N_U/N_UI and W_CE/W_U/W_UI, each suffixed _s0 or _s1.
Submission order per seed: W_UI, W_CE, N_UI, N_CE, W_U, N_U. Seed0 precedes
seed1. Fits have no dependencies on each other and no concurrency cap.

## Scientific recipe

All cells use identical authenticated D000 HLT views: 299234 TRAIN, 74895
development-validation jets, native17 features, 11 classes, batch128 and the
original thirty-epoch optimizer/LR/selection rules. Seed0 uses the parent's
imported native initialization and sampler seeds. Seed1 adds 100003 to each.
Initialization is identical across objectives within a width/seed; shuffle
ordering is identical across widths/objectives within each seed. Head creation
uses isolated RNG streams. No continuation, optimizer reset or schedule search.

Native width128 keeps embed_dims=[128,512,128]. Width256 changes only this to
[256,1024,256]; eight particle blocks, eight attention heads, two class-attention
blocks, pair embedding [64,64,64], dropout, FFN ratio and input support stay as
registered in the installed native Weaver configuration. This increases both
encoder and pooled representation width; it is not a pooled-head-only test.
Predictors are width->128->128 GELU MLPs, with the original U100 and IT head
seeds. Both targets remain independently TRAIN-standardized 128-vectors and
use mean SmoothL1(beta=1). CE-only builds/reads no prediction heads/target banks.
Inference contains only the width-specific mod.* encoder/classifier. No target,
teacher, matching/source identity or predictor is a deployed input.

Use validation macro-OVR-AUC selection with original tie rules. Record all
epochs, weighted losses, sampled encoder gradient norms/cosines, selected full
and deployed states, strict same-width reload parity, parameter counts, state
size, and warmed deployed batch128 latency/memory. Never load a wide checkpoint
through the old fixed-width model loader. Poor finite scores never gate rows.

Report fixed historical M0HLT/OFFLINE gap recovery, same-width/same-seed CE and
U100 deltas, wide-minus-native for each objective, IT incremental benefit at
each width, and differences of these effects. Two seeds give descriptive paired
replication, not a population-level significance claim. Report actual compute;
equal epochs are not equal FLOPs. Failure of one width/schedule is not a ceiling.

## Complementarity analysis

Authenticate completed parent IT, S2IT/S4/S6 task and fit receipts in their own
pinned worktree. Verify prediction file hashes, unique IDs, labels and exact
population; align by identity before averaging. References are never ensemble
members. Evaluate full validation and authenticated overlapping original50k.
Return AUC, recovery, CE, accuracy, per-class OVR AUC, QCD rejection at50% with
censoring/passing counts, pairwise correctness/disagreement tables and confusion
matrices. Top1 complementarity is distinct from ranking/rejection improvement.

Use 500 deterministic paired class-stratified jet bootstrap replicates for
full-validation AUC differences against EVERY member (and identify the fixed
observed strongest member). Resample the same jets for every model. Fixed
historical recovery denominator gives corresponding recovery-point intervals.
Intervals are conditional on already selected checkpoints, not training-seed
uncertainty or selection-corrected significance. The source only authenticates
jet-row IDs; cross-file event independence is unverified. Explicitly label the
bootstrap as a jet-IID sensitivity analysis, never event-cluster uncertainty.
Original50k is report-only, not another independent confirmation. Final test
remains sealed. All ensemble probabilities are simple arithmetic means.

## Execution and artifacts

New namespace LUKA_CAPACITY_HYDRA_*/v1. Pin exact clean pushed source relative
to donor 3fb9e7f11f628ad9d0d8c4c0d3694cc918e6e8a6. Old runtime is unchanged.
Bind parent spec, original source, target manifests/payloads/checkpoints,
prediction receipts, prepared population and immutable recipe. Validate before
reuse; file existence alone is insufficient. New outputs publish atomically.

Bootstrap submits independent CPU analysis and GPU setup, plus a durable CPU
release after successful setup. Setup runs actual-target production miniatures,
initial installed-factory FP32 parity, largest-support and partial backward,
warmed representative timing and full validation for every cell. Setup also
replays the historical native checkpoint on this GPU against its saved AUC.
Use original portability tolerance 5e-5; differing finite scientific accuracy
never fails setup. Each fresh science worker discards its startup miniature.

A6000: one typed GPU, six CPUs,32000MiB RAM;48h fit ceiling. Requests derive
from measured work with2x safety and fixed overhead. CPU analysis/release/report
use compute/default/normal, one requested CPU,4000MiB; accept observed Slurm
one-requested/two-allocated CPU rounding. Analysis has a120-minute bound;
release/report60 minutes. Setup720 minutes. Analysis is not a science quality
gate. Release submits all12 independent fits and afterany CPU report without
login-host polling. Full bounded dry review covers all16 jobs. Exact IDs,
exclusive claims and immutable journals forbid blind duplicate retry.

Ready-to-queue means local contracts/model/integration/submission tests pass
and the runbook can queue real setup. It does not assert remote throughput,
installed Brown parity or scientific gain before that setup runs.

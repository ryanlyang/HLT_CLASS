# Small-data U100-w5 plus soft-subclass weight sweep

Scientific registration v1, 2026-11-10. User authorized ten new independent
HLT student fits at subclass coefficients 0.03, 0.1, 0.3, 0.5, 1, 2, 5, 10,
40 and 100. Reuse the authenticated same-seed Hydra C0 U100-w5 control;
no new teacher or control fit. This plan governs this separate campaign.

Execution v2 adds explicit A6000, L40 and L40S profiles after the user's Brown
diagnostic showed gpu2201 draining for maintenance. Scientific registration,
losses, targets, batch size, seeds and schedule remain v1. One execution selects
one GPU type at creation; its setup and every fit use that same type. A fresh
spec and setup are required to switch profiles; never edit the old execution.

The [GTX 1080 Ti benchmark plan](LUKA_SUBCLASS_GPU_BENCHMARK_PLAN.md) defines
a separate technical one/two-GPU FP32 experiment requested after the GPU
availability discussion. Its artifacts cannot qualify this BF16 campaign or
trigger its release. A later production Pascal profile needs explicit execution
semantics and a fresh matched FP32 U100-w5 control.

## Fixed experiment

Use the original 299234 TRAIN / 74895 development-validation D000 jets,
native installed-Weaver ParT (17 inputs, 11 classes, width128, eight particle
and two class blocks), original initialization/sampling seeds, batch128,
30 complete epochs, AdamW groups, LR schedule and native AUC checkpoint ties.
Each fresh student has the original U100 128-GELU-128 predictor and an isolated
Linear(128,60) subclass head, seed2026110901. One native forward supplies the
pooled embedding to both auxiliary heads; labels are never forward inputs.
Only the native mod.* state deploys. No attention change or extra inference
parameters. The objective is

`CE + 5*SmoothL1(U100 prediction, original U100 target) + lambda*r*KL(q || p)`

where `r=min(1, update/(3*ceil(N_train/128)))` uses one-based updates. All
losses are means over jets; SmoothL1 additionally averages over128 coordinates.
Only the true class's subclass logit group is normalized: QCD10, signals5
each. No 60-way softmax and no subclass-count division. Targets are detached.
No contrastive, CONCAT/FUSION or OFFLINE-embedding prediction term is added.
There is no adaptive loss normalization or gradient projection/clipping.

| Task | Final subclass weight |
|---|---:|
| S0p03 | 0.03 |
| S0p1 | 0.1 |
| S0p3 | 0.3 |
| S0p5 | 0.5 |
| S1 | 1 |
| S2 | 2 |
| S5 | 5 |
| S10 | 10 |
| S40 | 40 |
| S100 | 100 |

Submission order S0p5, S5, S40, S1, S10, S100, S0p3, S2, S0p1, S0p03 covers
moderate and aggressive coefficients early. All ten fits are independent.
Order does not guarantee scheduler start order. High weights may worsen the
scientific result; finite poor metrics never fail or skip any row.

## One frozen TRAIN subclass bank

Authenticate the completed small-data teacher-sweep T00 in its original clean
worktree, including selected checkpoint, full30 report, exact task receipt
and TRAIN OFFLINE embedding bank. Its population, preparation, native recipe,
U100 bank and environment must agree with C0 and this study. No FullStat
3M model, prototypes or targets are reused. No fresh OFFLINE particle forward
is required: use T00's existing standardized preclassifier embeddings.

Reuse the original FullStat array-level clustering kernel and conditional KL
at donor commit5571240c8c42fce1103344dff94c88bf8edf13c5, while giving this bank
a distinct contract, source lineage, population and recipe. Reconstruct raw
embeddings using saved TRAIN mean/scale, apply the frozen linear readout in
FP64, and form ten other-logit minus true-logit margins. Standardize margins
within each true class on TRAIN, population std floor0.001.

Clustering features concatenate sqrt(.75/128)*standardized embedding and
sqrt(.25/10)*standardized margins. Fit MiniBatchKMeans separately inside each
class: QCD10/signal5, seed2026110902+class, batch4096, n_init5, max_iter40,
max_no_improvement20, reassignment_ratio0, tol0 and one numerical thread.
Use all TRAIN rows. No equal-occupancy constraint, pruning or online refit.

Targets are softmax(-squared distance/class temperature). Calibrate each
temperature on TRAIN toward median maximum membership0.8 using48 log-space
bisections over[1e-8,1e8]; finite degeneracy keeps the appropriate bound and
records the achieved value. Publish identities, labels, padded10 probabilities,
prototypes, margins' moments, temperatures and occupancy/entropy diagnostics.
All fits use the exact same bank hash. Validation never fits subclass targets.

## Diagnostics and scientific comparisons

Every epoch publishes native full-validation AUC/accuracy/rejection metrics,
historical recovery, raw losses, weighted losses and current subclass ramp.
First TRAIN batch at epochs1,3,10,20,30 records CE, weighted U100 and weighted
subclass gradient norms and pairwise cosines over shared encoder parameters
excluding the native classifier. Probes retain the original forward and do
not consume RNG or mutate accumulated gradients. They are sampled diagnostics,
not full-epoch averages. Record target-vs-student subclass argmax agreement
and predicted conditional entropy on TRAIN as observations only.

Checkpoint selection ignores all auxiliary diagnostics. Export selected full
and native states; strict reload must reproduce native probabilities/metrics.
Read-only results/progress compare against same-seed C0 and label historical
SPORC U100 separately. Original50k overlaps validation; no independent test
exists in this development population. The sweep is a single-seed screen;
a best validation coefficient is not proof of a repeatable gain.

## Hydra setup and automatic release

New additive namespace luka_subclass_hydra, pinned donor base
5571240c8c42fce1103344dff94c88bf8edf13c5. Existing runtime/worktrees/jobs stay
unchanged. Registration needs the existing teacher-sweep spec only; its C0,
Group A and input ancestry are authenticated by the original source.

Queue one selected-profile GPU setup and durable CPU release(afterok). Setup builds the
shared bank, proves native initialization/forward and zero-subclass U100
model/AdamW update parity, replays the historical native checkpoint, runs all
ten actual-target production miniatures, and measures warmed representative,
longest-support and partial-batch updates plus full validation. Include target
preparation in setup evidence; it is not repeated by scientific workers.
Each science worker repeats a disposable actual-target startup miniature,
then starts its own fresh30 fit. Zero-subclass parity is technical only.

Release authenticates successful exact setup accounting, receipts and measured
acceptance, then submits ten independent fits and report(afterany):13 jobs.
Only walltimes can shrink from the reviewed envelope. Exclusive claims,
content hashes and exact-ID journals reject stale or ambiguous reuse; no
automatic retries. Setup uses12h, each fit ceiling48h with2x measured work,
900s fixed overhead and15min startup margin. Request one selected GPU,6 CPUs,
32000MiB; host peak below85% and GPU below88% of allocation. CPU release/report
request1 CPU,4000MiB,1h and accept Hydra's observed two-CPU rounding.

| Profile | Hydra GRES | CUDA device name | Compute capability |
|---|---|---|---|
| a6000 | gpu:nvidia_rtx_a6000:1 | NVIDIA RTX A6000 | 8.6 |
| l40 | gpu:nvidia_l40:1 | NVIDIA L40 | 8.9 |
| l40s | gpu:nvidia_l40s:1 | NVIDIA L40S | 8.9 |

Keep the original BF16 path and software environment. Runtime checks require
the registered GPU name/capability, native BF16 support and one allocated,
visible GPU. Resource acceptance binds the measured device; A6000 measurements
cannot qualify L40/L40S fits. The full numerical replay and parity checks still
apply. No generic GPU request, reduced batch, FP16 fallback or mixed-GPU sweep.
The original C0 is an A6000 reference: numerical replay checks cross-device
compatibility but does not prove an identical optimization trajectory.

GPU capabilities follow [NVIDIA's CUDA table](https://developer.nvidia.com/cuda/gpus).
The user's 2026-11-10 snapshot lists L40 and L40S in partition gpus without
maintenance drains, but with all6 and8 GPUs allocated respectively. It establishes
eligibility, not a promised start time. A new setup must pass scheduler dry review.

The user's Hydra target and this active plan use real installed-Weaver GPU
miniatures in setup in place of legacy Tigris-only acceptance. Local synthetic
tests establish readiness to queue setup, never remote qualification or benefit.
Final test stays sealed. Commit/push, full dry review and bounded authorization
precede live submission. No submission is part of implementation-only work.

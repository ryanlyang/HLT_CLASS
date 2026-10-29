# FullStat smooth-neighborhood final-rung experiment v1

## Authority and scope

Implement the user's new, fresh HLT-only DE000 using the exact frozen OFFLINE,
hybrid UE100 and recovered non-relational split-head DE066/DE033 teachers behind
`luka_splithead_finalize_retry_5e704ef4_r1/train_DE000`. Do not use the newer
relational ladder, DE022/DE011, or M0HLT as a teacher. Preserve all prior source,
results, banks, data and jobs. This is an Oscar port, NOT the small-trial SPORC
30-epoch fourteen-arm neighborhood campaign.

The user explicitly chose one new smooth-w10 fit and its CPU target build,
reusing historical results rather than training a matched control. All jobs queue in
one reviewed transaction. There is no standalone preflight, teacher training,
automatic retry, final-test access, or cancellation of existing jobs.

## Scientific registration

Reuse the exact FullStat 3,000,000 TRAIN / 500,000 validation split and all input
bytes. Final test 1,500,000 stays sealed, 105,325 unused remain unused; event-level
independence and overlap with the old trial remain unverified. Train from scratch
for the existing 20 complete epochs, physical batch128, original coordinate-paired
initialization/sampler seeds and optimizer/warmup2/cosine20 schedule.

The deployed classifier receives only 0.5 CE. Four removable independent logit
heads receive total K=1 (0.25 each), forward KL at T=2 with T-squared once.
Four removable independent embedding heads receive total E=40 (10 each), mean
SmoothL1 beta1 on each teacher's native 128D representation standardized using
that teacher's TRAIN moments, floor1e-3. No averaging teacher latent axes.

Add one removable Linear128-GELU-Linear36 head at particle block8 before pooling.
Its total SmoothL1 coefficient is 10, reduced as mean36 channels, mean real HLT
particles within each jet, mean jets. Isolated seed94271 does not perturb the
native model, four existing predictors, dropout or sampling. Neither true nor
predicted targets enter the deployed classifier. Zero-weight local tests check
that the added head cannot change the native computation when its loss is off;
they do not constitute a new scientific control run.

Targets are the `smooth_w10` physical descriptors documented in
`docs/LUKA_SMOOTH_W10_NEIGHBORHOOD_HANDOFF.md`: at each real HLT particle, Gaussian
offline neighborhood sums with sigma .02/.05/.10 in wrapped eta-phi distance,
six reconstructed categories, count and offline-pT sums normalized by total
HLT scalar pT. Count transform log1p; momentum transform log1p(100*x).
FP64 geometry/reductions, canonical context ordering, FP32 stored transformed
36-vectors, equal-jet TRAIN normalization with floor0.1. No matching, clipping,
truncation, hard radius, self-removal, validation targets or test particle reads.

## Execution and evidence

Build only ONE smooth target bank, in bounded spawned CPU chunks, replaying
native D000 input bytes and exact identity/label/support ordering. Retain raw
transformed FP32 targets, normalize on batch loading. Borrow existing teacher
banks with their original contracts; no re-export or duplicate large payloads.
Declare distinct durable result and new scratch target roots with separate
personal-quota headroom. Compute actual required target bytes from authenticated
prepared TRAIN support before allocation. Check full RAM residency against the
inherited 180000MiB L40S envelope. No deletion or precision reduction to fit.

Each authorized fit performs disposable native FP32/BF16 parity, CE-only main
head gradients, connected finite auxiliary gradients, max-support batch128
stress and bounded full-objective timing before science. Discard these weights
and reset scientific RNG. This is not a separately scheduled preflight or a
claim of measured full-epoch speed. Insufficient resources fail before science.
Finite poor performance never fails a job.

Select checkpoints by full-validation native-main-head AUC. Publish a durable
FIT before final TASK; saved native-only checkpoint must strictly load and
reproduce selected validation metrics. Retain a completed training snapshot
before publication checks to avoid losing an expensive fit on publication failure.
No silent retry/retraining of partial output. Report the same selected checkpoints
for R30/R50. Previous DE000 is historical (different K/E), not a matched ablation.

Queue readiness requires local target, loss/gradient, installed-Weaver native
parity, miniature production-loop, source/role/tamper, storage, submission and
shell tests. Real Oscar checks remain worker-enforced and must not be claimed
as already run locally.

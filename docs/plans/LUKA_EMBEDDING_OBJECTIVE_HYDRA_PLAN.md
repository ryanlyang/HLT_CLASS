# U100 embedding objectives that emphasize classification information

Scientific registration v1, 2026-11-10. Eight new independent HLT student fits;
reuse the completed same-seed Hydra U100-w5 C0 as the control. No teacher fits,
IT targets, architecture changes, logit targets or additional control fit.
This plan is authoritative for this separate Hydra campaign.

## Fixed experiment

All fits use native installed-Weaver ParT (17 inputs, 11 classes, width128,
eight particle/two class blocks), the original 128-GELU-128 U100 predictor,
the authenticated299234 D000 TRAIN/74895 validation jets, the original seeds,
batch128,30 epochs, AdamW groups, warmup/cosine schedule and checkpoint ties.
Inference exports the native HLT classifier only. Every objective is CE+5*aux.
Select by full validation native AUC using the existing tie rule. All epochs
run; finite poor performance is a scientific result, never an admission gate.
Original50k overlaps development validation. There is no independent test.

Let z be the existing TRAIN-standardized frozen U100 embedding, p its student
prediction and e=p-z. B=mean SmoothL1(e,beta1). The frozen teacher readout in
these coordinates is A=W diag(scale), b'=b+W mean; use centered class rows
A0=A-mean_class(A). Thus common-logit shifts do not define importance.

| Run | Auxiliary objective | Shared-encoder gradients |
| --- | --- | --- |
| R1 | .5 B + .5 calibrated global coordinate-weighted SmoothL1 | ordinary |
| R2 | .5 B + .5 calibrated teacher class-span error | ordinary |
| R3 | .25 B + .75 calibrated teacher class-span error | ordinary |
| R4 | .5 B + .5 calibrated per-jet Fisher coordinate-weighted SmoothL1 | ordinary |
| R5 | .5 B + .5 calibrated TRAIN discriminant-span error | ordinary |
| R6 | .5 B + .5 calibrated covariance-balanced error | ordinary |
| R7 | B | protect CE from opposing auxiliary gradient |
| R8 | R2 | protect CE from opposing auxiliary gradient |

Submission order R8,R2,R4,R1,R7,R3,R5,R6. All eight have no dependencies on
one another; order encourages early coverage but cannot guarantee start order.

## Frozen TRAIN geometry and calibration

Geometry uses only original TRAIN targets/labels and the selected frozen U100
classifier. It never fits validation statistics. All floating-point geometry
is constructed in FP64 and exported in FP32 with its source and payload hashes.

R1 importance is mean_class(A0**2) per coordinate. R4 uses probabilities
softmax((A z+b')/2), and the diagonal Fisher of those probabilities with
respect to z: (E[A**2]-E[A]**2)/4, clipped at zero for roundoff. This is
teacher sensitivity, not certainty that the information is inferable from HLT.
For both, divide importance by its per-vector mean, clip to [.25,4], then
renormalize to mean1. An all-zero vector becomes uniform. Final coordinate
weights are bounded [1/16,16]; the ordinary-loss mixture retains broad coverage.

R2/3 use the orthogonal projector onto row(A0), SVD rank tolerance1e-8 relative
to its largest singular value. The subspace error is mean SmoothL1(e P);
projection is in the original128 standardized coordinates, not an arbitrarily
rotated SVD basis. Rank is at most10. Degenerate rank0 yields zero and is
reported, not rejected. R5 builds class-balanced between/within-class scatter
on TRAIN z. Within scatter has .1 isotropic shrinkage (trace/128, floor1e-6).
Solve the symmetric whitened generalized eigenproblem; retain its up-to-ten
positive directions above relative1e-8, map back and orthogonalize. Use the
resulting Euclidean projector in the same loss as R2. No validation labels
enter this construction. R6 uses the sample covariance of centered TRAIN z,
with .1 isotropic shrinkage, eigenvalue floor1e-6 and inverse-square-root
gain capped at3; apply its symmetric whitening matrix to e before SmoothL1.

Calibration is fixed before any fit: deterministic seed2026111001 selects up
to2048 TRAIN rows without replacement; the original freshly initialized
student predicts them in eval mode, BF16 native forward/FP32 predictor, batch128.
For each alternative term, compute its prediction-gradient RMS across this
same probe and scale to B's RMS, clamped [.1,10]. A zero-gradient term gets
scale1 and an explicit degenerate flag. Freeze these five scales for all fits.
No trained model, validation score or per-epoch adaptive scale participates.
This balances initial prediction-level forces, not guaranteed encoder-gradient
equality. Log actual weighted encoder norms and conflicts during training.

## CE-protected gradient implementation

For R7/8, let c be CE's gradient and a the already-weighted5*aux gradient over
all shared native parameters except mod.fc.*. If c.a<0 and ||c||^2>1e-20,
replace a by a-(c.a/||c||^2)c. Otherwise leave a unchanged. Accumulate c+a
on shared parameters; classifier uses CE only, predictor uses5*aux only.
No projection between layers, no symmetric CE modification, no extra forward,
and no second derivatives. Compute aggregate inner products in FP64; record
pre/post dot products, norms and projection frequency. Nonfinite gradients
fail. Zero gradients are valid. This asymmetric PCGrad-inspired intervention
does not guarantee improvement after AdamW preconditioning or on validation.

## Observation and reuse

Retain selected_full.pt including predictor, plus native-only selected.pt.
Every epoch evaluates native probabilities and predicted U100 embeddings on
full validation using a separate evaluation-only frozen U100 target bank.
Targets use the original TRAIN mean/scale; never feed them into optimization.
Record ordinary error, class-span/complement error, teacher readout accuracy
and CE on real/predicted embeddings, and native AUC/recovery. Diagnostics never
select hyperparameters/checkpoints. Fixed first-batch gradient probes at
epochs1,3,10,20,30 preserve RNG/model state and add no optimizer updates.
R7/8 log projection statistics every epoch. Report per-epoch and selected C0
deltas separately, with historical SPORC U100 as another labeled reference.

## Hydra qualification and automatic release

Reuse Group A's authenticated C0/input lineage and existing U100 encoder and
validation transfer; no new SPORC export is normally needed. Parent validation
runs inside its original pinned worktree. New additive source is based on
37ed91205764667588105948f95a1784964ac8bc. No existing runtime or jobs change.

Queue one A6000 setup and a durable compute release(afterok). Setup authenticates
inputs, builds target geometry/validation diagnostics, proves original native
initialization and a complete ordinary U100 update including AdamW parity,
replays the historical checkpoint numerically, runs each actual-target
production miniature, measures warmed full/partial/longest-support updates and
full validation diagnostics. Discard technical weights before every fresh fit.
Release checks successful exact-ID accounting, receipts and measurements,
then queues eight independent fits and an afterany CPU report: eleven jobs total.

GPU requests: gpus/default/normal, one RTX A6000,6CPUs,32000MiB; setup12h,
fit ceiling48h, measured2x work+900s+15min startup margin. Host<85% allocation,
GPU<88% VRAM. CPU release/report: compute,1 requested CPU,4000MiB,1h;
accept Hydra's observed rounding to2 allocated CPUs. No quality gates.
Full dry review, clean exact pushed source, bounded authorization, exclusive
claims and exact-ID journals precede live submission; ambiguous retries stop.
Local tests establish queue-ready code; genuine Brown setup remains mandatory
before scientific release. No local fixture is evidence of remote qualification.

Method context: [PCGrad](https://arxiv.org/abs/2001.06782) motivates gradient
conflict handling; [function-consistent distillation](https://arxiv.org/abs/2304.11832)
motivates sensitivity to downstream use. The class-span losses are related to
functional/logit distillation; no novelty or guaranteed recovery claim is made.

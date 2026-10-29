# Oscar FullStat U100-w5 student-benefit continuation v1

User-approved port of the student-benefit strategy, not a change to the SPORC
trial or any completed FullStat campaign. Continue the selected HLT-only
U100-w5 student from `luka_fullstat_u100_embedding_w5_7ea8b0c6_r1`, using exactly
its original regular COARSE_U100 teacher and TRAIN target bank. Do not substitute
hybrid UE100, OFFLINE, a new teacher or a model trained from scratch.

## Scientific comparison

Two matched arms: `fixed` keeps the original targets; `meta` adapts them using
TRAIN-internal student-benefit feedback. Both restore every native weight/buffer
from the same selected checkpoint. That checkpoint has no auxiliary predictor
or optimizer. Shared useful setup calibrates a newly seeded Linear128-GELU-
Linear128 predictor for three inner-TRAIN passes, AdamW lr3e-4/decay0, with the
native model frozen/eval. Calibration may not change any native tensor. The
same complete calibrated initializer seeds BOTH arms, with fresh AdamW states.

Keep original 3,000,000 TRAIN / 500,000 validation membership and input bytes;
1,500,000 final-test and105,325 unused rows stay sealed. Choose exactly150,000
meta-query TRAIN rows by lowest SHA256(identity + registered salt) ranks. The
remaining2,850,000 are the real/virtual-support and calibration population for
BOTH arms. Query labels enter only META's outer objective. The old initializer
already saw these rows, so this is NOT an independent holdout. Event independence
and old-trial overlap remain unverified. Never compute validation targets or
optimize against validation labels.

Each arm runs20 additional complete inner-TRAIN epochs, physical batch128,
same sampler/native seeds and fresh AdamW parameter groups. LR is0.1 times the
original FullStat warmup2/cosine20 curve: peak3e-5, floor1.5e-6. Identity targets
in epochs1-2; meta events every100 real batches in epochs3-17; adapter frozen
in epochs18-20. No early stop, performance gate, accumulation or batch fallback.

Real loss is CE +5*mean SmoothL1_beta1(predicted embedding, detached target).
No student logit KD, extra class heads, relational or neighborhood loss. Neither
actual nor predicted embeddings enter the classifier. Deployment is unchanged
native HLT-only ParT; all predictor/adapter machinery is training-only.

## Meta algorithm (unchanged core from7a6dc268)

Original COARSE_U100 128D native `mod.fc` input targets retain their ORIGINAL
full-TRAIN FP64 mean/std, floor1e-3, FP32 standardized bank. Reuse authentic
bank bytes read-only; do not re-export or refit on the new95% split.
Frozen teacher classifier remains Linear128-to11.

Adapter A(z)=z+0.1*tanh(W2 GELU(W1 z+b1)+b2), bottleneck32, zero final layer.
FIXED holds the identical adapter at identity. META clones all student weights,
buffers and real AdamW moments, then differentiates through FIVE full-student
AdamW updates on CE+5*SmoothL1 with non-detached A(z). Draw five distinct support
batches16 and disjoint query64, label-blind. Use upcoming scheduled rates,
FP32/math SDPA/noTF32, cloned support-updated BN, eval query CE. Do not silently
replace with SGD, first-order gradients, frozen backbone or a shorter horizon.

Outer loss = virtual HLT query CE +0.1*KL(q_original||q_adapted)
+1.0*mean((A(z)-z)^2). Preservation distributions come from the ORIGINAL
COARSE_U100 classifier after undoing original normalization, T1. They are not
student KD or CE on the student's predicted embedding. Adapter Adam lr1e-4,
gradient clip1. Discard virtual weights/moments/buffers; restore RNG/modes.
Only detached new targets affect subsequent real student updates.

This short-horizon query-CE surrogate does not directly optimize AUC and does
not guarantee a recovery gain. META has extra compute and query-label exposure.
The primary result is META minus matched FIXED continuation, not improvement
over historical U100-w5 alone. Historical epochs are not continuation epochs.

## Oscar execution

Exactly three jobs: useful shared GPU `setup`, then independent `fixed` and
`meta` afterok dependencies. No separate preflight job or login GPU work.
Setup authenticates original-source completion, exact teacher/bank/checkpoint,
500k native prediction readback, shared head calibration, installed-Weaver
parity and maximum-support FIVE-step hypergradients. It measures a full
inner-TRAIN technical epoch plus full validation/readback for EACH arm (META
uses epoch3 cadence), then checks a1.75x conservative20-epoch estimate.
These disposable probes do not select/init scientific models. This useful
setup can take hours; "no preflight job" does not mean no technical checks.

L40S,6CPU,180000MiB; ceilings setup12h, fixed24h, meta48h. Walltime ceilings
are not measured duration claims. Check site admission, real GPU/environment,
max memory (85%host,90%GPU) and projected time before publishing setup receipt.
An unsupported runtime/gradient/memory/timing failure prevents both fits;
poor finite scientific metrics never do. Historical acceptance is provenance,
not evidence that the new higher-order workload already fits Oscar.

Use repaired runtime explicitly:
`/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar`.
Keep8GiB personal result headroom; no new particle/target banks, deletion, job
cancellation, implicit resume or ambiguous retry. Source stays additive relative
to7a6dc268800b0788e0e89d7ec56db37583397946, with original runtime byte checks.

Save completed native/full state and training report before publication checks;
strict native reload must reproduce selected validation predictions/metrics.
Selected full state includes predictor, adapter, optimizers and RNG for audit
and future explicit recovery; no automatic rolling resume is claimed.

Detached launcher takes bounded explicit three-job authorization, saves complete
dry plan before live submission and returns immediately to a log tail. Exclusive
claims and exact command journals survive disconnect; no blind retries. A
detached process survives SSH disconnect, not host reboot or administrative kill.

Queue-ready requires local higher-order/core regression, installed-Weaver
production-worker miniature,20-epoch schedule/report/selection, role/lineage/
corruption, source, budget, exact DAG, failure and detached-shell tests. Real
Oscar admission remains inside setup, not falsely reported as local evidence.

# U100 embedding pretraining then HLT classification, SPORC debug

Authorized 2026-10-22: one HLT-only student on the existing trial ALLJETS
299234 TRAIN / 74895 validation population. No independent test exists;
original50k is report-only. This is not the separate FullStat Oscar experiment.

Freeze the original fixed30 regular COARSE_U100 selected teacher. Export its
128-dimensional pre-classifier representation in FP32/eval for TRAIN and
validation. Compute population moments on TRAIN only (FP64, std floor1e-3),
apply the same transform to validation, and join by exact jet identity. No
teacher gradient, validation optimization, test access or normalization fit
on validation. Keep the small128-GELU-128 prediction head and native HLT path.

Stage1: fresh matched HLT backbone/head,20 complete epochs of mean SmoothL1
beta1 embedding prediction alone. Classifier parameters are frozen/excluded
from optimization. Select lowest whole-validation embedding loss, earliest
epoch on ties. Save and restore the entire backbone AND predictor, including
buffers; preserve the untouched original classifier initialization. No AUC
selection or CE/class-label loss in this stage.

Stage2: start from that selected stage1 state, reset optimizer and scheduler,
unfreeze classifier,30 complete epochs of CE +5*the SAME embedding loss.
Use original batch128/AdamW/fixed30 warmup3 to3e-4/cosine to1.5e-5 and matched
stage2 sampling seed/order. Stage1 uses warmup3/cosine-to20 independently.
Select full-validation AUC with existing CE/rejection/update tie breaks.
Native deployment omits predictor; no teacher input, KD, fusion or attention.
Poor scores never fail or terminate registered epochs.

Queue one reviewed three-job debug A100 DAG: setup -> pretrain -> finetune.
Setup exports compact targets, authenticates teacher prediction readback,
checks native/predictor parity and maximum HLT support, then runs a disposable
one-epoch-per-stage full-population miniature and verifies native readback.
Science uses fresh weights, never probe weights. Admission measures both
stage epoch costs and checks memory and conservative projections against6h
per-stage requests; setup has8h. All jobs6CPU/90000MiB. Queued dependent fits
must verify successful exact setup accounting/receipt before any training.
This isolated plan uses a genuine SPORC setup instead of the generic legacy
Tigris miniature. Local CUDA tests do not replace remote acceptance.

One submission queues the whole dependency chain (afterok, kill-on-invalid-dep),
without a login-node controller. Immutable specs, exact pushed clean source,
reviewed plan hashes, permanent submission/run claims, exact journals and
stage-parent hashes prevent accidental retries or mixing stages/teachers.
No cancellation, partition editing, release/hold or other campaign mutation.

Reserve2GiB persistent space: compact TRAIN+validation banks (~183MiB raw
vectors), one selected full stage1 checkpoint, final native checkpoint,
histories/predictions and compact acceptance evidence. No particle caches,
per-epoch checkpoints or attention matrices. No automatic deletion; retaining
the selected stage1 state is necessary for reproducible transition/recovery.

Historical M0HLT, OFFLINE, OFFLINE-targetw5 and U100 oracle comparisons use the
same validation rows. They are NOT a same-target/compute-matched causal test:
the new run changes teacher and adds20 epochs. No extra control training is
authorized here. Optional future direct-U10030 and joint-loss50 controls must
be registered separately. Report all epochs of both stages.

Implementation additive under luka_u100_pretrain, baselinee75237a803d3ea6dfb55f110be1a0a899b467bc1.
Do not change old source checks/records. Require focused donor regression,
role/normalization/selection/transition/optimizer tests, real installed-Weaver
CUDA20+30 miniature, DAG/claim/failure tests and shell checks before handoff.

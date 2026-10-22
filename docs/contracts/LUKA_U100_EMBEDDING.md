# LUKA_U100_EMBEDDING / v1

Artifacts use `LUKA_U100_EMBEDDING_<KIND>/v1`, schema 1, canonical content hashes,
parent hashes and the ALLJETS exploratory flags. They are not interchangeable
with D066 banks or U100_PRETRAIN train/validation banks.

- RECIPE: fresh fixed30 CE + weight5 SmoothL1, frozen regular COARSE_U100,
  TRAIN-only 128-dimensional normalized targets; no student logit KD.
- TEACHER: original fixed30 spec, selected checkpoint, training report,
  validation predictions and completed receipt; target_role=train.
- BANK: TRAIN/U100, exact prepared and teacher/recipe parents, row count,
  ordered uint8 identities/int64 labels, FP32 targets, FP64 mean/scale.
- DIRECT_RESOURCES: authenticated completed OFFLINE-w5 resource inheritance,
  donor refs and accounting, 120-minute setup allowance, no new preflight.
- DIRECT_SPEC/PLAN/CLAIM: one tier3 science job, zero preflights/technical
  epochs, exact pushed clean source, canonical dry plan and explicit authority.
- TEACHER_READBACK: native U100 validation probabilities match selected saved
  probabilities within rtol1e-5/atol1e-6; not a validation target bank.
- TRAINING_REPORT/FIT: all30 epochs; exact matched schedule and losses,
  full-validation selection, native-only selected state and readback evidence.
- DIRECT_RESULTS/SCIENCE: retained predictions for all comparisons, original50k
  indices, complete output references and deterministic metric revalidation.

The accepted OFFLINE-w5 reference retains its original D066-import REFERENCE
envelope for compatibility; it contains the OFFLINE-w5 node/files, not D066
targets. It is explicitly validated with its original contract. All newly
generated teacher, targets, scientific and execution artifacts use this family.

Poor metrics never fail admission. Invalid lineage/data, nonfinite values or
missing artifacts fail closed. No independent test exists. Shared data are
read-only; publication is immutable and partial runs are not auto-retried.

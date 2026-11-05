# Shared-message Hydra contracts v1

Authority: [scientific plan](../plans/LUKA_SHARED_MESSAGE_HYDRA_PLAN.md).
Artifacts use `LUKA_SHARED_MESSAGE_HYDRA_{KIND}/v1`, schema_version 1,
content hashes, exact parent hashes and existing exploratory-validation flags.
Kinds: RECIPE, SPEC, PARENT_TEACHER, EPOCH, TRAINING_REPORT, FIT, TASK,
RESOURCE_CHECK, ACCEPTANCE, PLAN, SUBMIT_CLAIM, RUN_CLAIM, RESULTS.

SPEC binds the complete twelve-arm recipe, source snapshot, original A6000 setup
proof, selected teacher_m16 proof and all original input/environment identities.
No original artifact is relabelled. TRAIN and validation identities must align
with original embedding/prediction artifacts. No final-test access.

FIT binds spec, recipe, prepared data and embedding bank; non-control fits also
bind the frozen encoder, and the frozen arm binds its selected teacher report.
FIT authenticates selected.pt, training_report.json and validation_predictions.npz.
Reports select HLT-only AUC, contain every epoch and finite component losses,
message diagnostics and measured runtime. Deployment strips training-only heads.

ACCEPTANCE binds warmup/measured production FIT receipts for every arm, GPU and
environment identity, actual memory peaks, projections and resource limits.
Only technical validity can gate execution. `technical=True` artifacts cannot be
reported as scientific fits. Report assembly replays saved probabilities and
checks population, selection, lineage and file hashes without new inference.

PLAN binds exact commands, source/spec identity and explicit authorization.
Bootstrap and science plans use separate exact DAG journals and single-use
claims. Only accepted exact jobs may publish TASK receipts. Dependencies express
data readiness (setup -> release; fits -> report), never science lanes. Workers
validate source before and after work; completed results remain read-only.

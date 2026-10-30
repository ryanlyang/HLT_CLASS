# LUKA_U100_MULTIVIEW / v1

Authority: [scientific plan](../plans/LUKA_U100_MULTIVIEW_READOUT_PLAN.md).
All new envelopes use canonical content hashes, schema 1, the ALLJETS exploratory
flags, and exact parent hashes. Historical artifacts retain their original schemas.

- **RECIPE/SPEC:** eight registered student arms, two fresh teachers, exact
  historical regular COARSE_U100, fixed30/batch128, CE1, total embedding5,
  separate TRAIN moments and heads, fixed readout ramp, debug execution.
- **IMPORT:** replay the completed direct U100-w5 result in its clean original
  source, plus original fixed30 anchor/teacher replay. Result-only archival
  validation checks all retained receipt outputs except the obsolete TRAIN
  payload; its receipt digest must match the retained bank manifest. Original
  fit validators and comparison replay still run. No training payload from the
  historical U100 result is reused. No checkpoint-existence shortcut, new
  teacher substitution or test access.
- **BANK:** selected checkpoint hash, teacher name and physical-view semantics,
  recipe/prepared parents, TRAIN-only ordered identities and labels, FP32 targets
  and logits, FP64 population mean/std with 1e-3 floor, exact FP32 classifier.
  Classifier reconstruction tolerance: rtol=2e-5, atol=2e-5. No validation bank.
- **TRAINING_REPORT/FIT:** all 30 scientific epochs, full-validation selection,
  exact schedule/sampler, full selected state plus deployed state, readback
  predictions and metrics. Technical miniatures have two epochs and never
  become science teachers or initializers. Poor finite metrics never fail a fit.
- **ACCEPTANCE:** real debug allocation, installed-Weaver environment/GPU,
  all-eight-student and two-new-teacher miniatures, real full-batch longest-support
  training and validation stress, no crop or accumulation, runtime projections,
  GPU/host headroom, exact job and output evidence. Science submission requires
  this completed record and a separate review of measured requests.
- **PLAN/CLAIM/TASK:** independent exact-ID journals for acceptance and science,
  pushed clean source, immutable dry review and authorization, permanent run
  claims, true producer `afterok` dependencies only, report `afterany`.
- **RESULTS:** authenticated completed predictions, fixed historical M0HLT and
  OFFLINE anchors, original50k report-only subset, explicit privileged teachers
  and incomplete rows. No inference from disappearance from `squeue`.

CONCAT unions the existing common physical particle records before the
ALLJETS `literature_context.transform.build_inputs` transform. Keys/source labels
never reach the model. Unknown PID and tracking validity are preserved. The
generic JetClass raw14 transform is deliberately not used: it is not the
historical Luka frontend. No old contract or cache is rewritten.

Deployment uses native HLT for six arms, and encoder + U100-shaped predictor +
combined classifier for the retained arm and capacity control. The latter
control has no target-bank or teacher-model access in its training objective.
Readout CE/KD gradients enter the predicted embedding; teacher weights/moments
are buffers. Neither actual nor predicted teacher features enter native arms'
classifier. New teachers are privileged oracles, not deployable HLT results.

Storage is a fixed four-GiB allowance plus one GiB filesystem headroom, not the
user's entire rounded free-space report. Original inputs/results stay read-only.
The acceptance U100 bank comes from the original selected teacher and is reused
without re-export; technical CONCAT/FUSION banks remain distinct from science.
No automatic retry, broad
cleanup, or epoch checkpoint accumulation is supported.

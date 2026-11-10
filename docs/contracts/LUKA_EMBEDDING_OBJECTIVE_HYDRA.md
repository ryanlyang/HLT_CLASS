# U100 classification-aware embedding objective contract v1

Authority: [scientific plan](../plans/LUKA_EMBEDDING_OBJECTIVE_HYDRA_PLAN.md).
New namespace `luka_embedding_objective_hydra`; artifacts use
`LUKA_EMBEDDING_OBJECTIVE_HYDRA_<KIND>/v1`, schema1, content hashes, exact parent
hashes and original exploratory-population flags. Old artifacts keep their
original validators. Native HLT inference is unchanged; final test is sealed.

RECIPE/SPEC bind eight independent R1-R8 fits, queue order, exact native seeds,
original optimizer/schedule and U100 predictor, the authenticated original C0
reference, Group A source/input ancestry, teacher encoder/validation transfers,
runtime source, fixed TRAIN geometry/calibration, resource ceilings and quota.
C0 is not retrained. All arms have CE1+5*aux and only U100 embedding targets.

GEOMETRY binds the complete TRAIN bank, frozen teacher classifier, preparation,
recipe and spec. Its NPZ includes centered standardized readout/bias,
mean-one bounded global importance, class/discriminant projectors, capped
symmetric covariance whitener, five frozen scales and exact calibration indices.
Calibration records before/after gradient scales, clipping and degenerate flags.
Validation never fits geometry, selects scales or determines gradient weights.

VALIDATION_BANK explicitly authorizes evaluation only: original frozen U100,
native FP32 extraction, batch128, original FP64 TRAIN moments, FP32 targets,
exact ordered D000 validation identities/labels. Optimizer calls receive only
the TRAIN bank. Native selection ignores all auxiliary diagnostics.

EPOCH/TRAINING_REPORT record native full-validation metrics, original selection
ties, weighted CE/auxiliary loss, geometry and validation-bank hashes, auxiliary
validation errors/readout accuracy/CE, scheduled first-batch gradient probes
and protected-update counts/dot products. R7/8 use two first-order gradient
passes over one native forward; only shared encoder gradients are projected.
No quality-based skipping or cancellation. Nonfinite required quantities fail.

FIT inventories selected_full.pt, selected.pt, validation_predictions.npz,
training_report.json and deployment.json. Full state retains the predictor;
native export contains only mod.*. Strict native reload reproduces selected
probabilities and metrics. Both full30 and discarded two-epoch miniatures
have explicit populations, schedules and technical flags; they cannot substitute.

RESOURCES/BASELINE_PARITY/REPLAY/ACCEPTANCE bind actual-target miniature execution,
native FP32 initialization and original U100/AdamW complete-update equivalence,
historical BF16 replay tolerance5e-5 AUC, full validation with diagnostics,
12 warmed update samples, longest-support/partial backward, memory and runtime.
Acceptance is recomputed from saved measurements. Setup receipt inventories
geometry, evaluation targets, parity, replay and every miniature. A scientific
worker repeats a disposable startup miniature, discards it and starts fresh.
Training reports preserve process/allocated-GPU peaks before deployment-only
counter resets; startup admission uses the maximum of this training evidence
and the current process measurement.

PLAN/SUBMIT_CLAIM/TASK and existing HCWDL journals bind exact owned Slurm IDs.
Bootstrap is GPU setup plus CPU release(afterok). After successful accounting
and authenticated measured acceptance, release submits eight independent fits
and a CPU report(afterany). Eleven jobs total. Only walltimes may shrink from
the reviewed envelope. No login dispatcher, fit lanes or automatic retry.
Partial or ambiguous submissions preserve evidence and stop for inspection.

RESULTS/PROGRESS read saved artifacts only. They distinguish the matched
Hydra C0 from historical SPORC U100, selected-control and same-epoch deltas,
finished and provisional results, overlapping original50k and full validation.
Screening is one seed across eight distinct configurations; significance and
recovery improvements require further evidence.

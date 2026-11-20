# Full-validation Hbb checkpoint reselection

Authorized exploratory analysis, 2026-11-20. The user chose all 566,032
validation jets from the completed six-class balanced Oscar campaign,
spec `a68c689360012272a1703f67e6da160a617a2d7937450dd5d83ac0a473a5c7a4`.
This adds an analysis; it does not revise the original campaign's in-range
macro-AUC selection policy or its published results.

For M0HLT, OFFLINE and DE000 independently, evaluate all 20 committed epoch
checkpoints with the original source, runtime, native classifier, BF16 forward,
batch size 128 and original separate in/out validation batching. Pool the
predictions before computing any metric. Do not average in/out AUCs.

Hbb AUC means binary X_bb versus QCD ROC AUC with score
`p(X_bb)/(p(X_bb)+p(QCD))`. Only the 26,937 X_bb and 432,708 QCD rows enter this
metric. The other four signal classes remain in inference batches to preserve
the original padding and batching, but are not treated as background.
Select the highest Hbb AUC per model; exact ties choose the earliest epoch.
Report both original-selection and Hbb-selection results, and
`100*(DE000_AUC-M0HLT_AUC)/(OFFLINE_AUC-M0HLT_AUC)` for each selection policy.
A nonpositive anchor gap gives an unresolved recovery. Also report Hbb R50/R30
at those checkpoints, without independent rejection-based reselection.

This is post-hoc development validation, not an unbiased new test estimate.
No claim of improved all-class macro AUC follows from Hbb reselection. Final
test remains sealed. No training, teacher-bank export, optimizer step, source
mutation, or replacement of original checkpoints/reports is permitted.

One GPU analysis job reuses the HLT resident view for M0HLT/DE000 and builds
OFFLINE separately. Only validation particle spans may reach the original
reader. Shared scalar metadata is authenticated to replay role assignments;
raw file checksums are permitted. Reproduce the published selected checkpoint's
probabilities exactly for both populations before evaluating other epochs.
Reproduce saved in-range macro metrics at every epoch. Publish small immutable
per-epoch metric receipts and a final report in a separate reports directory.
An exclusive process lock prevents concurrent evaluation; a new allocation may
resume authenticated completed metric receipts after an interruption.

Use the existing persistent HOME environment and one L40S, six CPUs,
192000 MiB RAM, 48-hour ceiling. This is a conservative allocation ceiling,
not a runtime estimate. No preflight fit or separate preflight job. Evaluation
parity is checked inside the analysis allocation. The worker records source,
runtime, population, original epoch and job bindings. Contract:
`LUKA_SIXCLASS_HBB_RESELECTION_{SPEC,EPOCH,REPORT}/v1`.

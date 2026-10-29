# FullStat fine-tail DE000 late fine-tuning

Active additive implementation authority, 2026-10-29. The user requested a
short late refinement of the completed 31.8%-recovery fine-tail endpoint,
without a standalone preflight. No other ladder, meta or neighborhood job is
changed. The exact parent is the completed fine-tail campaign at source
`04d1f0f5945ab9af38d83e192eff48e124ad8dc5`, selected DE000 (reported epoch20,
AUC0.96688543); scores are authenticated at execution, never hardcoded targets.

## Science

One warm-start continuation, not an exact resume: the parent saved native
weights/buffers but not auxiliary heads or AdamW moments. Restore selected
native DE000 exactly. Recreate its seven independent embedding and seven
logit heads; cache frozen evaluation-mode HLT TRAIN representations once in
RAM, then calibrate the heads for three TRAIN passes at LR3e-4, batch128,
zero weight decay. Native parameters and buffers must remain byte-identical.
No calibration validation selection, target regeneration or persistent feature bank.

Then unfreeze the complete native model and run ten additional full-TRAIN
epochs with fresh original-group AdamW, cosine LR1.5e-5 to1.5e-6, no warmup
or high-LR restart. Batch128 and sampling seed remain; absolute sampler epochs
21-30 avoid replaying original epochs1-10. Model RNG is explicitly reset.
The same original CE0.5/K1/E40 objective and teacher order are retained:
OFFLINE, UE100, DE066, DE033, M0HLT, DE022, DE011. Each teacher receives
KD1/7 and embedding40/7; T2, native TRAIN-standardized embeddings, separate
auxiliary heads, CE-only deployed classifier. No relational, attention,
neighborhood, meta adaptation or new teacher training.

Exact original 3M TRAIN / 500k validation; no inner split, new data or test
access. Final-test1.5M and unused105325 remain sealed. HLT-only inference
architecture is unchanged. Initializer validation readback must reproduce
its saved predictions/metrics. Select with the original full-validation
key among unchanged initializer (continuation epoch0) and all10 new epochs;
also report best new epoch separately. Poor finite metrics never fail a fit.
This is adaptive single-seed development, not an independent generalization claim.

## Execution and protection

Exactly one Oscar L40S job, 6CPU/180000MiB/24h ceiling, no setup/preflight jobs.
Reuse original acceptance, runtime_repair_v1 and existing compact TRAIN banks
read-only. Inside the fit, verify installed-Weaver/native parity, CE-only head
gradients, maximum-support batch128 memory and bounded full-update timings.
Disposable probe state is discarded. Conservative timing includes calibration,
ten epochs, validation, authentication and publication; technical failure stops
before science, not low performance. No claim that local tests measure Oscar.

New result root with4GiB personally available quota; never modify the parent.
Save calibrated full initializer, every completed epoch's complete model,
optimizer and RNG, selected native/full state and final report before final
publication checks. Partial attempts are preserved; no implicit retry/resume.
Exact pushed source, full dry review and explicit bounded submission authority
are enforced. Detached launch continues after terminal disconnect; no controller
or job cancellation. Parent source replay authenticates fit, banks and scheduler
completion. Saved R30/R50 use the same AUC-selected checkpoint.

## Evidence required

Recipe/schedule/teacher identity and role tests; frozen-native calibration;
head/main gradient and native inference parity; all10 epochs even if worse;
baseline-inclusive selection and full-state publication; corrupt/mismatched
source/bank/receipt rejection; single-job dry/live and detached shell checks;
production-worker miniature with installed Weaver. Real Oscar allocation
admission remains an in-worker requirement, not a separate preflight.

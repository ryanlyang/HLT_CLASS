# LUKA_SIXCLASS_LOW_LR_OSCAR/v1 artifact family

Authority: [scientific plan](../plans/LUKA_SIXCLASS_LOW_LR_OSCAR_PLAN.md).
Implementation: `src/hlt_classification/luka_sixclass_low_lr_oscar/`.

All owned artifacts use canonical content hashes, schema version 1, explicit
parent hashes, development-validation scope, sealed final test and unchanged
event-independence/old-trial-overlap flags. They are distinct from the original
`LUKA_SIXCLASS_BALANCED_OSCAR` artifacts. The old population and external target
banks retain their original contracts; they are never relabeled.

* RECIPE/MODEL/SPEC bind three fresh fits, batch128, 20 epochs, LR1e-4 to5e-6,
  original AdamW and seeds, main CE1 only, DE000 auxiliary embeddings10/KD1
  and contrastive0.3 ramp3. SPEC contains the original population, immutable
  REUSE receipt, current clean pushed source, exact runtime/backend, allocation
  and storage reservation. A rehashed semantic change fails validation.
* REUSE binds the original spec bytes and source, and CONCAT/UE100/DE050 result
  and bank hashes. Registration and worker startup replay authentication in
  the original source/environment. Fresh OFFLINE is never an external teacher.
* PLAN/SUBMISSION_INTENT/SUBMITTED bind reviewed commands to exact job IDs.
  Intents precede sbatch; ambiguous submissions stop. Execution order is
  OFFLINE -> DE000 -> M0HLT. This execution dependency does not make DE000 a
  scientific teacher for M0HLT.
* FIT_BINDING/EPOCH_COMMIT/TRAINING_REPORT bind the population and each teacher
  receipt. Every epoch includes draw-order/exposure evidence, component losses,
  primary in-range metrics, full-validation Hbb AUC/R50/R30, LR and complete
  optimizer/model/RNG state. Epoch publication is completion-last. Explicit
  resume restores the last committed state and the independently tracked best
  primary checkpoint; successful fits are reused, never overwritten.
* STARTUP_PHASE/STARTUP include discarded installed-Weaver parity and a real
  integrated miniature, exact CE-only classifier-gradient checks, candidate
  parity, maximum-support memory and conservative remaining-time estimates.
  Both validation roles are timed; the slower estimate is used for all
  validation batches. Only OFFLINE incurs a TRAIN export.
* BANK contains unique eligible TRAIN identities/labels, standardized128-wide
  embeddings, native T2 soft probabilities and normalization statistics, all
  bound to the new primary-selected OFFLINE checkpoint/report. No M0HLT or
  DE000 TRAIN bank is exported. Frozen teacher banks use original validators.
* RESULT publishes the selected native checkpoint, full training report,
  in/out validation predictions, metrics, startup and view evidence, plus
  OFFLINE's bank. METRICS includes in-range, outside-range, pooled full and
  100GeV-bin diagnostics at that primary checkpoint. RESULTS separately labels
  exploratory full-Hbb-selected metrics for each model, with new-model anchors.

Binary Hbb AUC uses only Hbb/QCD from pooled validation and score
`p(Hbb)/(p(Hbb)+p(QCD))`. Rejection uses the descending signal rank at
`ceil(efficiency*N_signal)` with ties included. Zero QCD passing is censored
(`null` rejection), not measured infinity. Hbb epoch ties select the earliest
epoch. Negative scientific recovery is valid; nonpositive anchor gaps or
censored rejection produce unresolved recovery. No quality threshold fails a job.

The final-test roles and their counts are registered but neither loaded into
particle caches nor evaluated by this campaign. Outside-range validation is
a diagnostic and never contributes to optimization or teacher normalization.

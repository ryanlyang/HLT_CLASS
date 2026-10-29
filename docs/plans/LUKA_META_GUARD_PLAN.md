# Guarded meta-embedding continuation — active scientific plan v3

This is additive to LUKA_META_EMBEDDING, not a change to its running experiments.
The question is whether bounded target adaptation helps when a strong HLT student
is allowed smaller, guarded updates. It does not promise higher recovery or
guarantee that overfitting cannot occur.

## Two matched arms

Both start from the **same explicitly selected, authenticated full student**:

- `completed_fixed`: the completed original FIXED run's selected full state,
  including its trained embedding predictor. Its completion is still mandatory.
- `shared_w5`: the original completed setup's `initial/student.pt`, containing
  the historical OFFLINE-w5 native student plus its already calibrated predictor.
  This mode is authorized after the user cancelled the original FIXED/META fits.
  It requires only completed setup evidence, not completion of either fit. It
  does not recover any continuation improvements from those cancelled fits.

The mode is explicit in the CLI, immutable spec, recipe hash and results; no
automatic fallback or partial-fit reuse is permitted. V1 required completed
FIXED exclusively; that meaning is retained in its original pinned source.
Both modes use fresh AdamW optimizers, CE + 5 times mean SmoothL1, and the
original frozen OFFLINE model's TRAIN embeddings and normalization. No student
logit KD. No new teacher training, target export, or predictor calibration.

FIXED keeps targets unchanged. META starts an identity-initialized, bounded
128→32→128 residual adapter; the teacher itself remains completely frozen.
META performs 100 adapter-only updates before real block 1, 25 before each block
2–16, and none before blocks 17–20. Each update differentiates query CE through
five full-student virtual AdamW steps, using cloned real moments, support batches
of 16, query batch 64, and the current real learning rate. The outer loss retains
the original 0.1 teacher-classifier KL and 1.0 residual-MSE penalties. No virtual
update mutates the actual student, its buffers, optimizer, or RNG.

## Split and schedule

Retain the parent's deterministic 95% TRAIN inner subset. Split its original
5% query pool by a new identity hash into floor-half meta-query and remaining
guard. Real gradients and virtual support use only inner; outer gradients use
only meta-query; rollback uses only guard. Historical initialization trained on
these TRAIN examples: guard is not an independent evaluation set. Reporting
validation is never used for gradient computation or rollback decisions.

Twenty half-pass blocks cover ten additional passes over inner. Consecutive
blocks partition one deterministic shuffled pass. Initial real LR is 5e-6,
constant unless reduced on rollback; no warmup back to 3e-5. Both arms have the
same data order and block seeds. Adapter phases leave the real student stationary.

## Bounded guard

Measure full guard macro-OVR AUC at initialization and after each real block.
Accept a new anchor only if it exceeds the accepted anchor by 0.0002. Two
consecutive blocks more than 0.0002 below that anchor trigger rollback. Restore
the complete accepted student (encoder, classifier, predictor, buffers) and
optimizer moments, halve real LR, and retain the current adapter and adapter
optimizer. Do not rewind sample order or block counters. At most three rollbacks
are allowed (LR floor 6.25e-7). Once exhausted, continue the registered budget;
poor finite performance never fails or skips an experiment.

These thresholds are prespecified engineering defaults, not tuned outcomes.
The same policy applies to FIXED and META. A guard-selected trajectory may still
overfit the guard and may temporarily harm reporting validation.

## Selection, evidence and execution

Evaluate full reporting validation at block 0 and after each guard decision.
Select by the existing AUC/CE/rejection/earliest rule, including block 0. Rejected
candidates are not validation-selected. Save each improving selected full state
durably, plus per-block immutable history. These are audit checkpoints, not an
automatic-resume interface. Deploy only native HLT weights.

Log real loss, guard metrics, decisions, LR, adapter residuals, hypergradient
norms, and matched five-step query CE with identity versus current adapter.
Finite zero/negative improvement is a result; invalid/nonfinite evidence fails.

V3 repairs numerical acceptance only; objectives, optimization, initialization,
splits, selection and the guard policy are unchanged from v2. FIXED's two
separately computed five-step query CE values may differ by at most `1e-6`
absolute (zero relative tolerance). The recorded signed delta must still equal
adapted minus identity exactly, and FIXED residual RMS and maximum must both be
exactly zero. Keep the measured losses/delta, without rounding or clamping.
Larger FIXED discrepancies, changed targets, nonfinite values or malformed
diagnostics fail closed. META has no improvement-sign or near-zero-delta gate.
The diagnostic policy is bound in the recipe hash; v1/v2 artifacts stay under
their original source and cannot be silently reinterpreted with this validator.

Debug: one shared setup/check job then two independent fits, no standalone
preflight. Setup reuses the existing bank and performs native readback, installed
Weaver checks, maximum-support real/virtual checks, and whole-population timing
probes. Exact pinned pushed source, dry review, plan-hash authorization and inline
resource admission remain required. 4 GiB registered persistent allowance, no
automatic deletion. Datasets/Fresh_check/old campaigns are read-only. Final test
remains sealed; original50k remains overlapping report-only validation.

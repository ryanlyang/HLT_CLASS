# LUKA_META_GUARD v3

Authority: [active scientific plan](../plans/LUKA_META_GUARD_PLAN.md).
Artifacts use `LUKA_META_GUARD_<KIND>/v3`, schema 3, canonical content hashes,
explicit parent hashes and the existing exploratory-scope flags.

The immutable SPEC binds the exact new source, explicit `initializer_source`,
full student and native checkpoint, original OFFLINE classifier, original TRAIN
bank, preparation and reference predictions. `completed_fixed` requires original
setup and FIXED receipts and completed scheduler IDs. `shared_w5` requires only
the completed original setup receipt/ID, its full calibrated `initial/student.pt`,
initializer manifest and the historical native w5 checkpoint. It never reads
cancelled FIXED/META fit outputs and records no selected continuation epoch.
Original-source replay validates the chosen dependencies, full inventories and
calibration evidence. Existence alone is insufficient. Bank payloads are borrowed
read-only and must remain available until both new fits complete.

Mode-specific recipe hashes and separate mode-labelled roots prevent conflation.
`PARENT_SHARED_W5` is not a completed FIXED continuation result. No automatic
fallback, native-only initialization, calibration rerun, or checkpoint forgery.
V1/v2 artifacts retain their old meaning and must use their original pinned
source; the v3 validator rejects them rather than migrating an existing spec.

V3 versions the repaired diagnostic acceptance policy, not a new training loss:
`abs(adapted_query_ce - identity_query_ce) <= 1e-6` for FIXED, with zero relative
tolerance. These are separate five-step GPU evaluations; bitwise loss equality
is not required. FIXED residual RMS and maximum remain exactly zero. Both arms
require finite nonnegative CE, finite bounded residuals, and exact recorded
signed-delta arithmetic. Raw measurements are preserved without rewriting.
META may have positive or negative query delta; this is not a performance gate.
The RECIPE binds this policy, and submission authorization is
`AUTHORIZE_LUKA_META_GUARD_DEBUG_V3`. Old roots/specs/receipts are never patched.

The RECIPE freezes 20 half-pass blocks, loss and optimizer semantics, three-way
TRAIN split, adapter events and guard thresholds. REPORT validation replays every
guard decision from TRAIN metrics, verifies block populations, event counts,
loss arithmetic, LR and final selection. Validation cannot be an input to guard
decisions. Published selected native/full states must agree and reproduce saved
validation probabilities. Block 0 is eligible.

TASK receipts authenticate exact output inventories and dependency hashes.
SETUP evidence includes measured full-block runtime, installed native parity,
maximum-support checks and original initializer readback. PLAN/CLAIM and the
exact-DAG journal prevent accidental resubmission after ambiguous scheduler
responses. No requeue, automatic resume, deletion or broad cancellation.

Poor finite guard/validation/query improvements do not invalidate receipts.
Nonfinite values, identity leakage, corrupt hashes, mismatched source, missing
authenticated initializer for the selected mode, or resource violations fail closed.

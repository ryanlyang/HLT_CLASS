# Joint U100 jet + particle prediction: direct debug trial

User-authorized 2026-10-24. Interpret the requested 4/3 as whole-jet weight4,
particle-token weight3, NOT the suggested alternative5/3. New experiment;
the completed token-only and jet-only studies retain their original meaning.

Two fresh HLT-only ParTs optimize CE + 4*jet SmoothL1 + 3*token SmoothL1.
Aligned and unordered arms differ ONLY in token association. Both use frozen
original fixed30 COARSE_U100: final normalized native class vector (fc input)
and eighth particle-block output before class attention. Each128-D target has
its own128-GELU-128 prediction head; neither prediction enters classification.
One native forward captures both surfaces. Both heads are removed from the
selected inference checkpoint; no offline input, target, slot ancestry or
additional computation enters deployment. No logit KD, warm start or ladder.

Same299234 TRAIN/74895 validation trial ALLJETS population, physical batch128,
paired M0HLT initialization/sampler seeds, original AdamW warmup/cosine and
exactly30 epochs. Select by original full-validation AUC/tie rule. Original50k
is a report-only subset; historical test retired; NO independent test remains.
Poor finite scientific results never fail the job or omit a comparison row.

Teacher exports are frozen FP32 deterministic, no TF32, from TRAIN only; moments
are independently fitted in FP64 with population SD floor1e-3. Jet moments weight
each jet equally; token moments weight real TRAIN tokens. Jet loss averages
coordinates and jets. Token loss averages coordinates, real tokens within each
jet, then jets. Aligned uses authenticated persistent slots; unordered uses exact
per-jet detached Hungarian assignment under the same SmoothL1 cost. No duplicate
nearest-neighbor matches or cross-jet transport. These are algorithmic offline
assignments, not truth-particle identity. Both predictor/loss paths use FP32;
native training keeps BF16. Padding is excluded from assignment/loss/moments.

Primary comparison: U100_EMBED_W5 plus TOKEN_ALIGNED/TOKEN_SET, with M0HLT,
OFFLINE and OFFLINE_EMBED_W5 references. Moving jet5 to jet4 as well as adding
tokens means this is a combined-recipe test, NOT a pure additive ablation.
Same numeric coefficients do not imply equal gradient magnitudes. Adaptive,
single-seed development comparison, not evidence of a test-selected winner.

Explicit operational exception: NO new preflight job, technical epoch or
benchmark. Reuse the completed original token-study acceptance at d8dba242
(user reports jobs21958579/80/83 completed), authenticate in its ORIGINAL clean
pinned source, including successful preflight accounting, report, both fits,
teacher and shared data. Compare actual worker GPU/environment/allocation to
that accepted evidence. Existing guards are not disabled; old acceptance is
not relabelled as a measurement of the new objective. Source drift fails closed.
This plan supersedes the token-only plan's NEW-preflight requirement solely for
this distinct contract and implements the user's explicit no-preflight request.

Exactly two directly submitted debug jobs, no dependencies/controller, exact
dry plan/hash authorization and duplicate/ambiguous-submission guards. A100,
6CPU,90000MiB. Per-arm request max(240,ceil(1.25*old measured projection)+30)
minutes; reject above1440 rather than truncating. New combined runtime remains
unmeasured. Teacher readback, target export and finite/input checks are necessary
scientific-job setup, not a hidden technical epoch. No auto-retry or cancellation.

Joint targets exported once per fit to unique owned node-local disk /tmp scratch;
authenticate real mount and actual required free space. Raw token bytes must
equal original accepted export; both new arms' joint descriptors must agree.
Separate jet/token normalization metadata retained. Retire only owned scratch
on normal/error exit; never delete shared caches/models or existing campaigns.
Reserve2GiB durable space for selected native models, predictions, both raw and
weighted loss histories, references, costs and provenance. Saved results work
without the retired scratch payloads. Partial fits cannot produce valid results.

Donor/integration d8dba242c20c65a9c5155f93a68a5eaf12ba5063. Additive namespace
luka_joint_prediction; unchanged older runtimes. Local focused tests, actual
installed-Weaver parity and synthetic production-worker miniature before handoff;
do not claim new remote acceptance from these local checks.

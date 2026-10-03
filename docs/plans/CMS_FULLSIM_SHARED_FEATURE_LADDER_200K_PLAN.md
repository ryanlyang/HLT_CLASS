# CMS FullSim common-interface direct/coarse experiment

## Question and scope

Does the direct-versus-coarse endpoint advantage survive when genuine CMS
FullSim offline/HLT pairs are presented through the JetClass2-compatible
17-channel interface? This is a classifier experiment, not a response-model
fit, synthetic generator, or proof that the two datasets have identical physics.

The default, authorized arm is SHARED17: 200,000 training and 50,000 validation
jets. An explicitly selected CMS21 control may run on the **same rows, matches,
support and seeds**. Selecting both arms doubles nine fits to eighteen; do not
silently queue that extra control. CMS21 tests the native feature representation,
not an exact reproduction of the historical campaign (sample size, no trimming,
and physical view conventions are now controlled).

## Fixed population and access

Import an authenticated existing Scouting file-split v2 manifest. Keep its
train/validation file boundaries, baseline selection and native 15-class labels.
Use the existing seed-1337 proportional-class, smallest-identity-rank selection
within each role, once for all arms. Do not rebuild splits, balance classes to
equal counts, import trained models, or access final-test particle files.
Record selected entries, labels, canonical identity hashes, source byte hashes,
counts and role fingerprints. The native class mixture differs from JetClass2;
cross-dataset macro-AUC/recovery numbers are contextual, not paired estimators.

Keep all source constituents, including offline lost tracks; never use the
response-study bridge which removes them. Capacity is 512 per constructed view,
with a hard overflow failure, never silent truncation. Existing saved CMS native
collections may already be producer-limited. This experiment cannot undo that.

## Physical and feature interface

Four-vectors are native GeV. CMS dxy/dz are cm, converted to mm before the
SHARED17 tanh/uncertainty transforms. Derive a positive uncertainty as
abs(value/significance) only when both stored fields are finite, non-sentinel
and nonzero; unavailable uncertainties become zero, not guessed values.
Zero displacement remains a valid value. Neutral tracking is inapplicable.
No CMS stored significance is an explicit SHARED17 input.
Keep the stored CMS displacement signs; do not guess a cross-producer sign flip.

SHARED17 uses exactly the existing proxy/JetClass2 17-channel analytic adapter:
log pT/E, relative log pT/E, radius, charge, five PID flags, tanh(d0/dz), clipped
errors, relative eta/phi. Rebuild axes from the supplied view. Preserve binary
raw PID flags, including zero/multi-hot cases; do not invent a particle species.
The matching category for ambiguous flags is unknown. Report ambiguity and
error-reconstruction coverage in the preparation audit.

CMS21 uses the registered native 21-field normalization. Geometry channels are
rebuilt from the current view, and other raw fields interpolate with validity
and charged-applicability guards. Native displacement/significance channels
retain CMS units. Quality and lost-hit counts switch discretely. Both arms
have identical physical p4/support/PID switches; this is a **representation**
comparison including scaling/derived channels, not solely an extra-field test.

## Matching and ladders

Reuse the PT_LINEAR full-cardinality salience matcher, not response ancestry.
Every smaller-side particle is paired, with one offline index per HLT slot.
The association is a forced coordinate, not truth matching. Compute it once and
authenticate compact assignments before reuse.

Use the current proxy ladder's persistent support and label-independent keyed
switches. U000 = offline in matched HLT slots, persistent unmatched HLT, plus
unmatched offline tail. U050 removes half the tail by the registered balanced
switches; U100 removes all of it. Matched U100 features are still **offline**.
D066/D033 retain exactly 2/3 and 1/3 offline content; D000 is exact HLT.
No matching, source index, validity mask, or row identity is model-visible.

Per arm, train M0HLT, pure OFFLINE and U000 CE controls, direct U000 -> D000,
and U000 -> U050 -> U100 -> D066 -> D033 -> D000. Every arrow is a **fresh**
fit with 25% CE + 75% forward-KL KD, T=2, from the immediate parent alone.
Five reducers publish selected-model training probabilities. There are nine
fits, five reducers and two closure tasks per arm; no dense/fusion/MT20 graph.

## Model, metrics, acceptance

Installed Weaver ParT, 15 outputs; 17 or 21 input channels, identical remaining
backbone configuration, trim=False in both arms, batch 256. Reuse the literature
recipe: AdamW 3e-4, three-pass warmup, hold through45, cosine to1.5e-5 at60,
maximum100/minimum60, patience15 from first validation, best-AUC checkpoint
with CE/log-R50/earliest-update tie breaks. Pair coordinate seeds, especially
M0HLT and both D000 students. No checkpoint continuation.

Print accuracy, macro AUC, per-class QCD rejection at50% (censor zero passes),
and recovery relative to that arm's M0HLT=0%, OFFLINE=100%. The primary contrast
is coarse-D000 minus direct-D000 **within** an arm. Single-seed validation is
exploratory; no declaration of significance or test qualification.

Preparation is CPU-only and file-sharded. A fresh source-bound A100 preflight
uses complete 200k/50k RAM caches, genuine installed-Weaver FP32 parity,
real miniature CE/KD/reduction, worst-length full-batch forward/backward probes,
and measured CPU/GPU/time/storage envelopes. Models are measured separately.
Science requires passed acceptance for every requested arm and a full reviewed
dry plan; it cannot reuse old CMS/JC2 gate evidence. Default site is SPORC/debug,
16 CPUs,160000 MiB,one A100; every GPU job fits the 24h partition limit.

All source/spec/artifact/teacher identity joins fail closed. Weak metrics never
fail or skip a node. New roots only, no cancellation or modification of existing
campaigns, immutable output receipts, durable exact-ID submission journals.
Failed jobs require explicit inspection/recovery, never blind resubmission.
Final test stays sealed throughout.

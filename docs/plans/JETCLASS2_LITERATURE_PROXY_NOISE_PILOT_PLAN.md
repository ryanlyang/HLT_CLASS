# Final count-preserving noise pilot (v3)

Status: implementation specification; not production authorization.

## Purpose and scope

Run one final, paired training-only pilot before deciding whether to make a
2.25-million-jet synthetic dataset (1M train, 250k validation, 1M final test).
This is a literature-inspired controlled benchmark, NOT CMS detector HLT and
not a response fitted to CMS. Increasing noise does not establish realism or
improved knowledge distillation. No classification-quality threshold is a job
failure. No production generation, submission, or automatic followup is
authorized by this plan.

The completed count38 v2 pilot is the immutable parent. The observed Tigris
run was job 218313, rooted at
`/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_literature_count38_d4b5d5fd_r1`.
Its 20,000 training jets contained 818,458 offline and 760,042 v2 particles;
38.0021 output particles/jet. It passed exact serial/process replay.

## Frozen intervention

Compare OFFLINE, the saved COUNT38_V2 particles, and NOISE_V3 on exactly the
same training identities. Do not read raw ROOT, labels, native HLT, validation,
final-test particles, or CMS particles. Inherit the original label-independent
random streams and the complete full-precision calibration artifact. Never
copy rounded console probabilities or repeat the count calibration.

| Mechanism | v2 | v3 |
| --- | --- | --- |
| PID probability scale relative to nominal | 3 | 3 |
| Added tracking-noise amplitude scale | 3 | 4 |
| Relative pT and angular-noise amplitude scale | 1.5 | 2 |
| Drops and disjoint merges | Saved calibration | Identical calibration and draws |

Tracking values and errors are mm. For original error sigma and crowding c,
added variance is `s^2 * (((1.5 + 0.5*c)^2 - 1)*sigma^2 + floor^2)` with
floors 0.020 mm (d0) and 0.050 mm (dz). Add a Gaussian with that variance to
each applicable value; set its error to sqrt(old variance + added variance).
Only particles with valid value AND positive valid error are smeared.
Conversion to a neutral removes charge and tracking applicability as in v2.

The v3 relative pT widths are 2% for charged particles, 6% for photons and
20% for neutral hadrons/unknowns. Use the same mean-one lognormal draws as v2;
the corresponding eta and phi Gaussian widths are 0.002, 0.006 and 0.020.
Preserve each constituent mass, wrap phi, then sum four-vectors for merges.
Topology is computed from original coordinates, NOT newly smeared coordinates.

For EVERY jet require exact equality to the saved v2 count, category, charge,
validity masks, key ordering and ancestry. Require all mechanism counts to
equal the parent per-file and aggregate counts. Fail closed on mismatch.
Physical smearing may differ; topology must not. Consequently this pilot's
mean count must equal its parent's exactly, not merely approach 38 in expectation.

## Diagnostics and acceptance

Keep complete moments, histogram tails/quantiles, PID counts, tracking validity,
kinematic and significance overlays and mechanism-selected examples. SD means
distribution width, not uncertainty. Compare paired jet pT ratios and axis
changes, plus single-parent surviving particle pT ratios, angular residuals,
tracking residuals and error ratios. Exclude merged descendants from the
single-parent diagnostics; include them in jet and inclusive distributions.
Directionless/empty jet axes have no defined angular response and are omitted
only from axis diagnostics; report their eligible counts. Do not require that
means shift: unbiased smearing can broaden residuals while preserving means.

Authenticate all parent spec/receipt/output hashes and exact training coverage
before reuse. Compare serial and spawn-process bytes, counts and ancestry for
up to 64 jets. Publish generated physical NPZ blocks separately from diagnostic
ancestry, all atomically, and publish a complete immutable receipt last.
Construction indices are forbidden classifier inputs and are not a substitute
for a declared particle matcher in subsequent ladder experiments.

## Execution and next decision

One Tigris job: 16 allocated CPUs, up to 16 spawn workers (one per input file),
one numerical thread per worker, 64 GiB, 2 hours, zero GPUs, account
`reu-aisocial`. Use the ARM `atlas_kd_tigris` environment. Require clean pushed
source, dry plan, exact hash authorization, site test-only validation and
duplicate/ambiguous-submission protection. Parent v1/v2 artifacts remain intact.

After this pilot, inspect response widths and physical sanity and either freeze
v2 or v3, or stop. Do not silently tune on validation/test or declare the
benchmark detector-qualified. A full 2.25M dataset requires a separate versioned
production plan, fixed split registry, measured resources/storage and explicit
authorization. Calibration stays frozen even if full-sample counts differ.

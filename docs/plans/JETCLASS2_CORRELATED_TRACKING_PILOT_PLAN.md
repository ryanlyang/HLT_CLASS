# Correlated tracking mechanism pilot

## Scope

Implement the approved 20,000-training-jet Tigris diagnostic, separate from
CONTEXT_V1 and NOISE_V3. This is a synthetic mechanism test, not CMS HLT or a
fit to CMS data. No classifier or production campaign is submitted by this
pilot. Poor diagnostic distributions are results, not grounds to drop jets.

Read original OFFLINE particles from the completed original literature pilot:
`/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_literature_pilot_08cbf063_r1/study_spec.json`.
Authenticate its spec, receipt, inventory/profile and physical banks using the
existing v2 parent reader. Do not apply the old NOMINAL response. The inherited
20k selection was label-stratified; this response reads no labels, raw ROOT,
native HLT, validation or test particles. Smaller authenticated parents are
allowed only for local integration tests, with their actual counts reported.

## Frozen response: CORRELATED_TRACKING_V1

Compare OFFLINE and six endpoints: CORR/INDEP at LOW, MID, HIGH amplitudes
`s=0.5,1,2`. Preserve p4 (GeV), PID, charge, keys, multiplicities, masks and
invalid placeholders exactly. No drops, merges, conversions, clipping or
kinematic smearing. Lengths are mm. These constants are synthetic hypotheses,
not measured resolutions and not chosen using classifier performance.

Use an explicitly assumed straight-line perigee coordinate convention:

```text
d0 = -(x-vx)*sin(phi) + (y-vy)*cos(phi)
dz = z-vz - ((x-vx)*cos(phi)+(y-vy)*sin(phi))*sinh(eta)
J0 = [sin(phi), -cos(phi), 0]
Jz = [cos(phi)*sinh(eta), sin(phi)*sinh(eta), -1]
tau = s * [0.020, 0.020, 0.050] mm
```

These define synthetic added perturbations to the stored fields; the historical
Delphes signed-d0/PV convention is NOT asserted to have been reverified. This
is not a helix refit or an event-level vertex simulation. No reliable event
grouping is assumed: the common reference perturbation is per **jet**.

Visible density is `rho=S/(1+S)`, with
`S=sum(j!=i) exp(-(deltaR_ij/0.1)^2)` in canonical particle-key order.
Set `q=0.25+0.25*rho+0.25/(1+pT/2 GeV)` and
`eps_sd = s*sqrt((q*offline_error)^2 + [0.002,0.005]^2)`.
Only valid value/error pairs are changed, independently per d0/dz component;
values without usable errors remain unchanged. No missing tracks are invented.

CORR draws one standard normal 3-vector per jet; INDEP draws one per particle.
Both use `delta_i = J_i @ (tau*z_i) + eps_sd_i*e_i`, with the SAME independent
two-component e_i draws across arms. Streams are SHA256/PCG64 keyed by jet
identity, then canonical particle-key ordering; all strengths share draws.
Keys are RNG/join metadata, never features. Reordering particles or changing
workers/files must not change an endpoint. No perturbation is redrawn per epoch.

`tracking_value' = tracking_value + delta` and
`error'^2 = error^2 + sum((J*tau)^2) + eps_sd^2`.
Within-track covariance is `J diag(tau^2) J^T + diag(eps_sd^2)` in BOTH arms;
cross-track covariance is `Ji diag(tau^2) Jj^T` in CORR and zero in INDEP.
Thus per-particle marginal distributions match in expectation, NOT sample by
sample. Standardized added residuals use the added variance, not the total
stored uncertainty. Error inflation is consistent with this added-noise model;
it does not establish calibration of the original offline uncertainties.

## Diagnostics and interpretation

Save physical banks for all seven sides with jet identity and offsets, but
no labels, latent shifts or oracle features. Report ordinary counts/PIDs, jet
kinematics, tracking values/errors/significances/masks, tails and moments.
Also report residuals, standardized residuals and their pt/density/PID slices.
For every jet/component with >=2 eligible tracks, report the average distinct
pair residual product and squared residual difference, alongside their analytic
expectations. These are equally jet-weighted pair summaries, not independent
particle samples or uncertainty estimates. Finite-sample marginal or covariance
disagreements are diagnostics, not pass/fail scientific quality thresholds.

Enforce structural identity, finite output and exact serial/spawn replay on up
to 64 jets. Record per-file times and process high-water RSS. CSV, overlay PDF,
mechanism PDF, JSON and receipt publish atomically; receipt last. One CPU-only
Tigris job: 16 CPUs, 64 GiB, 2h, account reu-aisocial. This is a requested
envelope, not a measured runtime prediction. No automatic followup.

## Next stage on SPORC (not implemented/submitted by this pilot)

The desired CE accuracy gap is approximately 1.5 **percentage points**, under
2 points. Diagnostics cannot measure it. After this pilot, use CE-only fits on
a separate train-only design holdout to choose among the three frozen strengths,
closest to 1.5 points, before seeing any KD comparisons. If none satisfies the
target, report that result rather than claim the target achieved or tune to a
ladder win. Freeze the selected recipe, use the same unclipped asinh/log1p
17-input frontend for every arm, and register a fresh SPORC experiment.

Proposed comparison: 100k training / 50k validation, paired seeds, OFFLINE and
HLT CE, direct KD, D066 -> D033 -> D000 coarse KD, and same-HLT repeated KD
with a matched training budget. With topology unchanged, U000/U050/U100 are
duplicates; do not spend fits on the ascent. Fresh initialization at each KD
stage remains the baseline; this is not weight continuation. No construction
keys, offline coordinates or latent vertex shifts at deployable inference.
Final test remains sealed. Require real GPU preflight, measured resources,
clean pushed source and a reviewed full plan before that campaign.

## Motivation, not calibration

CMS describes PV-constrained HLT tracking and sensitivity of impact-parameter
resolution to alignment in [DP2024/013](https://twiki.cern.ch/twiki/bin/view/CMSPublic/DP2024013).
Delphes [TrackSmearing](https://github.com/delphes/delphes/blob/master/modules/TrackSmearing.cc)
models impact-parameter resolutions. These support investigating tracking
response structure, not these constants or a claim that ladder KD will win.
The shared-versus-independent comparison is the falsifiable mechanism test.

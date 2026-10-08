# Frozen S3: direct versus coarse, 100k/50k

## Authority and scope

User authorized this comparison after the completed baseline screen selected
S3: OFFLINE accuracy 0.867480, S3 0.857000 (1.048 percentage points). This
explicit follow-up supersedes the screen's stopping boundary only for this
new campaign. Preserve the screen, original CORR_HIGH_TOPO dataset, sources,
reports and jobs. No new strength search or 2.25M export.

Require the authenticated completed screen from
`8abdc943c02d9844f98c91028f68e82e02b62fd4`, including all three full baseline
reports and its summary. Recompute and verify its registered choice is S3;
never pick a replacement based on KD results. This is exploratory reuse of
development validation, not independent confirmation or measured CMS HLT.

## Fixed comparison

Same exact 100k training / 50k validation identities, labels, 17-field
frontend, ParT, coordinate-paired initialization/sampler seeds, complete
60--100-pass recipe, checkpoint selection and CE25/KL75 at T=2 (one T squared).
All four fits start fresh; no weight continuation or extra U000 control.

- Reuse completed original OFFLINE and screen S3 CE controls, without refitting.
- Reuse the authenticated original OFFLINE train probability bank (T=2), with
  its original foundation/report provenance and an exact ordered identity join.
- Direct: OFFLINE -> S3_DIRECT_D000_from_OFFLINE.
- Coarse: OFFLINE -> S3_COARSE_D066_from_OFFLINE ->
  S3_COARSE_D033_from_D066 -> S3_COARSE_D000_from_D033.
- New reducers for D066 and D033 publish train-only probabilities. Never reuse
  old CORRHT intermediate models/banks: their endpoint was different.

## Intermediate views

Preserve the existing fixed-degraded-topology ladder convention. Generate the
unchanged S3 endpoint **once from the full frozen CORR_HIGH_TOPO context**, not
from an interpolated view. Every D rung keeps its final particle support, p4,
PID, charge and tracking-validity masks. No restoration of drops, merges,
neutralized PID, or erased measurements. Construction keys are used only to
join the authenticated original representative mapping; not model inputs.

For still-valid value/error pairs, interpolate original singleton tracking:
value = offline + f*(S3-offline), error^2 = offline_error^2 +
f^2*(S3_error^2-offline_error^2), with f=1/3 at D066 and 2/3 at D033.
Merged neutral measurements remain invalid. D000 is **exact unmodified S3**,
with no need for offline or mapping in the view function. OFFLINE remains the
original unmodified teacher. Intermediate validation is explicitly an oracle
diagnostic; final D000 deployable inference consumes HLT alone.

This convention does not promise that KD can recover erased information or
that coarse will win. It tests the same descending tracking path on a stronger
fixed endpoint. All outcomes are retained regardless of performance.

## Execution

New isolated namespace/package; every inherited scientific source file must
remain byte-identical. No matching or physical export: bounded ephemeral RAM
caches. A genuine SPORC debug A100 preflight under the new source is required:
installed-Weaver parity, exact serial/process replay of all three D views,
full train/validation D000 digest equality to the completed S3 baseline,
full-size D066 KD acceptance with the real OFFLINE bank, reducer timing and
GPU/RAM checks. This site-specific gate is authoritative for this SPORC
experiment; generic Tigris validation is not claimed or transferred.

Gate initially requests 2h/1 A100/6 CPUs/90000 MiB. Separately dry-review and
authorize seven science jobs: four fits, two reducers, CPU summary. Direct and
coarse D066 have no mutual dependency. Measured fit/reducer requests must fit
debug's 24h limit. No automatic submission, cancellation, resume, or expansion.
Clean pushed source, exact plan hash, durable claims and journals are mandatory.

Report raw accuracy/AUC/per-class QCD rejection and recovery relative to S3=0%
and OFFLINE=100%, plus coarse-final minus direct-final raw differences. No
statistical significance or compute-matched superiority claim from one seed;
coarse has a larger total training budget. Final-test inference stays sealed.

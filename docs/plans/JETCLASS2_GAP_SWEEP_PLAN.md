# CORR_HIGH_TOPO baseline-gap screen

## Scope and decision

User authorizes a fast, bounded follow-up on 2026-10-07: increase synthetic
HLT degradation until an independently trained HLT baseline is more than one
percentage point below OFFLINE. This supersedes the old plan's prohibition
on extra fits only for this separately named screen; the original campaign,
source, datasets and running jobs remain immutable.

Reuse the exact CORR_HIGH_TOPO 100k train / 50k validation population and
authenticated OFFLINE and M0HLT reports from source
`8c5b47029ed15f88aa852b6f38d06e98bbbfcd9b`. Do not load KD results for selection.
Keep the same 17-input transform, model, complete CE training schedule,
checkpoint selection, and D000 initialization/sampler seeds. No short training
run is a scientific baseline. No additional OFFLINE fit or teacher reducer.

## Frozen candidates (not measured CMS response)

Three ordered strengths S1=0.5, S2=1.0, S3=1.5 post-process the saved
CORR_HIGH_TOPO endpoint. Its previous merges and correlated response remain.
Compute context once, before any additional loss: c is nearest-neighbour
crowding within 0.08 rad; d=tanh((log1p(abs(d0)/0.2 mm)+
log1p(abs(dz)/0.5 mm))/4), with missing values zero; q=(c+d)/2.

Using common identity-keyed independent streams across the strengths:

- Drop original-to-this-stage CH/NH/photon tokens with pT<10 GeV with
  probability s*0.10*(0.5+0.5*q)*(1-pT/10).
- Convert surviving charged hadrons to neutral hadrons with probability
  s*(0.08+0.16*q), clearing charge and all tracking validity/values/errors.
- Independently remove all tracking measurements from surviving, unconverted
  charged particles with probability s*(0.15+0.30*q). PID remains unchanged.
- Retain p4 of surviving tokens exactly; no new merging, p4 rescaling,
  clipping, invented valid errors, or empty-jet rescue. Invalid tracking is zero.

These changes intentionally remove information, not just invertibly encode it.
Probabilities remain below one for all registered strengths. Decisions are
nested with common draws; accuracy is NOT assumed to be monotonic. Inputs to
the generator contain no labels, teacher predictions, or role identifiers.
Construction keys, probabilities and context are never classifier features.

## Fast execution and scientific limits

No new durable 150k or 2.25M particle export or foundation matching. Generate
the candidate tensors in bounded process-local RAM from authenticated existing
particles. Save train-only descriptive diagnostics, endpoint digests, selected
weights, and full baseline reports. Parent files remain required for replay.

Separate preflight stage: genuine installed-Weaver parity, full-population S1
(largest retained population) CE acceptance pass and measured resource checks;
serial/process exact replay for ALL candidates on selected training rows.
Candidate caches cannot exceed the parent particle count or capacity. A
separately reviewed science plan then runs three independent GPU fits and a
CPU summary. No artificial chain between fits. Same SPORC debug A100 environment;
actual concurrency depends on allocation. No automatic KD jobs or new sweep.
The technical preflight requests 2h (the supplied original preflight took
9m58s); this is an allocation ceiling, not a runtime guarantee. Scientific
fits keep the full schedule and separately measured walltime requests.

Selection only after ALL three valid completed baseline reports: choose the
mildest registered strength with 0.01 < (OFFLINE accuracy - HLT accuracy) <=
0.02. If none qualifies, record `no_candidate_in_band` with no winner and exit
successfully. Poor accuracy never fails or skips a registered fit. Raw accuracy,
AUC and per-class QCD rejection are retained for every candidate. Single seed,
checkpoint/strength selection on reused development validation: exploratory,
not an unbiased confirmation or a guaranteed population gap. No final-test
access. Fresh held-out confirmation and any full dataset/ladder are separate
authorizations; this workflow stops after the development choice.

# B_DZ joint tracking repair: one bounded development round

## Authority, evidence and stop rule

Ryan authorized this separate study on 2026-09-27 after the frozen significance
audit. The downloaded report's content hash is
`e5c6830b2cc694dc338e0c464d75a57f63a59232608d774941b25224cb1b6177`.
Independent marginal maps amplified one charged-hadron dz by about 52x while
its error grew only 16%; unsupported muons inherited pooled corrections.
These examples motivate a test, not proof that all tails are unphysical.

This plan supersedes the audit's stop rule ONLY for this new, isolated study.
It does not amend its frozen winner, models or reports. Exactly two new
candidates, no grid, no recursive optimization, no confirmation or JetClass2
particle access. Stop after the development report, including if neither helps.
This is not full detector-response qualification. Count, p4, charged fraction,
association coverage and provisional producer conventions remain limitations.

## Frozen inputs and candidates

Require a completed original or debug-replacement B_DZ significance audit and
authenticate its report, shard receipts, original comparison, model, maps,
samples and source. Donor scientific files must be byte-identical; only the
existing execution-only allowlist can differ. Use a disjoint root.
Keep B fit, dz scale, shared physical interface, random keys, three replicas,
particle order/keys, PID, charge, p4, validity and missing placeholders fixed.
Generation consumes offline only; real HLT is a target, never generator input.

Five candidates, in deterministic tie order:

1. B_DZ: exact frozen base.
2. TRACK_HALF: exact historical half-strength marginal map.
3. TRACK_FULL: exact historical full-strength marginal map.
4. PID_SAFE: full marginal map only for coordinates with their OWN fitted PID
   cell. Unsupported coordinates remain byte-identical to B_DZ, never pooled.
5. JOINT: PID-specific error-conditional significance transport below. Starts
   from B_DZ, not TRACK_FULL or PID_SAFE. Unsupported pairs remain B_DZ.

All three controls replay their previous histograms, tracking, corrections,
flags and identities before publishing. Nontracking equality is checked for
every candidate/jet/replica. A poor diagnostic is a result, not a failed job.

## Joint map (fully specified)

Use only the same 4,000 training-residual calibration jets. Real HLT enters
once; three B_DZ replicas enter the proxy pool. These are previously used fit
residuals, not fresh validation. Arrays stay in RAM with a 2 GiB raw-array cap;
exceeding it fails, never subsamples. Count UNIQUE jets, not particles/replicas.
Quantiles are particle-weighted within each cell; unique-jet counts are the
support gate, not weights. This preserves the earlier marginal calibration
weighting and does not imply equal per-jet multiplicity weighting.

For each of six PIDs and each (d0,d0err)/(dz,dzerr) pair:

- Use complete, valid pairs only. Source and target each require 1,000 unique
  contributing jets. No cross-PID fallback and no pooled map.
- Fit the existing fixed-quantile monotone log-error transport on these pairs.
  Quantile knots, repeated-knot averaging and constant endpoint extension are
  identical to the historical map. Positive errors stay positive.
- Separately compute source/target log-error quartile boundaries at .25/.5/.75.
  Assign with searchsorted(side=right), including ties; no jitter or invented
  within-atom spread. Degenerate SOURCE error support makes the whole pair identity.
- For each corresponding error-rank bin, fit a monotone asinh(significance)
  transport with the historical quantile knots. Each side must have >=500
  unique contributing jets of NONZERO displacement in that bin. Degenerate
  source significance support gives identity for that bin. Repeated quantile
  knots otherwise collapse by mean target, as above. Disclose support.
- At application, find the SOURCE bin using the ORIGINAL B_DZ error. For a
  supported nonzero pair, map error and significance, then reconstruct
  displacement = mapped_error * mapped_significance. Never independently map
  displacement. An unsupported bin leaves BOTH value and error unchanged.
- Exact zero displacement stays zero; if its PID error map is supported, map
  its error only. A value without a valid error stays unchanged. Invalid
  placeholders stay zero. No clipping of observed CMS data or output ratios.
- Endpoint extension is explicit and counted. This finite-resolution conditional
  map does not guarantee exact joint closure or repair missing sign/magnitude
  dependence within bins; report these limitations and all poor outcomes.

Publish maps with support counts, identity reasons, units, sample/model hashes.
Evaluation never fits or selects a map using real HLT for an individual jet.

## Evaluation and decision

Same 10,000 repeatedly used development jets, four fixed 2,500-jet shards,
three dependent replicas, all jets regardless of association quality.
Retain the previous full conditional histograms and six-field tail/correlation
score. Also retain frozen audit PID moments, significance, value/error heatmaps,
tail squared moments, uncertainty-conditioned moments and bounded top-five
hashed examples, with accurate before/after routes. Report all six PIDs.

New registered development score = mean(old three-block score, PID score).
PID score equally averages marginal TV, balanced tail discrepancy, and joint
value/error histogram TV, averaged equally across eligible PIDs and replicas.
Eligibility requires >=50 unique contributing jets on BOTH sides for the
particular coordinate/pair in EACH replica; real and proxy counts are reported.
No eligible cells contributes score 1 and an explicit unavailable flag, not a
claim of closure. Rare/empty groups still appear in diagnostics, not hidden.
Tail discrepancy uses the existing thresholds and denominator
`p_real+p_proxy+1/n_real`. This is an exploratory objective, not the old score.

Reject a candidate for selection (not execution) if any old per-variable TV,
tail score or joint correlation block exceeds B_DZ by >.02, or any eligible
PID/variable TV, tail score or pair joint-TV exceeds B_DZ by >.02. Eligibility
is identical across candidates because validity/PID/support is invariant.
Choose lowest eligible score, ties in the order above; B_DZ is always eligible.
Preserve the historical TRACK_FULL choice in a separate field. Publish all
scores, rejected rows, guard reasons and leave-one-source-file-out sensitivity.
Do not present three-file intervals or dependent replicas as independent evidence.

## Execution and safety

Two separately created/authorized CPU-only stages, default debug; tier3 can be
chosen at CREATION for the whole fresh study. Frozen partition cannot be edited
in-place. No cancellation or mutation of another campaign is provided.

- joint_gate: bj_acceptance (2 CPUs/32 GiB/2h) then bj_calibrate
  (36 CPUs/128 GiB/8h). Acceptance uses 32 existing residual jets, actual frozen
  controls and temporary nonidentity joint probes; measures serial/process
  replay and memory/time. Probe support is synthetic, explicitly not science.
- joint_compare: four bj_eval_0..3 (36 CPUs/128 GiB/8h) then bj_select
  (1 CPU/32 GiB/4h). Creation requires completed gate/map receipts and acceptable
  measured resource envelope. No automatic advance after gate or selection.

Bounded spawn workers, eight-jet chunks, at most twice worker count outstanding,
one native numerical thread per worker, canonical ordered merge, 15s heartbeat.
No speedup assumption in conservative runtime projection. Source pinned, clean,
pushed; exact plan hash and stage phrase for live submission. Immutable hashes,
receipts/claims and scheduler journals are reused. No durable particle banks.
Queue helper defaults dry and retains errors, stdin and exit status correctly.

## Tests and handoff

Hand-calculated joint ratios, ties/zeros/masks, PID fallbacks, unique-jet support,
endpoint accounting, serial/spawn parity, malformed maps and nonfinite refusal;
synthetic ROOT through donor authentication, calibration, all shards/report;
historical replay, corruption/access/source guards; CPU-only dry/live dependency
and idempotency tests. Local fixtures do not establish real CMS improvement.
Genuine SPORC acceptance remains mandatory. Weaver/GPU parity is not applicable.

# Frozen B_DZ tracking-significance replay

## Scope and stop rule

Ryan authorized this next diagnostic on 2026-09-26 after the downloaded
TRACK_FULL report was inspected. This is an additive development-only replay,
not a correction, refit, new winner, or production qualification. Preserve the
completed B_DZ / TRACK_HALF / TRACK_FULL comparison and its selected winner.
Stop after the report. Any joint/PID-aware repair needs a separately frozen
candidate protocol; confirmation and JetClass2 transfer are not authorized here.

The motivating observations are inclusive significance-width inflation and a
pooled muon dz correction that can offset an underspread charged-hadron dz
distribution. They do not establish that the extreme significances are muons,
that the observations are invalid, or that matching is broken.

## Frozen replay

Import a completed original BDZ_STAGE comparison or completed BDZ_TIER3
replacement. Authenticate its calibration, selection, four shard receipts,
source, numerical environment, preparation and sample membership. A tier3
parent additionally needs its immutable retirement evidence. Compare all donor
scientific source files byte for byte; only the existing execution allowlist
may differ. New audit code must be source-pinned and clean on RC.

Use exactly the same 10,000 development jets, four fixed 2,500-jet shards,
three label-independent keyed replicas, B model, dz scale, tracking map and
candidate strengths. Reuse the ordinary evaluation-only reader. No association
refit or calibration particles are needed. Include every registered jet,
regardless of diagnostic quality. Generator inputs remain offline-only.

Every shard must reproduce the previous ordered identity digest, source groups,
histograms, tracking accumulators, generator flags and correction counters.
Integer counts/bins/identities must match exactly; historical floating moments
use the existing 1e-10 relative/absolute replay tolerance. Model/map outputs
are never altered. Invalid, missing, stale, or corrupted lineage fails closed.

## Additional diagnostics (no selection score)

Keep real CMS HLT once per jet, and each generated replica separately. Stratify
by source file and all six physical PID categories plus inclusive `all`.
Publish all groups, including empty/rare groups, without a significance claim.

For d0, dz, d0err, dzerr and both significances publish valid-particle counts,
unique contributing jets per side, physical mean/SD/extrema, signed-asinh
histograms for values/significances and log histograms for positive errors.
Use fixed diagnostic bins, not fitted bins. Include explicit under/overflow.
Count absolute exceedances and their sum-of-squares contributions at the
previous tracking-screen thresholds. SD is distribution width, not uncertainty.

For each value/error pair publish all four validity combinations (including
valid value without valid error), a joint histogram of log10 absolute value
versus log10 error, and complete-pair correlations in both signed-asinh and
log1p-absolute-value versus log-error coordinates. Zero values occupy a
separate underflow bin in log10 magnitude; no epsilon is substituted. Add
significance moments conditional on fixed error bins. Include complete-case
and marginal denominators; do not interpret signed correlation alone as a
test of magnitude/error dependence.

Retain at most five largest absolute significances per coordinate, side, PID
and source file, with stable identity-based tie breaking. Each bounded forensic
record contains hashed jet/particle identities, all four physical tracking
values/masks, pT, pre-map values, post-map significance, and the exact
own/pooled/identity map route for the value and error. These small excerpts
are diagnostic evidence, not a durable full particle bank, classifier input,
training target or a particle correspondence between real and generated data.
Do not discard, clip or modify any extreme observation.

Report PID-specific histogram TV and tail rates separately per replica, plus
pooled proxy moments with the dependent-replica warning. Report the number
of valid jet-replica exposures rather than claiming pooled unique-jet support;
unique contributing jets remain available separately for each side/replica.
Report the fraction
of significance sum-of-squares from each PID and from the bounded examples.
Publish displacement/error heatmaps. No uncertainty interval from only three
files, automatic recommendation, or new model selection is made.

## Execution

One isolated `bdz_audit_r1` stage, CPU-only tier3 (current preferred partition),
account reu-aisocial, qos_tier3, atlas_kd_sporc. No old job changes or cleanup.

- `ba_acceptance`: 2 CPUs, 32 GiB, 2h. Replay 32 existing development jets
  serially and with two spawn workers, including actual frozen maps and new
  accumulators. Record measured memory/time and conservative projections.
- `ba_eval_0..3`: each 36 CPUs, 128 GiB, 8h, afterok acceptance. Bounded
  eight-jet batches, at most twice worker count outstanding, one native thread
  per worker, ordered merge, progress at least every 15 seconds while waiting.
- `ba_report`: 1 CPU, 32 GiB, 4h, afterok all four new shards. No new raw reads.

New metrics increase work, so old runtime evidence is not sufficient alone:
the real in-DAG acceptance must pass before the full replay executes. A poor
physics result still completes; only integrity/resource failures stop work.
Create and dry review before explicit live phrase/plan-hash submission. Do not
commit/push/submit on the user's behalf. Queue helper defaults to dry review.

## Verification

Test hand-calculated ratios/masks, zero values, muon fallback attribution,
tail moment fractions, stable bounded extrema, per-replica and unique-jet
denominators, merge invariance, nonfinite refusal, and serial/process replay.
Exercise real synthetic ROOT data through authenticated donor reuse, all
shards/report, corrupted replay and source rejection, exact CPU dependencies,
submission authorization/idempotency and old artifact preservation. Local
tests do not certify real CMS response quality or remote resource acceptance.

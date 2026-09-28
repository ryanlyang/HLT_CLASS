# Frozen JOINT: one-shot CMS collection-response confirmation

## Authority and finite endpoint

Ryan authorized this separate campaign on 2026-09-28 after completed JOINT
development jobs 21788038--21788042. This supersedes that study's stop rule
only to permit this confirmation, not further calibration or candidate search.
Freeze its selected JOINT model/map and B_DZ reference, exact feature interface,
histogram ranges, random keys and replicas 0/1/2. Authenticate the completed
comparison, maps, model, receipts, source and numerical environment. No refit,
reselection, automatic repair, JetClass2 access, classifier training or final
test read. Other campaigns/artifacts/jobs are unchanged.

This is a **collection-response confirmation**, not fulfillment of the earlier
three-family production protocol. In particular, it retains three replicas and
the frozen development histograms rather than claiming the production five-
replica six-block Wasserstein/association test was performed. The original
99% association target, producer-convention qualification, class-conditioned
studies and JetClass2 support/transfer requirements are not waived. A positive
result supports the frozen CMS proxy for further review; it does not certify
true detector HLT or automatically publish a production/transfer lock.

## Data and access lock

Use ALL eligible rows in ALL authenticated original-training files assigned
`response_confirm` by the existing preparation. Hard minimum 250,000 jets;
fail on insufficient capacity, never borrow fit/select/original-test files or
silently lower the budget. This is untouched by this response development, not
necessarily by historical classifiers. Require the same source-category set
as fit, but retain the natural confirmation mixture and disclose per-file and
per-source counts. No labels become generator inputs. Unknown cross-file event
dependence remains a limitation; event_no is not a global event identity.

Membership is frozen from metadata without reading particles. Sort paths, sort
eligible entries, split each file into consecutive groups of at most 5,000
eligible rows. A shard contains one file: streaming and source authentication
do not repeatedly scan every confirmation file for every shard. All rows are
covered exactly once; ranges, ordered identity digests and file hashes are
checked. Before opening a confirmation file, require the published immutable
selection/membership/protocol lock AND the exact worker's execution claim.
The CLI requires an explicit assertion that this response_confirm population
has not previously been inspected for this mapping. It cannot prove absence
of reads in other programs; this assertion and its limits are recorded.

## Registered checks (before fresh particle access)

No aggregate score can hide failure of an individual check. All checks apply
to JOINT, with B_DZ shown as a reference, not as a second selectable finalist.
Report full moments/histograms, all fixed conditional cohorts, PID tracking
tails/joint distributions, validity rates, generator flags and map routes.
SD means distribution width; proxy replicas are dependent exposures.

Checks use the mean of the three separately computed replica statistics:

- Mean relative bias <=5%: jet multiplicity, vector pT, mass (zero reference
  is unavailable, never divided by an epsilon).
- Overall histogram TV <=0.10: those three jet variables, jet width, charged
  fraction, paired jet-pT ratio and axis delta-R, particle pT and energy, all
  four tracking coordinates and both significances.
- The same TV checks <=0.25 in every fixed non-`all` cohort with >=1,000
  contributing real AND proxy jets in every replica. Sparse cells are disclosed.
- Overall tracking-validity TV <=0.05 for each of four validity flags.
- Each populated PID tracking-coordinate/significance TV <=0.25, with >=1,000
  contributing jets on both sides in every replica. Rare cells remain reported.
- Paired jet-pT-ratio SD ratio must be in [0.75,1.25]; all six tracking
  coordinate/significance SD ratios must be in [0.5,2]. These deliberately
  inspect widths as well as central bins; they are engineering tolerances,
  not CMS-certified targets and are not relaxed to accommodate existing bias.
- Each defined pairwise jet-summary Pearson correlation discrepancy <=0.10.
  Undefined correlations are disclosed as unavailable, not zero disagreement.

For every registered check publish its point value and a paired source-file
bootstrap 95% interval: 200 draws, seed 20260928, uniformly resample whole
files with replacement, keep real/proxy/candidates/replicas together. At least
four contributing files are required for an interval usable for acceptance.
Files, not shards or replicas, are the resampling units. Small file counts are
explicitly warned about; these intervals are not a proof of event independence
or simultaneous coverage. A required unavailable cell cannot count as a pass.

A check is supported only if its entire interval is inside the tolerance;
rejected only if entirely outside; otherwise inconclusive. With insufficient
groups use point failures as rejection evidence but never grant support.
Overall: any rejection -> `rejected`; otherwise all checks supported ->
`supported`; otherwise `inconclusive`. Publish every outcome successfully.
No switch to B_DZ, no map adjustment and no automatic downstream access.

The current development jet-pT response width (~2.66 times CMS), muon response
and displacement tails are known concerns. Reproducing them is not a pass.

## Execution, storage and partition choice

New isolated root and pushed clean source. Two separately authorized stages:

1. `frozen_gate`: `jf_acceptance`, 8 CPUs/32 GiB/2h; 64 existing residual jets,
   no confirmation reads. Serial/process parity of the actual frozen candidate
   and reference, new reduced two-candidate worker, time/RSS/output projection.
2. `frozen_confirm`: one `jf_eval_NNNN` per membership shard, 16 CPUs/64 GiB/4h,
   then `jf_report`, 1 CPU/32 GiB/4h. Creation requires successful real acceptance.
   Eight independent dependency lanes cap concurrent evaluations at 128 CPUs.
   Every shard still runs when earlier results are scientifically poor.

Eight-jet chunks, <=2*workers pending chunks, spawn processes, native thread
limits one, canonical ordered merges and 15s progress heartbeats. Frozen model
initialization occurs once per child. No durable particle banks. Aggregate
shards sequentially. Project disk demand from acceptance with a safety factor;
require free space before confirmation creation, never delete prior artifacts.
The measured projection must also fit the inherited 12-GiB total and 2-GiB
reports/examples caps including existing files, with publication headroom of
twice projected remaining writes plus 5 GiB. Exceeding an envelope stops before
confirmation creation; it does not silently discard diagnostics.

`probe-frozen-joint` performs ONLY `sbatch --test-only` for both debug and tier3
using these real CPU/memory/time shapes, no GPU. Show outputs; prefer the earlier
evaluation estimate, tie to tier3, unknown estimates do not justify a promise.
The helper resolves `auto` only for a new root; an existing root stays on its
frozen partition. One partition per study, no in-place migration, cancellation
or resubmission. Estimates are transient single-job hints, not total-campaign
completion guarantees. Fresh gate and confirmation need exact dry-plan hashes
and explicit live authorization. SSH-safe helper log/exit status supported.

## Verification

Frozen controls and source unchanged; local numerical/parallel tests,
metadata-only disjoint membership with uneven shards, capacity/forbidden-role
refusal, claim-before-read, source/maps/receipt corruption, statistics and
file-bootstrap tests, failed scientific confirmation still publishes,
dependency-lane/resource/probe tests, and synthetic ROOT end-to-end execution.
Real CPU acceptance remains a separate SPORC gate; GPU/Weaver parity is not
applicable. Stop after report, including rejection or inconclusive findings.

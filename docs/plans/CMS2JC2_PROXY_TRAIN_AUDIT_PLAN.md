# Committed proxy training-population audit

## Scope and question

Inspect all 1,000,000 registered TRAIN_1M jets in the saved dzfix offline to
CMS-calibrated JOINT proxy dataset. Compare saved proxy particles with their
exact offline source rows. Reuse the frozen CMS confirmation report's real
offline and HLT histograms, without opening CMS particles. This is descriptive
diagnostics, not a new mapping fit, classifier result, or production qualification.

Validation and final-test particle blocks are forbidden. Do not require or
publish the full dataset manifest: an independent audit binds all committed
training receipts before work begins. Missing training receipts fail closed;
unfinished validation/test generation does not block this audit. The original
generation campaign, files, locks, queue, and scientific selection are untouched.

## Exact comparisons

Reuse `dev_diagnostics` observables, validity masks, offline-only cohorts, and
the original CMS-fitted bin edges. Common bridge units are GeV and mm; do not
convert already bridged values again. Record raw count, covered jets, mean,
distribution SD, min/max, and explicit histogram under/overflow. SD is not a
standard error. Jet statistics are jet-weighted; particle statistics are
particle-weighted over valid observations, with the valid sample count shown.

Produce four labeled sides: CMS offline, CMS HLT, JetClass2 offline, and saved
JetClass2 proxy-HLT (one physical replica). Never pool the three CMS proxy
replicas into the real reference. Report marginal TV distances and per-dataset
offline-to-HLT mean shifts; these are different populations, not paired across
datasets. Conditional tables use the existing coarse offline cohorts, not
reweighting or exact kinematic matching. Poor agreement is a result, not a job
failure. Preserve the reduced CMS confirmation coverage/status in the report.

Additional exact paired JetClass2 count diagnostics: per-jet multiplicities,
HLT-minus-offline multiplicity and each PID count change, and fractions gaining,
losing, or retaining count. Integer-histogram quantiles are exact order
statistics. CMS saved marginal histograms cannot establish paired count-change
distributions; show only differences of means there. No class labels are read.

## Execution and artifacts

Use a separate clean, pushed worktree and new audit root outside both data
roots. Freeze source, dataset study, complete ordered training receipt list,
CMS report, and ranges by hashes. Physical checks and analysis occur on a
Tigris CPU job, never on the login node. One job requests eight CPUs, 64 GiB,
eight hours and no GPU, with at least 2 GiB free in the audit parent directory.
This is an initial diagnostic allocation, not a measured
full-population runtime claim. A spawned process pool processes train shards;
each retains at most a physical block and bounded ROOT read window. Publish
immutable per-shard histograms, then merge in registered order. Completed
authenticated shard diagnostics are reusable after an interrupted audit.

Verify physical block hashes/schema, exact source row identities/order, counts,
and physical digest. Reuse the production offline-only reader (no native HLT or
labels). Each process reports progress and timing. Report output includes JSON,
CSV, a readable table, and multipage PDF distribution plots with overflow
disclosure. A receipt authenticates every final artifact. No automatic job
cancellation, downstream training submission, or model modification.

## Acceptance

Focused synthetic tests must cover full train membership, no validation/test
reads, paired identity mismatch, corrupted blocks, raw moments/TV, count-change
quantiles, reference scope, immutable artifacts and CPU-only command shape.
Existing production tests must continue passing. The first real audit is the
remote validation of this new diagnostic worker; local tests do not claim it
has already run on Tigris.

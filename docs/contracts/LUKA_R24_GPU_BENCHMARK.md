# R24 GPU benchmark contract v1

Authority: [active plan](../plans/LUKA_R24_GPU_BENCHMARK_PLAN.md).
Namespace `LUKA_R24_GPU_BENCHMARK_{SPEC,PLAN,CLAIM,BATCHES,PARITY,PHASE,REPORT,RECEIPT}/v1`.
Each JSON carries schema_version1, content_hash, parent hashes and inherited
exploratory-development flags. No artifact is scientific campaign acceptance.

SPEC pins two independent sources: the new clean, pushed benchmark worktree
and the original clean `58d0c0edf5f41cff2398fb168fc16fc92e169294` R8 execution.
The CLI imports scientific helpers only from the latter. Original spec,
completed setup/R24 receipts, full native R24 fit and submission ledger are
authenticated before reuse. Original files are read only; no retargeting,
checkpoint replacement or original job edits. Environment must equal the
original reference and interpreter prefix. Registration requires a fresh root
separate from original sources, data and output, two GiB output allowance and
one GiB reserve. Benchmark source checks require all implementation files.

PLAN requests exactly one node, one task, twelve CPUs,64000MiB,two hours,
four `geforce_gtx_1080_ti` GPUs. Dry scheduler review precedes exact-hash live
submission using the reusable journal. Ambiguous submissions are preserved,
never retried. Worker validates exact accepted job/controller allocation and
four actual Pascal6.1 cards. Only bounded read-only controller queries retry.

BATCHES pins complete TRAIN targets and paired U100 embeddings by checksummed
NPZ, with rows sampled across the original epoch-one permutation. OFFLINE
values/validity, normalization, frames and aggregates stay outside native
forward. Native inputs remain features/vectors/mask. Validation files contain
only those three fields. Global padding and global batch128 are preserved;
small partial/stress batches and original fixed decoder support are explicit.

PARITY authenticates three FP32 single-card snapshots with dropout disabled,
native training BatchNorm and original AdamW. Cases15/16/15 rows use R24
schedule positions3/4.5/7. PHASE compares 2/3/4-card SyncBatchNorm, sample-weighted
SUM gradients, parameters, buffers and moments. Decoder chunk8/VJP/matching
semantics are reused. No DDP gradient averaging is layered on top. Solver
workers use floor(12/world) per process. Each full-precision phase preserves
the original LR and losses; this is not an A6000 stochastic-trajectory replay.

PHASE saves all measured update and validation durations, diagnostic costs,
per-rank peak allocation/reservation/RSS, stress results and parity metrics.
Parity failure marks qualified=false; poor speed is valid evidence. Process
failure/OOM leaves exit codes and immutable logs in REPORT; subsequent world
sizes are attempted in fresh groups. Timed-out child sessions and descendants
are terminated before the next phase. No scientific fitting is released.

REPORT projections include thirty training epochs, thirty full validation
passes, measured input construction, diagnostic and startup estimates.
The startup estimate includes benchmark preparation/publication and is
conservative relative to a production worker that does not publish samples.
Per-rank diagnostic shuffles measure a possible parallel diagnostic's cost;
they are not the original global-shuffle scientific probe. Validation timing
includes native softmax and local CPU transfer, but excludes global metric
calculation. Queue wait, checkpoint I/O and extra audits are excluded. A6000
training/validation sums and optional exact-job Slurm elapsed are reported
separately. Active GPU-hour estimates differ from this job's four-card
reservation cost. Summed rank RSS peaks are conservative, not simultaneous
physical-memory occupancy. No final-test access, AUC evaluation, quality gate,
automatic retry or science acceptance exists.

RECEIPT binds the report, logs, batch manifest, phase reports and numerical
reference snapshots to the accepted job. Saved results only are read by the
results command. Hash/source/lineage corruption fails closed.

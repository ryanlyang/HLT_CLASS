# Batch-128 monitoring repair and report-only recovery

Authorized 2026-10-16: repair the disappearing-process sampler race and recover
the completed probes of SPORC job 21898881, source
922bd972ef9ec52079e9e750e6e7896e857aba8b. This plan supersedes ONLY the successful
GPU-coordinator-job requirement of the batch128 plan for this specific recovery.
Batch128, all 17 fresh fits / 30 science jobs, split, full support, seeds, model,
losses, optimizer, 60--100 passes and all admission thresholds remain unchanged.

The sampler tolerates ENOENT/ESRCH for disappearing children/threads. Failure to
read its own process, permission errors, malformed statistics and other errors
remain fatal. No broad OSError catch and no invented measurements.

Preserve the original source, artifacts and FAILED/1:0 accounting. Run the
original validators in their original pinned worktree. Require all 12 child
results, matching logged result objects in canonical execution order, successful
coordinator progression through the loop, and the exact deferred ProcessLookupError
at Measurement.__exit__. Authenticate all source, cache metadata, logs, outputs,
selected checkpoints and technical TRAIN-bank hashes and lineage. Never replay
GPU work, decode raw particles, create training banks, or submit science during
recovery. Technical predictions are never scientific teacher targets.

Because the coordinator's RSS sampler stopped early, do not manufacture its
missing peak or use a lone child peak as a total. Capture live sacct evidence for
the exact failed job and its single-node, single-task batch step. Require valid
nonzero MaxRSS, matching name/partition/account/CPU/memory and FAILED/1:0. Use
batch MaxRSS PLUS the largest successful child's measured tree high-water as
a deliberately double-counted conservative replacement estimate. This is sampled
evidence, not a proof of the exact instantaneous peak; preserve that caveat.
For single-task Slurm step accounting, MaxRSS represents the step peak:
[Slurm accounting](https://slurm.schedmd.com/accounting.html).
Retain the 85% host / 90% GPU checks and 1.75 * 100-pass / debug24h projection.
Missing evidence or excessive requests blocks science without changing schedule.

New GATE/SPEC/PREFLIGHT and execution artifacts v5 bind original v4 GPU evidence
separately from the new recovery source and CPU job. One CPU-only report job
(1 CPU, 32 GiB, 4h, SPORC debug) authenticates and publishes recovery evidence.
Only its own COMPLETED/0:0 job plus an admitted report can authorize a separately
reviewed 30-job science plan. The original GPU job remains FAILED; no success
receipt is inserted into it. Execution source compatibility permits only the
sampler fix, named version dispatch, and recovery modules; models/data/recipe
remain byte-identical. No commit, push, submission, cancellation or retry is
authorized by local implementation alone.

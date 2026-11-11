# GTX 1080 Ti benchmark for the small-data subclass campaign

2026-11-10. The user requested a benchmark on idle older Hydra nodes, including
two GPUs per run. This plan authorizes one bounded technical allocation. The
[scientific subclass plan](LUKA_SUBCLASS_HYDRA_PLAN.md) continues to govern the
ten 30-epoch fits. The benchmark neither launches nor qualifies those fits.

## Allocation and comparison

Request one node, two `gpu:geforce_gtx_1080_ti` devices, six CPUs, 32000 MiB RAM
and a two-hour limit, in Hydra `gpus`, account `default`, QOS `normal`. Use the
existing pinned teacher-sweep environment and a fresh exact-source worktree and
output root. Do not pin a node: Slurm chooses an eligible 1080 Ti node. The
user's snapshot showed several idle nodes, but availability can change.

Within that allocation:

1. Authenticate the existing small-data T00 and U100 artifacts. Rebuild the
   subclass bank from all 299234 TRAIN rows using the unchanged scientific
   target recipe. Do not fit anything on validation.
2. Publish identical, authenticated padded batches for both measurements.
3. Run a single-GPU FP32 phase, leaving the other allocated GPU unused.
4. Exit that process, then run a two-GPU FP32 phase with one process per card.
5. Publish timings, memory, numerical checks and an explicit benchmark report.

There is one Slurm job, no release job, no fit dependencies and no automatic
retry. Subprocess separation releases full preparation caches before training
and resets CUDA allocations between phases. Login logout is safe after the
accepted job ID has been journaled.

## Training semantics

Reuse the native ParT, auxiliary heads, original initialization, AdamW parameter
groups and actual `CE + 5*U100 SmoothL1 + lambda*subclass KL` implementation.
Measure `S0p5` and `S100`, each with fresh weights. The benchmark holds the LR
and subclass ramp at schedule position 3, where the ramp is complete; this is
a short technical probe, not an epoch or learning-curve experiment.

The global minibatch remains 128: one card receives 128 jets; two receive 64
each. FP32 is explicit in both phases. The existing science executions use
BF16; these timings are not evidence of numerical equivalence to those runs.
Pascal is not silently substituted into a BF16 production execution.

Two-card execution uses PyTorch DistributedDataParallel and SyncBatchNorm.
Padding is computed for the global batch before splitting the jet axis, so
input and pairwise BatchNorm see the same global particle support. Uneven
partial batches weight each local mean by `local_rows * world_size / global_rows`
before DDP averages gradients. Each rank contributes every trainable parameter.
Targets, labels and identities never become native model forward inputs.

The framework mechanisms are documented in [PyTorch DDP](https://docs.pytorch.org/docs/stable/notes/ddp.html)
and [SyncBatchNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html).
The installed environment, rather than current online documentation, defines
the actual execution version.

## Numerical and resource measurements

- Deterministic parity uses dropout disabled, training BatchNorm, actual-target
  batches of 16 and 15 jets, and one full AdamW update for each measured weight.
  Compare global loss, flattened gradients, all buffers, parameters and AdamW
  moments against the single-card reference. Record a parity pass using loss
  absolute error <=1e-3, gradient relative L2 <=5e-4 and buffer absolute error
  <=5e-5. Parameter and optimizer differences are additionally reported for
  review. Failure is visible evidence, never permission to release science.
- Timing enables normal dropout. Each weight has three warmup updates, then
  twelve timed full128 updates from the original first-epoch permutation.
  Timing synchronizes CUDA and takes the slowest distributed rank. It includes
  transfers, real losses, finite-gradient checks, backward, AdamW and collectives.
- Exercise an 11-jet partial batch, the actual final TRAIN batch, and a batch
  comprising the 128 longest TRAIN jets. Preserve global padding in every case.
- Time native HLT inference on eight validation batches (1024 rows). No AUC,
  recovery, checkpoint selection or final-test inference is performed.
- Record peak allocated GPU memory per rank. The largest-support stress is a
  memory test, not the throughput sample. Report the full list of step times.

A single-card OOM on the longest-support stress is recorded as a memory failure.
An earlier OOM or any distributed error fails that phase; torchrun terminates
its process group. The other phase is still attempted, and exit codes/logs are
preserved. Invalid inputs, nonfinite required values and corrupt artifacts fail
closed. A finite poor loss or small speedup is a valid benchmark result.

## Decision after the benchmark

`speedup = one_gpu_mean_step / two_gpu_mean_step` measures latency improvement
for a single training run on the same hardware. The report also calculates
`2/speedup`, the **idealized** throughput advantage of two independent one-card
runs over one two-card run. This estimate assumes both one-card fits fit memory
and do not slow each other through shared CPU, memory or I/O contention; it is
not a measurement of simultaneous independent training.

The training-only 30-epoch projection is `30*ceil(299234/128)*mean_step_seconds`.
It omits preparation, full validation, diagnostics, checkpoint writes and
scheduler wait. It must not set a production walltime directly. No assertion
is made that two 1080 Ti GPUs match an A6000/L40; that needs comparable measured
workloads. A6000 BF16 measurements are not a controlled FP32 speed comparison.

Prefer independent single-card fits for campaign throughput if they fit and
two-card scaling is modest. Consider distributed fits if needed for memory or
if lower per-fit latency is valuable. Any subsequent production profile needs
its own versioned execution, measured startup, native replay and complete
deployment checks, including a fresh FP32 U100-w5 control for a fair comparison.
The existing scientific automatic-release workflow is not enabled by this probe.

## Artifacts and acceptance boundary

Separate `LUKA_SUBCLASS_GPU_BENCHMARK_<KIND>/v1` artifacts are SPEC, PLAN,
CLAIM, BATCHES, PARITY, PHASE and REPORT. They bind exact source, parent hashes,
original population/teacher/U100 lineage, file checksums and accepted Slurm ID.
The spec is `benchmark_spec.json`, not `campaign_spec.json`. It fixes
`benchmark_only=true`, `science_release=false`; report fields
`science_qualified` and `science_submitted` remain false.

The TRAIN subclass bank reuses the existing v2 bank format with the benchmark
spec hash as its parent. Its existence cannot qualify a scientific execution.
Input payload budget is 2 GiB; require at least 3 GiB free before registration.
Publish files atomically and preserve journals after interruptions. Source and
parent validation run before preparation and each phase. Each rank verifies
the batch payloads; only one local node and one/two ranks are accepted.

Implementation lives in `luka_subclass_hydra/benchmark*.py`, with separate CLI,
queue helper and Slurm worker. Follow the [runbook](../LUKA_SUBCLASS_GPU_BENCHMARK_RUNBOOK.md)
for commit, registration, dry review, submission and read-only reporting.

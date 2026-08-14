# HCWDL Ragged Preprocessing Performance Guide

## Purpose

This document records a concrete preprocessing failure found in the 300k
HCWDL structural-feature homotopy campaign and the exact repair now present in
the repository. It is intended for every agent implementing a related U, D,
J, P0, matched-particle, offline-endpoint, or variable-support campaign.

The central rule is:

> Materialize each ragged branch a fixed number of times per source chunk,
> normally once per endpoint representation. Never call a helper that
> reconverts the complete chunk from inside a per-row loop.

This is an execution rule, not a change to particle definitions, matching,
homotopy coordinates, training losses, or deployable inference.

## Observed failure

The affected 300k jobs included `P0CE`, `P0KD`, `D100direct`, `U020`, `J010`,
and `S100_01`. After roughly four hours on one GH200, each remained before its
first validation checkpoint. In the same campaign, an HLT-only stationary job
had reached update 38,676 of 70,320 and epoch 32 in about one hour.

Slurm accounting for the blocked jobs showed approximately one CPU-hour per
elapsed hour, despite an eight-CPU allocation. Peak resident memory was only
about 8--18 GiB. The evidence therefore indicated:

- the GPU optimizer loop had not started;
- the jobs were not out of memory;
- matching and residual coupling were not running;
- one CPU core was spending hours constructing the initial RAM views.

The absence of `rolling_resume.pt` was expected because the jobs had not yet
reached the first validation boundary. It did not mean that Slurm had failed
to launch the worker.

## Exact root cause

The public single-row helper
[`project_offline_endpoint_records()`](../src/hlt_classification/scouting/repair.py)
accepts a chunk-shaped mapping plus one row index. To validate and project that
row, it converts every required ragged branch into Python/NumPy rows for the
*entire chunk*. This behavior is correct for occasional one-row calls.

It is not safe to use like this:

```python
for row in range(chunk_rows):
    features, validity, p4 = project_offline_endpoint_records(
        chunk_arrays,
        row=row,
    )
```

For a chunk containing `N` selected jets, each call reconverts approximately
`N` rows of every required branch. Calling it for all `N` rows therefore makes
the conversion term quadratic in chunk population:

```text
per-row helper cost       ~ O(N * branches)
called once for each row  ~ O(N)
total conversion cost     ~ O(N^2 * branches)
```

The production stream uses chunks of up to 4,096 rows. The same anti-pattern
also existed for raw HLT ragged branches in endpoint partition construction.
P0, D100, U/J view building, scale calibration, coupling production, and
coupling audits could all pay the repeated-conversion cost.

This was not a slow matching algorithm. The assignments and residual coupling
were already compact, persistent, authenticated artifacts. Training jobs only
joined those artifacts by jet identity. The expensive operation was repeatedly
materializing already-loaded ragged source arrays.

## Correct repair

### 1. Prepare each endpoint domain once per chunk

[`hcwdl_homotopy.py`](../src/hlt_classification/scouting/hcwdl_homotopy.py)
defines two immutable prepared containers:

- `PreparedOfflineEndpoints`
- `PreparedHltEndpoints`

The corresponding constructors are:

- `prepare_offline_endpoints()`
- `prepare_hlt_endpoints()`

`prepare_offline_endpoints()` performs exactly one ragged conversion per
required offline branch within the prepared-offline representation for the
chunk. It then constructs, for every row:

- raw 21-field endpoint values;
- the exact registered validity mask;
- combined charged-then-neutral p4;
- charged and neutral collection counts;
- the original per-branch row arrays needed by the endpoint projector.

`prepare_hlt_endpoints()` similarly converts each raw HLT feature/vector branch
once for the raw endpoint-partition representation and records:

- raw HLT feature rows;
- HLT p4 rows;
- the untruncated canonical HLT collection length.

The same chunk may also make a separate constant-time pass through a canonical
model-input builder such as `build_hlt_inputs()` or Shell Exact. That is still
linear. The forbidden pattern is conversion count proportional to the number
of rows. Do not claim that the complete view builder performs literally one
conversion total when it constructs multiple registered representations.

The row-level partition function receives these prepared objects and performs
only row-local indexing and validation:

```python
offline = prepare_offline_endpoints(arrays)
hlt = prepare_hlt_endpoints(arrays)

partitions = tuple(
    build_partition_from_arrays(
        arrays,
        row=row,
        assignment=assignments[row],
        prepared_offline=offline,
        prepared_hlt=hlt,
    )
    for row in range(offline.rows)
)
```

Do not move `prepare_offline_endpoints()` or `prepare_hlt_endpoints()` inside
that row loop.

### 2. Thread prepared data through every consumer

The following functions accept or reuse prepared endpoint batches:

- `build_p0_inputs(..., prepared=...)`
- `build_partition_from_arrays(..., prepared_offline=...,
  prepared_hlt=...)`
- `build_homotopy_inputs(..., prepared_offline=...,
  prepared_hlt=...)`

[`hcwdl_upper_builder.py`](../src/hlt_classification/scouting/hcwdl_upper_builder.py)
centralizes chunk preparation in `_prepared_partitions()`. Scale calibration,
base coupling production, exhaustive audits, independent sample audits, and
the frozen serial reference all reuse its prepared arrays.

An agent must repair every call path, not only the main training stream. A
fast training builder with a still-quadratic authorization audit merely moves
the timeout to an earlier campaign task.

### 3. Keep exact endpoint semantics

This optimization must not replace or approximate any scientific operation.
The repair preserves:

- native charged-then-neutral indices;
- the P0 90 charged / 60 neutral bound;
- all 21 raw endpoint fields and validity groups;
- float32 source p4 followed by the existing float64 endpoint carrier;
- HLT slot order, masks, padding, and canonical raw-length overrides;
- Shell Exact endpoint behavior;
- residual coupling and switch coordinates;
- jet identity and source/entry order.

The scientific projection implementation in
[`repair.py`](../src/hlt_classification/scouting/repair.py) was not altered for
this optimization. The new preparation path is regression-tested against that
legacy single-row reference.

Do not gain speed by:

- dropping validity checks;
- changing float precision;
- truncating before the registered endpoint rules;
- recomputing relative features from a changing constituent set;
- skipping particles or rows;
- weakening endpoint equality;
- persisting a reconstructed U/J dataset;
- changing matching, coupling, or switch semantics.

### 4. Overlap chunks in bounded canonical order

[`hcwdl_homotopy_stream.py`](../src/hlt_classification/scouting/hcwdl_homotopy_stream.py)
contains `_ordered_blocks()`, a bounded `ThreadPoolExecutor` wrapper.

Its requirements are deliberate:

- at most one in-flight source chunk per worker;
- completed blocks are consumed in submission order, not completion order;
- emitted jet/source/entry order is identical to the serial stream;
- worker count must be positive;
- one worker remains a supported reference path.

The executor may finish later chunks early, but it waits for the oldest
submitted future before emitting anything. This allows safe CPU overlap
without changing sampler order, cache bytes, teacher-target identity, or
resume behavior.

Do not replace this with unordered `as_completed()` emission unless the
scientific ordering and all downstream identity contracts are deliberately
versioned. Do not submit an unbounded future for every chunk; that can retain
the entire dataset in memory.

### 5. Bind workers to the Slurm allocation

[`view_build_workers()`](../src/hlt_classification/scouting/hcwdl_homotopy_runner.py)
uses `SLURM_CPUS_PER_TASK` as the default worker count. The optional
`HCWDL_UJ_VIEW_BUILD_WORKERS` environment variable may request fewer workers
for diagnosis. It may never exceed the Slurm allocation.

For the corrected 300k recovery, the training class requests:

```yaml
CPUs:     16
RAM:      96G
Walltime: 06:00:00
GPU:      gpu:gh200:1
```

The move from 8 to 16 CPUs is an operational resource increase, not the source
of the main algorithmic improvement. Local synthetic threading showed only a
small improvement for a simple P0 fixture; the elimination of repeated
whole-chunk conversion is the dominant repair. Other campaigns should measure
before requesting more than 16 CPUs.

More workers also mean more simultaneously retained raw and constructed
chunks. Preserve the one-chunk-per-worker bound and include all in-flight
chunks in peak-memory reasoning.

## Required phase instrumentation

[`hcwdl_homotopy_runner.py`](../src/hlt_classification/scouting/hcwdl_homotopy_runner.py)
prints explicit phase boundaries:

```text
HCWDL-UJ phase=student_view_cache role=train ... status=started
HCWDL-UJ phase=student_view_cache role=train ... status=complete ... seconds=...
HCWDL-UJ phase=student_view_cache role=validation ... status=started
HCWDL-UJ phase=student_view_cache role=validation ... status=complete ... seconds=...
HCWDL-UJ phase=teacher_targets ... status=started
HCWDL-UJ phase=teacher_targets ... status=complete ... seconds=...
HCWDL-UJ phase=optimizer_training ... status=started
```

Every related campaign should expose equivalent phase timing. A single total
job time is insufficient because it cannot distinguish:

- ROOT reading;
- particle projection/view construction;
- teacher inference;
- validation;
- optimizer training.

The student train and validation views are each built once into authenticated
process-local RAM caches. Non-TOFF predecessor logits are also computed once
per job before training. Epochs replay the cache; they must not reconstruct U,
D, J, or P0 inputs each epoch.

## How to diagnose a similar job

### Slurm resource evidence

For one job:

```bash
scontrol show job JOB_ID | tr ' ' '\n' | grep -E \
  'JobState=|RunTime=|TimeLimit=|NumCPUs=|NodeList='

sacct -j JOB_ID.batch -n -P \
  --format=JobID,Elapsed,AveCPU,TotalCPU,MaxRSS,MaxVMSize
```

Warning signs are:

- elapsed time grows for hours;
- `AveCPU` is approximately one core despite a multi-CPU allocation;
- GPU training checkpoints and validation history are absent;
- memory remains far below the request;
- an HLT-only control trains normally on the same node class.

### Phase and checkpoint evidence

```bash
grep -E 'HCWDL-UJ phase=' slurm-JOB_ID.out

find CAMPAIGN_ROOT/training/NODE_ID -maxdepth 2 \
  -type f -name 'rolling_resume.pt' -o -name 'training_report.json'
```

Interpretation:

- cache `started` without cache `complete`: source I/O or view construction;
- cache complete, target `started` without target `complete`: predecessor
  domain reconstruction or teacher inference;
- optimizer `started`: ordinary training/validation time;
- no phase line under corrected source: worker/source mismatch or failure
  before the runner entered the node.

Do not infer the phase solely from GPU utilization: preprocessing may still
allocate a model or CUDA context before optimizer training starts.

## Code-audit checklist for other agents

Before launching any related campaign, search for conversions inside row
loops:

```bash
rg -n '_rows\(|ak\.to_list|ak\.to_numpy|np\.asarray|\.to_list\(' \
  src/hlt_classification scripts

rg -n 'for row in|for .* in range\(' \
  src/hlt_classification/scouting
```

Review the call graph, not just matching lines. A dangerous helper may hide
the whole-chunk conversion several calls below the visible row loop.

For each new input domain or campaign, answer all of the following:

1. What is the source chunk size?
2. Which ragged branches are converted?
3. How many times is each branch converted per chunk?
4. Does any row-level helper receive the full chunk?
5. Are train and validation views built once or once per epoch?
6. Are teacher-domain views/targets built once or repeatedly?
7. Is concurrency bounded by allocated CPUs and memory?
8. Is result emission deterministic and in canonical order?
9. Do one-worker and multi-worker paths produce identical arrays and identity
   ordering?
10. Are phase timings printed before the first validation?

The acceptable answer to question 3 is a small, fixed number per branch per
chunk—normally once per registered representation—and never once per row.

## Required regression tests

The current tests live in
[`tests/test_hcwdl_homotopy.py`](../tests/test_hcwdl_homotopy.py):

- `test_prepared_endpoint_batches_are_legacy_exact_and_convert_once`
  verifies exact raw feature, validity, and p4 equality against the legacy
  projector across multiple rows, and counts exactly one conversion per
  required branch;
- `test_parallel_view_blocks_preserve_submission_order` verifies serial and
  multi-worker block emission order;
- `test_view_build_workers_are_bounded_by_slurm_allocation` verifies default,
  reduced, and oversubscribed worker behavior;
- endpoint, carrier, U100/D100, J100/HLT, cache, identity, and resume tests
  protect the scientific invariants around the optimized path.

Any analogous campaign should add tests that:

1. monkeypatch or instrument its ragged conversion primitive;
2. use a fixture with more than one row;
3. assert each branch is converted once within the prepared representation,
   and only a fixed number of times across the complete chunk builder;
4. compare every output array and metadata field with the trusted reference;
5. compare one-worker and multi-worker output identities and bytes;
6. exercise an incomplete final batch;
7. test worker counts of 1, allocated maximum, and allocated maximum plus one;
8. verify exact resume after cache construction and after teacher-target
   construction.

Do not accept a speed benchmark without equality tests. A fast path that
silently changes particles is a scientific bug.

## Performance evidence and claims boundary

The local synthetic benchmark for P0 preparation at 1,024 rows improved from
4.837 seconds to 0.364 seconds, a 13.30x speedup, with exact p4 equality. The
removed term was quadratic, so the benefit should be larger at the production
4,096-row chunk size.

This is local evidence. It does not by itself establish Tigris walltime or
thread-scaling. Real acceptance requires phase timestamps from corrected
production workers and successful completion under the bound Slurm envelope.

## Recovery rule

Running jobs are pinned to their submitted source commit. Pulling or editing a
different checkout does not update them. Do not requeue an old job and expect
it to acquire this repair.

Use the authenticated failed-closure recovery appropriate to the situation:

- source-only recovery for corrected execution source with unchanged
  resources;
- resource-only recovery for identical source with a measured OOM/timeout;
- the separately versioned execution-and-resource recovery when both the
  reviewed source fix and a monotonic resource increase are required.

Never edit an immutable campaign spec or hand-edit an `sbatch` command. Cancel
only exact IDs from the campaign-bound ledger, publish a post-cancellation
monitor, dry-run the recovery, verify the generated resource flags, and then
submit with the exact authorization phrases.

## Short handoff blurb

The HCWDL U/J preprocessing slowdown was caused by a quadratic ragged-array
anti-pattern, not matching or GPU training. A single-row offline projection
helper reconverted every branch for the complete source chunk and was called
inside a loop over every row; raw HLT branches were handled similarly. At a
4,096-row chunk size, this left multi-hour GPU jobs before their first
validation while using roughly one CPU core. The repair converts every
offline and HLT branch once per prepared endpoint representation into
immutable containers, passes those containers through
P0/partition/U/J/calibration/audit callers, and overlaps a bounded number of
chunks while emitting them in canonical order. Scientific endpoint, matching,
coupling, coordinate, and training semantics are unchanged and
equality-tested. Related implementations must audit for any
`_rows`/Awkward/NumPy whole-chunk conversion below a per-row loop, build
student views and teacher targets once per job, add phase timing, and verify
one-worker/multi-worker byte and identity equivalence before Tigris launch.

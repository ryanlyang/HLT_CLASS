# Balanced reconstruction input execution: v1

Authority: [scientific plan](../plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md).
`SPEC.input_execution_policy` must equal the content-hashed
`LUKA_BALANCED_RECONSTRUCTION_HYDRA_INPUT_EXECUTION_POLICY/v1` returned by
`inputs.policy()`. This is an execution revision; the scientific recipe,
population, model, objectives, readouts and existing scientific schemas stay v1.
Old specs cannot be run with revised source. Register a fresh committed study.

## Authorized reads and numerical identity

Keep the original file -> TRAIN/validation role -> entry acceptance order.
Every logical ROOT request remains a contiguous span wholly within one original
authorized role, now bounded to128 source rows. Reduced-out rows are discarded
before physical feature or target construction. No test/unused particle row is
finalized. HLT-only validation never requests OFFLINE branches for reconstruction;
U100/OFFLINE retain their explicitly authorized privileged input roles.

The new reader caches only Uproot's temporary numeric basket buffers, using its
documented `basket_array`/`final_array` interpretation stages. A ROOT basket can
physically span forbidden rows, as it already did under the old reader. Only
the original authorized entry bounds reach `final_array`, and only the selected
subset reaches physical particle construction, statistics and learning. This
does not authorize finalized whole-file arrays or persistent particle caches.
Production finalization uses numeric NumPy/jagged arrays to avoid repeated
Awkward row conversion; the same column stack and float64 conversion precede
physical construction. Original-reader payload equality is required.
See [Uproot's interpretation contract](https://uproot.readthedocs.io/en/stable/uproot.interpretation.Interpretation.html).

Retain at most128MiB of temporary buffer payload per decoder process with LRU
eviction. Oversize baskets are processed without retention; support is never
clipped. Each process owns one open file; changing files or closing the reader
discards its cache. Only numeric scalar and numeric-jagged particle branches
from the declared field allowlist are supported. Unexpected interpretations,
entry coverage or source hashes fail closed.

Features, vectors, identities, labels, offsets, mixed22 reconstruction values,
applicability masks and frames/aggregates must match the53bab0c5 reader exactly.
TRAIN FP64 normalization accumulation remains in its original per-jet order.
Publication authenticates all payloads, canonical joins and original source
hashes before/after construction. Timing and execution-policy metadata can
differ; scientific payloads and target-normalization identities must not.

## Parallel work and failures

Bundle consecutive same-file/same-role spans into batches containing at most128
selected rows. Four spawned CPU workers decode at most four outstanding batches.
Accept results in original order, irrespective of worker completion order.
The128MiB cache is per process, not a new HOME output allowance or physical
memory reservation. Existing node capacity/resource qualification remains.

Each submitted batch has a600s result deadline. Workers emit their own stage
markers and120s nonfatal stack samples. On failure/timeout, terminate the owned
worker pool and propagate failure; do not silently retry, skip rows, alter
support or release science fits. Keep exclusive Slurm/task journals and fail
before qualification publication. This repairs Executor shutdown's indefinite
wait on unfinished work. It does not promise recovery from an uninterruptible
kernel/filesystem fault; normal Slurm termination remains the outer boundary.

Scientific48h and the original qualification6h envelopes, four GTX1080Ti devices,12 CPUs,
64000MiB, initial-only HOME admission, test sealing and all other campaigns are
unchanged. Real Hydra qualification still must measure full inputs and four
GPUs before science release; local probes do not certify remote throughput.

The later [worker-launch policy](LUKA_BALANCED_RECONSTRUCTION_HYDRA_LAUNCH.md)
raises qualification only to12h after job217775's measured preparation time.
This INPUT_EXECUTION_POLICY/v1 and all authorized reader semantics are unchanged.

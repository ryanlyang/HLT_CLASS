# PMARD Ephemeral View Cache Contract

Contract: `hlt_classification_pmard_ephemeral_view_cache_v1`  
Campaign adoption: `hlt_classification_pmard_campaign_spec_v10`

This contract removes repeated offline decoding and alpha repair from
multi-epoch PMARD jobs without creating another dataset representation.

## Scope

- A nonzero-alpha teacher constructs its selected train and validation
  `privileged` `ParticleInputs` exactly once, in FP32, and retains them only in
  its process memory.
- A positive-coefficient representation-KD student retains aligned `hlt` and
  `privileged` train views. Validation remains HLT-only.
- Logit-only students continue to precompute compact teacher logits once and
  then use the HLT-only stream; they do not retain repaired particle views.
- Alpha-zero identity aliases and the native-offline oracle do not use this
  cache.

The cache never stores raw offline branches, assignments as model inputs, or
extra offline particles. It contains only the same fixed-shape model views,
labels, and canonical label-free identities that the prior online stream
would have emitted.

## Exact replay

Cached rows are indexed by canonical `(source, tree, entry)` identity. For
each epoch, replay reconstructs the existing deterministic schedule from the
authenticated split records:

1. epoch-seeded source-file shuffle;
2. rank/worker file partition;
3. eight-file chunk round-robin;
4. epoch-seeded within-chunk permutation;
5. epoch-seeded 32,768-row shuffle-buffer permutation;
6. unchanged batch packing and tail behavior.

The cache build order is not a sampling order. Tests compare multiple epochs
of cached replay against the online stream, including identities, batch
boundaries, labels, features, vectors, masks, and raw lengths.

## Memory and lifecycle

The campaign cap is 320 GiB. When Slurm exposes `SLURM_MEM_PER_NODE`, usable
cache arrays are additionally limited to 75% of the job request. Allocation
fails before the full arrays are created if the exact shape-derived estimate
exceeds the safe budget. Campaign v10 keeps 192 GiB for smoke and pilot, where
the bounded roles need only a small fraction of that allocation, and requests
384 GiB for production teachers, representation KD, generations, and
confirmation. Smoke exercises the same cache path on 4,096-row roles.

At 21 features, four vector channels, 200 tokens, a boolean mask, and int32
raw length, one cached view is about 20.2 kB per jet. The 300k/100k pilot is
therefore about 7.5 GiB of teacher view arrays, while an aligned two-view
representation cache is about 11.3 GiB for its 300k training jets. At the
nominal full-role counts, the corresponding estimates are roughly 70 GiB and
105 GiB before modest identity/index metadata. The 192/384-GiB requests leave
substantial room for construction transients, model state, and the OS.

The cache header records role, row count, view keys, array bytes, safe budget,
identity order/set digests, split/selection/assignment lineage, alpha, repair
configuration, and repair seed. It declares
`durable_artifact_published = false`.

Nothing is written to the campaign tree, `/dev/shm`, or `$SLURM_TMPDIR`.
Process exit releases the cache. A resumed or requeued job reconstructs it
before restoring or continuing the exact training state; no cache is treated
as a reusable artifact.

Campaign-spec v9 remains supported and retains its original streamed behavior
so an active immutable v9 execution is not reinterpreted by newer source.

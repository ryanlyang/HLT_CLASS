# HCWDL TRI100 Four-Spine LOGIT Contract

This contract governs the isolated full-data four-spine path-density study.

## Invariants

- The sole imported model is authenticated U000 from the bound TRI60 source.
- Source completion is not required; the U000 report, selected checkpoint, and
  train/validation probability bank must already be complete and authentic.
- Branches are DIRECT, COARSE, DENSE, and ULTRADENSE with lengths 1, 5, 8, and
  15 respectively.
- Exactly 29 fresh fits and 25 single-component reducers are registered.
- Every edge is immediate-parent-only LOGIT KD at C25/P75 and T=2.
- Every fit uses four-rank synchronous DDP with one GH200 rank per node,
  global batch 256, nominal rank-local batch 64, and unchanged optimizer
  update count.
- Partial global batches are disjoint and exactly row-weighted before DDP's
  gradient average; padding, duplication, and dropped rows are forbidden.
- Validation and early stopping are globally identical: rank zero evaluates
  the complete canonical validation population and broadcasts the result.
- Only rank zero publishes durable artifacts or attestations.
- All branch heads require the immutable four-node NCCL acceptance lock.
- Same-coordinate fits have matched initialization, sampler, and view seeds.
- There is no ensembling, checkpoint continuation, M1, final-test access,
  rolling resume, or partial checkpoint reuse.
- Each fit completes 60 through 100 whole passes under the registered
  floor-tail schedule and patience policy.
- Exact best-checkpoint selection is independent of the patience delta.
- Poor validation performance cannot invalidate, skip, or cancel a row.
- New artifacts never mutate or schedule against the source campaign.

## Durable artifacts

All graphs, recipes, source locks, distributed-acceptance locks, campaign
specifications, command plans,
training reports, selected/final checkpoints, probability shards/manifests/
locks, stage reports, aggregates, completion reports, monitors, and recovery
specifications use `HCWDL_TRI100_FOUR_SPINE_LOGIT_*/v1` contracts with content
hashes and explicit parents.

Probability banks contain only uint8 identity digests and float32 class
probabilities.  Train banks use T=2; validation banks use T=1.  Particle views,
hidden states, optimizer states, and representation targets are not durable.

## Recovery

Only exact ledger IDs may be monitored.  Recovery is created only after all
subject jobs are terminal.  A valid completed task is inherited; every other
task is retried from update zero on all four ranks in a new pinned worktree.
Cleanup and publication are rank-zero-only, and cleanup targets are resolved
beneath the campaign root before removal.

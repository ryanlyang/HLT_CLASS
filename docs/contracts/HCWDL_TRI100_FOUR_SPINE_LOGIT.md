# HCWDL TRI100 Four-Spine LOGIT Contract

This contract governs the isolated full-data four-spine path-density study.
Version 2 supersedes the cancelled four-node DDP execution registered under
version 1. The ladder science is unchanged; v1 and v2 artifacts are not
interchangeable.

## Invariants

- The sole imported model is authenticated U000 from the bound TRI60 source.
- Source completion is not required; its U000 report, selected checkpoint,
  and train/validation probability bank must be complete and authentic.
- DIRECT, COARSE, DENSE, and ULTRADENSE contain 1, 5, 8, and 15 fits.
- Exactly 29 fresh fits and 25 single-component reducers are registered.
- Every edge is immediate-parent-only LOGIT KD at C25/P75 and T=2.
- Every fit is one process on one GH200 with batch size 256. Multi-process
  data parallelism, batch partitioning, gradient synchronization, and
  distributed publication are forbidden.
- A production acceptance lock proves one Slurm node, one task, one visible
  GH200, and a real CUDA forward/backward pass before any fit begins.
- Same-coordinate fits have matched initialization, sampler, and view seeds.
- There is no ensembling, checkpoint continuation, M1, final-test access,
  rolling resume, or partial checkpoint reuse.
- Each fit completes 60 through 100 whole passes under the registered
  floor-tail schedule and patience policy.
- Exact best-checkpoint selection is independent of the patience delta.
- Poor validation performance cannot invalidate, skip, or cancel a row.
- New artifacts never mutate or schedule against the source campaign.

## Durable artifacts

Graphs, recipes, source locks, execution-acceptance locks, campaign
specifications, command plans, training reports, checkpoints, probability
artifacts, stage reports, aggregates, completion reports, monitors, and
recovery specifications use `HCWDL_TRI100_FOUR_SPINE_LOGIT_*/v2` contracts
with content hashes and explicit parents.

Probability banks contain only uint8 identity digests and float32 class
probabilities. Train banks use T=2; validation banks use T=1. Particle views,
hidden states, optimizer states, and representation targets are not durable.

## Recovery

Only exact ledger IDs may be monitored. Recovery is created only after all
subject jobs are terminal. A valid completed task is inherited; every other
task is retried from update zero in a new pinned worktree on one GH200.
Cleanup targets are resolved beneath the campaign root before removal.

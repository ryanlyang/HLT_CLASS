# HCWDL TRI100 Four-Spine LOGIT Implementation Plan

Status: the single-GH200 v2 implementation is locally complete. The cancelled
four-node DDP execution is retained only as v1 provenance after production
measurement showed no per-pass speedup. The replacement execution is the next
action after an exact pushed commit, canonical Tigris dry run, and in-campaign
single-GH200 acceptance lock pass.

## Scientific question

The existing ensemble ladders demonstrate that teacher proximity can preserve
more of the offline-to-HLT performance gap, but they mix path density with
probability ensembling.  This study removes that ambiguity.  It asks whether a
single selected model passed down a progressively denser path can retain more
information when every edge receives the same optimization budget.

This is a path-density experiment, not an ensemble experiment.  Every student
is freshly initialized, receives LOGIT KD from only its immediate parent, and
becomes the only teacher for the next rung.

## Registered branches

One authenticated full-data U000 is reused read-only.  The four branches start
concurrently and are sequential only within themselves:

```text
DIRECT
U000 -> D000

COARSE
U000 -> U050 -> U100 -> D066 -> D033 -> D000

DENSE
U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000

ULTRADENSE
U000 -> U020 -> U040 -> U060 -> U080 -> U100
     -> D090 -> D080 -> D070 -> D060 -> D050
     -> D040 -> D030 -> D020 -> D010 -> D000
```

There are 29 fresh fits: 1 + 5 + 8 + 15.  There are no M1 fits, no
probability ensembles, and no final-test inference.

## Coordinate semantics

U coordinates change the support/input homotopy while keeping matched fields
offline.  For example, U020 is `s=1/5, f=0` and U100 is `s=1, f=0`.

D coordinates keep full HLT particle support and move matched fields toward
HLT values.  The label denotes retained offline matched-field strength:
D090 is `s=1, f=1/10`, D050 is `s=1, f=1/2`, and D000 is exact HLT at
`s=1, f=1`.

Discrete matched fields retain the deterministic identity-keyed mixing rule
already established by HCWDL.  No degradation coordinate or source index is a
model input.

## Fit semantics

Every new fit uses:

- full authenticated mapped train and validation populations;
- fresh initialization, with no checkpoint continuation;
- immediate-parent selected-checkpoint logits only;
- constant `25% CE + 75% KD`;
- temperature `T=2`;
- batch size 256 on one GH200 process;
- unweighted CE and validation every pass;
- exact macro-AUC checkpoint selection, then minimum CE, maximum logR50, and
  earliest update as deterministic tie breakers;
- restoration of the exact selected checkpoint;
- no rolling resume or partial checkpoint reuse.

The same coordinate uses the same initialization, sampler, and view seed alias
in every branch.  In particular, all four D000 endpoints are matched-seed
fits.  Their causal teacher path is therefore the intended changing variable.

## Optimization schedule

The maximum budget is 100 passes:

```text
passes  1-3:   linear warmup to 3e-4
passes  4-45:  hold at 3e-4
passes 46-60:  cosine decay to 1.5e-5
passes 61-100: constant 1.5e-5 refinement
```

Early stopping is evaluated once per validation pass:

- minimum 60 completed passes;
- patience 15 passes;
- meaningful patience reset threshold `delta macro-AUC > 5e-5`;
- the patience clock accumulates before pass 60 and is not reset at pass 60;
- exact checkpoint selection remains independent of the patience threshold,
  so smaller improvements are still restorable;
- if patience was already exhausted by pass 60, the fit completes at pass 60;
- otherwise the fit continues through at most pass 100.

Stopping at an authorized pass is a complete scientific row.  Weak metrics do
not fail a task or prune a later registered rung.

## Single-GPU execution semantics

Every fit uses one process on one GH200. Batch 256, optimizer updates, seed
domains, data order, loss, validation, early stopping, and checkpoint
selection remain the registered scientific semantics. No DDP process group,
batch partition, gradient synchronization, distributed validation, or
multi-rank publication occurs.

This v2 execution replaces the cancelled v1 four-node execution after the
first production fits measured approximately 13.7--14.4 minutes per pass,
which was not faster than the established single-GPU jobs. Keeping one full
batch on one GH200 also avoids the observed rank-local batch of 64 and removes
cross-node all-reduce latency. This is an execution-only revision; the graph
and scientific comparison are unchanged but all artifact contracts are
versioned so v1 and v2 cannot mix.

The campaign first runs a short one-node acceptance task. It proves a genuine
Tigris Slurm job with one task, one visible GH200, and a real CUDA
forward/backward pass. The four branch heads depend on that immutable lock.
This is an execution gate inside the campaign, not a scientific smoke run.

## Teacher materialization

U000's existing authenticated probability bank is imported read-only through
a new consumer-binding lock.  After each nonterminal fit, one reducer runs the
selected checkpoint over the complete train and validation identities:

- train probabilities are stored at T=2 for the next KD edge;
- validation probabilities are stored at T=1 for stage auditing;
- every bank has exactly one component and complete identity coverage;
- the bank records report, checkpoint, logits, split, graph, recipe, and
  campaign lineage;
- no particle views, hidden states, optimizer states, or resume states are
  durable.

Twenty-five such reducers are required.  The terminal D000 fits do not publish
downstream target banks.

## Isolation and scheduling

The campaign has its own root, worktree, job prefix (`hcwsp4`), immutable
specification, command plan, journal, ledger, attestations, monitor, and
restart-from-zero recovery contract.  It does not cancel, hold, depend on, or
write into any TRI60, DX, RSET, RREL, CE5, SD5, or optimization-screen run.

The four first fits all depend only on the new preflight job, so Slurm may run
them simultaneously.  Later jobs depend only on the preceding reducer in the
same branch.  Aggregate completion depends on all four terminal D000 fits.

Fit request: one node, one task, 72 CPUs, 320 GiB, one GH200, three days.

Execution acceptance request: one node, one task, 4 CPUs, 32 GiB, one GH200,
30 minutes.

Reducer request: 72 CPUs, 192 GiB, one GH200, one day.

CPU lock/report jobs request 4 CPUs and 32 GiB.  The campaign reserves 20 GiB
of free storage and declares a conservative 20 GiB durable-output projection.

## Validation and execution policy

No standalone smoke campaign is required or authorized by this plan.  The
implementation reuses the installed-Weaver and genuine full-data TRI60 worker
evidence already bound to the authenticated source campaign.  Before live
submission it still requires:

1. focused local tests;
2. local single-process training, early-stopping, artifact, and recovery tests;
3. full command-plan construction;
4. canonical dry-run ledger materialization;
5. exact pushed source commit and clean detached worktree;
6. successful read-only authentication of U000, its selected checkpoint, its
   probability bank, the full foundation, and all role counts on Tigris;
7. a successful in-campaign one-GH200 execution acceptance lock;
8. explicit creation and submission authorization phrases.

Recovery never resumes a partial optimizer.  A failed task is cleaned only
inside this campaign root and rerun from update zero.  Completed upstream
artifacts remain authenticated and reusable.

## Primary report

The validation aggregate must include U000, all 29 fresh fits, selected and
completed pass counts, all 25 target-bank attestations, the exact branch paths,
and the four matched-seed D000 endpoints.  It reports every registered result
regardless of whether denser paths help.  Final test remains sealed.

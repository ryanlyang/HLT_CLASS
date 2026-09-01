# HCWDL TRI100 Persistent-HLT Four-Spine MT20 Implementation Plan

Status: active scientific implementation plan.

## Question

The immediate-parent persistent-HLT four-spine campaign can lose privileged
information at every sequential distillation edge. This campaign tests a
skip-connected multi-teacher alternative: every non-anchor student receives
temperature-scaled probability supervision from every earlier selected model
on its own spine, with most weight reserved for the immediately preceding
model.

This is intentionally the combined `MT20` intervention. It changes both the
base loss from C25/P75 to C20/P80 and the teacher topology from one immediate
parent to all earlier same-spine ancestors. It does not separately identify
which of those two changes causes a result.

## Frozen inputs and views

The campaign reuses read-only:

- the authenticated full-cardinality lexicographic bottleneck assignment;
- the persistent-HLT-support view policy and its all-row support audit;
- the collection-aware raw offline endpoint identity policy;
- the identical train and validation populations;
- the DIRECT, COARSE, DENSE, and ULTRADENSE coordinates;
- batch size 256, single-GH200 execution, initialization/view/sampler seed
  domains, the 100-pass floor-tail schedule, and early stopping;
- M0CE60 and pure-offline U000 only as reporting references.

The campaign trains its own fresh CE-only persistent-support anchor. It has a
new contract family, root, worktree, job namespace, checkpoints, probability
banks, ledger, aggregate, completion marker, monitor, and recovery. It has no
Slurm dependency on and never writes into an existing campaign.

## Graph

The four spines are unchanged:

- DIRECT: `U000 -> D000`;
- COARSE: `U000 -> U050 -> U100 -> D066 -> D033 -> D000`;
- DENSE: `U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000`;
- ULTRADENSE: `U000 -> U020 -> U040 -> U060 -> U080 -> U100 -> D090 -> ... -> D010 -> D000`.

The experimental anchor is `SP4MT20_U000`. All later teachers for a student
must be earlier selected models on that same spine, in nearest-to-farthest
order. Cross-spine teachers, endpoint ensembles, checkpoint continuation, and
weight continuation are forbidden.

## Exact loss and teacher weights

Every non-anchor fit uses 20% ordinary CE and 80% probability KD at T=2.

For one available teacher, that teacher receives all 80 percentage points.
For two or more available teachers:

- the immediate parent receives 50 percentage points;
- all older ancestors share the remaining 30 percentage points;
- older weights form a nearest-first geometric sequence with ratio 1/2 and
  are normalized exactly over the available older ancestors.

For `k >= 2` teachers ordered nearest first, the KD-loss contribution is:

```text
teacher 0: 0.50
teacher j: 0.30 * 2^(-j) / sum_{m=1}^{k-1} 2^(-m),  j >= 1
```

Weights are represented as exact integer numerator/denominator pairs in the
graph and campaign artifacts. Float materialization is derived only after
validating those rational identities. Examples are:

```text
D066: CE .20 + U100 .50 + U050 .20 + U000 .10
D033: CE .20 + D066 .50 + U100 6/35 + U050 3/35 + U000 3/70
D000 on COARSE:
       CE .20 + D033 .50 + D066 .16 + U100 .08 + U050 .04 + U000 .02
```

The weighted sum of teacher contributions is always exactly .80.

## Probability semantics and storage

Each selected non-endpoint model publishes the same authenticated per-model
probability bank used by the immediate-parent campaign: train probabilities at
T=2 and validation probabilities at T=1. A student authenticates every
required ancestor lock, requires byte-identical identity ordering and complete
coverage, and constructs one weighted T=2 teacher distribution in host RAM.

The weighted distribution is computed once per job in float64 accumulation,
validated finite/nonnegative/unit-normalized, converted to contiguous float32,
and joined by identity during training. It is never written to disk. Durable
artifacts contain only per-model probability banks, rational weight metadata,
component hashes, and reports. No particle views, mixed target arrays, hidden
states, optimizer states, or rolling-resume state are durable.

Mixing T=2 probabilities and applying one forward KL loss has the same student
gradient as the registered weighted sum of per-teacher forward KL losses. The
reported scalar losses differ only by a teacher-dependent constant entropy
term, which cannot affect optimization or AUC-based checkpoint selection.
Logits must not be averaged before softmax.

## Training and lineage

The anchor is CE-only and uses the persistent-support U000 view. Every other
fit is fresh initialization with C20/P80 T=2. A training report binds every
teacher probability lock, selected report, selected checkpoint, rational
weight, teacher order, and the hash of the RAM-materialized mixture. The
mixture hash is diagnostic lineage, not a reusable durable array.

Poor metrics never gate descendants. Missing or reordered identities,
incomplete ancestry, wrong weights, stale locks, nonfinite probabilities,
forbidden final-test access, and source or contract drift fail closed.

## Task and resource shape

The campaign has 30 fresh fits and 26 per-model reducers, plus authentication,
support audit, genuine-GH200 preflight, aggregate, and completion: 61 tasks.
Reducers remain necessary because all non-endpoint selected models can be
teachers. Multi-teacher fits read several compact probability banks but create
no additional durable mixture-bank task.

The existing measured single-GH200 resource profile remains the initial
request. Preflight must additionally prove construction of a real two-or-more
teacher RAM mixture before full live submission. Projected durable storage
must remain within the established 20-GiB campaign allowance.

## Reporting and recovery

Recovery is reported on `M0CE60 = 0%`, pure-offline `U000 = 100%`.
`SP4MT20_U000` is an experimental hybrid anchor, not the oracle. The aggregate
records every fit's teacher registry and compares endpoints with the
immediate-parent persistent campaign when its reports exist; absence of those
comparison reports never blocks completion.

Recovery uses exact ledger IDs. Completed authenticated tasks are inherited;
all others restart from update zero in a new source-pinned worktree. Cleanup
is confined to resolved paths beneath this campaign root.

## Queue-readiness evidence

1. Exact-weight tests cover every spine and all teacher counts.
2. Mixture tests prove identity alignment, float64 accumulation, unit sums,
   weighted-KL equivalence, and fail-closed tampering.
3. Graph tests prove same-spine complete ancestry and forbid cross-spine use.
4. Campaign tests prove 30 fits, 26 reducers, 61 tasks, isolated paths/jobs,
   measured resources, and no existing-campaign mutation.
5. Focused, adjacent, and repository-wide local tests pass.
6. Exact pushed source passes a genuine Tigris MT20 preflight before the full
   live campaign is submitted.

Final-test access is forbidden throughout.

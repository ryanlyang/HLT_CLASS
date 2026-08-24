# HCWDL MHPE TRI60 Dense Extension Plan

Status: implementation authority for the additive dense-extension campaign.

## Objective

Run a new full-population, 60-pass, three-track projection-ensemble campaign
beside the already-running TRI60 campaign. The source execution remains
immutable and continues unchanged. The extension imports only authenticated
source reports, selected checkpoints, and compact probability banks; every new
fit, expanded ensemble, task, report, lock, root, job name, and ledger has a
new dense-extension identity.

This is an additive campaign, not a revision or recovery of TRI60. It may wait
for source artifacts, but it may never submit, hold, cancel, rewrite, rename,
or delete a source job or artifact.

## Frozen paths

The LOGIT track is

```text
U000 -> U050E -> U100E -> D083E -> D066E -> D050E
     -> D033E -> D017E -> D000E -> M1_LOGIT
```

The RSET and RREL tracks are independently

```text
U000 -> <track>_U100E -> D075E -> D050E -> D025E -> D000E -> M1_<track>
```

The three endpoints then retain the established terminal compression:

```text
M1_LOGIT + M1_RSET + M1_RREL -> M1E -> M2
```

All `E` artifacts are uniform class-probability ensembles with predeclared
equal weights. Metrics never select or reweight a component.

The exact degradation coordinates are:

| Name | exact removed-feature fraction |
|---|---:|
| D083 | 1/6 |
| D075 | 1/4 |
| D066 | 1/3 |
| D050 | 1/2 |
| D033 | 2/3 |
| D025 | 3/4 |
| D017 | 5/6 |
| D000 | 1 |

Decimal rung names are labels only. Construction uses the exact rational
coordinates.

## Multi-horizon component policy

At each new rung, one specialist is trained from every eligible higher
teacher. Exact source specialists are reused when both their student
coordinate and teacher identity remain unchanged. A changed ensemble teacher
is a changed scientific edge and receives a new fit.

The expanded LOGIT ensembles have 3, 4, 5, 6, 7, and 8 components at D083,
D066, D050, D033, D017, and D000 respectively. D066 imports the three existing
U000/U050E/U100E specialists and adds the new D083E specialist. D033 and D000
likewise retain only source specialists whose teacher identities are still
exact, then add the new dense teachers. No old local-edge specialist is
mislabelled as a specialist of an expanded ensemble.

Each representation track has 2, 3, 4, and 5 components at D075, D050, D025,
and D000. Existing U000- and track-U100E-taught D050/D000 specialists are
reused. New D075E, expanded D050E, and D025E teacher edges are fresh.

The graph therefore registers exactly 48 fresh fits, 15 reducers, and 23
source-fit imports. All fresh IDs begin with `DX_`; source IDs retain their
original names. Expanded ensembles also begin with `DX_`, so an existing
`LOGIT_D066E` can never be mistaken for or overwritten by
`DX_LOGIT_D066E`.

## Representation supervision

RSET and RREL continue to use `C25P75`, temperature 2 logit KD plus their
declared representation loss. Representation targets are regenerated once per
fit, retained only in process RAM, audited without reconstructive payloads,
and released before exit. They are never published to disk.

Representation carriers remain track-local and distribution-local. The local
predecessor carrier chain is:

```text
D075E -> D075_from_<track>_U100E
D050E -> D050_from_D075E
D025E -> D025_from_D050E
D000E -> D000_from_D025E
```

Skip specialists use the carrier associated with their own teacher
distribution. RSET never imports an RREL carrier and vice versa.

## Training and data

- Every fit uses all authenticated mapped train rows and validation rows.
- Final test is inaccessible.
- Every fit uses 60 passes, validation every pass, effective batch size 256,
  the inherited optimizer/schedule, fresh initialization, and AUC-first
  checkpoint selection.
- Ordinary specialists use `C25P75`, temperature 2.
- Each M1 and M2 compression uses `C10P90`, temperature 1.
- M1 and M2 are exact-HLT, unified 21-channel deployable models.
- Poor metrics are results and never prune the graph.

## Source synchronization and isolation

Campaign creation authenticates the already-finished U000/U050E/U100E source
reports and compact probability locks. This releases D083 LOGIT and D075
RSET/RREL work immediately.

The original probability manifests retain their original immutable consumer
lists. A new dense source lock therefore acts as an explicit read-only adapter:
it binds each unchanged source probability lock and the exact new `DX_`
consumer registry. Runtime joins require both locks and reject any unregistered
dense consumer. The source manifest is never relabelled or rewritten.

One CPU-only source-gate task has an exact Slurm `afterok` dependency on the
running source campaign's campaign-complete job. After that job succeeds, the
gate authenticates source completion and the lower reusable specialists. It
does not write inside the source root. Reducers needing those imports depend
on the gate; independent early work does not.

The command plan declares zero source-campaign commands and contains no
`scancel` or `scontrol`. Source outputs are read-only. The dense campaign uses
its own worktree, root, `hcwtri60x_` job namespace, task attestations, dry-run
ledger, live ledger, reports, and recovery roots.

## Storage and resources

Durable outputs are selected/final checkpoints, small reports and locks, and
compact train/validation class-probability banks. Particle views, hidden
states, logits, and representation targets are not durable. No duplicate M2
checkpoint copy is made. Rolling resume and partial checkpoint reuse remain
disabled.

GPU jobs request all 72 CPUs on a GH200 node so the authenticated
full-population cache builder can use the established spawned-process source
parallelism. LOGIT fits request 256 GiB/3 days; RSET and RREL fits request
384 GiB/6 days; reducers request 192 GiB/1 day. The graph reserves 20 GiB free
space and projects at most 14 GiB of new durable campaign output.

## Completion and recovery

Completion is structural: all 48 fits and 15 reducers execute, aggregate, and
publish a finalist lock and campaign-complete report. No validation threshold
controls completion. M2 is designated as the single deployable model, but the
lock does not unlock final test.

Monitoring is exact-ledger and exact-job-ID only. Recovery refuses active or
unknown tasks, preserves valid completed attestations, cleans only the exact
failed task's dense-root outputs, and restarts that fit or reducer from zero
under a new source-pinned recovery worktree. Recovery never touches the source
campaign and never creates rolling state.

## Acceptance

Before live submission the exact pushed commit must pass focused graph,
probability, source-lineage, DAG, isolation, workflow, and recovery tests; the
complete repository suite; CLI help; contract-version checks; and
`git diff --check`. Existing genuine TRI60 production-worker and installed
Weaver evidence is inherited because the model factory, full-population cache
builder, loss kernels, and execution environment are unchanged. The additive
graph receives a full canonical dry run. No separate smoke campaign is added.

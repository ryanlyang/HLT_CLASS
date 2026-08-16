# HCWDL-MHPE paired 300k/60-pass study

Status: implementation authority for the paired reduced-population campaigns.

## 1. Question and fixed comparison

This additive study asks whether specialist loss balance explains the behavior
of the complete multi-horizon projection-ensemble ladder. It runs two campaigns
whose only intended scientific difference is specialist CE/KD weighting:

| profile | specialist loss | specialist T | M1 loss | M1 T |
|---|---:|---:|---:|---:|
| `C25P75_300K60` | 0.25 CE + 0.75 KD | 2 | 0.10 CE + 0.90 KD | 1 |
| `C10P90_300K60` | 0.10 CE + 0.90 KD | 2 | 0.10 CE + 0.90 KD | 1 |

Both use exactly 300,000 authenticated training jets, 100,000 validation jets,
and a sealed 100,000-jet final-test role. Ordinary execution reads only train
and validation. Every fresh fit trains for 60 complete passes, validates after
every pass, uses unweighted per-jet CE, and selects macro AUC followed by CE,
logR50, and earliest update. There is no performance early stopping.

## 2. Complete graph

Each campaign runs the complete 16-fit, four-ensemble MHPE graph:

```text
imported U000
  -> U050_from_U000
  -> {U100_from_U000, U100_from_U050} -> U100E
  -> {D066_from_U000, D066_from_U050, D066_from_U100E} -> D066E
  -> {D033_from_U000, D033_from_U050,
      D033_from_U100E, D033_from_D066E} -> D033E
  -> {D000_from_U000, D000_from_U050, D000_from_U100E,
      D000_from_D066E, D000_from_D033E} -> D000E
  -> M1
```

Arrows into specialists are KD. `E` nodes are uniform probability ensembles,
not trained models. Component ordering, temperature handling, FP64 accumulation,
FP32 publication, coordinates, teacher identities, and exact-HLT endpoint
semantics are inherited unchanged from the MHPE v1/v2 implementation.

The two profiles share target-coordinate initialization, sampler, repair, and
dropout seed aliases. Their roots, specifications, output artifacts, ledgers,
and Slurm names are separate. Outputs may not cross between profiles.

## 3. Foundation reuse and access

The campaigns reuse one completed, authenticated
`HCWDL_UNIFIED_BALANCED_FOUNDATION_SPEC/v2` 300k foundation read-only:

- imported unified `U000` report and selected checkpoint;
- imported CE-only exact-HLT `M0paired` report and selected checkpoint;
- the durable U000 train-logit manifest;
- split, row selection, assignments, balanced coupling, and endpoint lineage;
- the 60-pass optimizer, batching, and schedule recipe.

Reuse is authorized by a new v2 MHPE reuse lock. It binds the exact foundation
specification and lock, checkpoint/report/target bytes, role counts, original
foundation source registry, current source-pinned MHPE registry, and the known
historical U000 target-digest classification when present. It never rewrites
the foundation. Final test remains inaccessible without the ordinary finalist
and separate human execution locks.

## 4. Contracts and resources

The paired profiles use graph/node/recipe/campaign/training-report/waiver
contracts v3 and v4 respectively. The 300k reuse lock is v2. Unchanged MHPE
probability target, stage, aggregate, finalist, completion, monitor,
attestation, ledger, and recovery formats remain v1 while transitively binding
the new graph and campaign hashes.

GPU training and reducer jobs request exactly:

```yaml
CPUs: 8
RAM: 96G
Walltime: 06:00:00
GPU: gpu:gh200:1
```

CPU report jobs request 4 CPUs, 32G, and one hour. Student train and validation
views are built once per job with a 72-GiB combined array ceiling. The original
300k balanced repair seed is used. Every GPU worker requests `USR1` 120 seconds
before walltime and retains exact checkpoint/resume behavior.

## 5. Operations and recovery

Each profile renders the same 23-task topology: 16 fits, four reducers,
aggregate, finalist lock, and completion. The two topologies may run in
parallel. Poor finite metrics never stop descendants. Invalid inputs,
nonfinite targets, incomplete identities, stale source, corrupt artifacts, or
lineage mismatch fail closed.

Source and monotone resource recovery support the exact failed/downstream
closure while preserving completed artifacts. Cancellation is by exact IDs in
one campaign ledger only. Existing production-worker evidence, the unchanged
MHPE graph implementation, focused local regressions, and a complete dry run
are the operational basis; this paired study does not require a new standalone
smoke campaign. This decision does not authorize submission.

## 6. Acceptance

Queue readiness requires:

1. exact v1/v2 full-data graph hashes remain unchanged;
2. both 300k graphs contain identical 16-fit topology, coordinates, teachers,
   seeds, and 60-pass schedules;
3. only specialist loss weights differ, while M1 remains C10P90/T1;
4. both plans render 23 absolute, dependency-correct Slurm commands with
   distinct roots, names, and ledgers;
5. focused and full repository tests, CLI help, compilation, contract checks,
   Markdown links, and `git diff --check` pass;
6. each live submission is preceded by its exact nonmutating dry-run ledger
   and explicit profile-specific authorization phrase.


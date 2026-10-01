# JetClass2 CMS-proxy Oscar 100k DIRECT+COARSE plan

Status: active scientific implementation plan

## Objective

Run a shorter Oscar comparison using exactly 100,000 training jets and 50,000
validation jets from the authenticated CMS-proxy foundation. Register only:

```text
DIRECT: U000 -> D000
COARSE: U000 -> U050 -> U100 -> D066 -> D033 -> D000
```

The DENSE branch is absent, not merely held or omitted after submission. The
three fresh controls remain `M0HLT`, pure `OFFLINE`, and `U000`; recovery uses
`M0HLT` as zero and pure `OFFLINE` as one hundred percent.

## Population

Reuse the exact relocated 200k-train/100k-validation release, full-cardinality
matching assignments, proxy blocks, and offline ROOT bytes already validated
on Oscar. Do not redo particle matching and do not copy another dataset.

Derive a nested ordinary-role population by ranking each authenticated row
identity with SHA-256 over a versioned domain, parent foundation hash, role,
and identity. Select the lowest 100,000 train ranks and lowest 50,000
validation ranks, then restore original foundation order for cache creation.
The selection reads no label or particle feature. Its exact ordered identity
digests and parent foundation hash are recorded in
`POPULATION_SELECTION/v1`. Final-test membership remains inaccessible.

## Registered execution

The scientific plan contains exactly:

- 9 fresh fits: 3 controls, 1 DIRECT student, and 5 COARSE students;
- 5 probability reducers: `U000` and the four nonterminal COARSE teachers;
- 1 aggregate and 1 completion task;
- 16 Slurm tasks in total.

The existing six-CPU/90,000-MiB/two-slot Oscar resource shape is retained, but
the new source and smaller selected population require their own genuine
full-selected-population preflight. `GATE_SPEC/v6` imports the immutable Oscar
portable materialization, binds `POPULATION_SELECTION/v1`, measures U000 and
D050 caches for all selected rows, and produces `RUNTIME_PROFILE/v6`.

Only a completed v6 gate may create `SCIENTIFIC_PLAN/v2` and
`CAMPAIGN_SPEC/v2`. A v4/v5 profile cannot be relabelled. Source, selection,
foundation, runtime profile, task graph, and commands are content-hashed and
fail closed on drift.

## Invariants

- Deployable students consume only their registered proxy-HLT-derived view.
- Every comparison uses the same selected identities and existing assignments.
- DENSE tasks cannot appear in the plan, command plan, or submission ledger.
- Population selection is label blind and ordinary-role only.
- Final-test particles and predictions remain sealed.
- Scientific metric quality never controls task success or downstream rows.
- Live gate and science submission require separate exact authorization
  phrases after dry-run inspection.

## Narrow hardware-failure recovery

An uncorrectable GPU ECC failure is an operational failure, not a scientific
result. `CAMPAIGN_ECC_RECOVERY/v1` discovers every direct ECC root from exact
Slurm accounting and immutable failed-log bytes, then retries precisely their
registered downstream closure. Tasks with authenticated successful outputs
are retained. Every recovered GPU task excludes all evidenced failed nodes;
data, views, seeds, model, optimizer, population, and parent campaign remain
unchanged.

The observed Oscar incident has two roots on `gpu3001`: `train_OFFLINE` and
`reduce_U000`. The authenticated `train_M0HLT` and `train_U000` outputs remain
reusable, while the two roots, all DIRECT/COARSE descendants, aggregate, and
completion are replaced. The recovery fails closed for a non-ECC root or any
unrelated terminal task. Its command plan and dry/live ledgers are immutable.
Live recovery requires the exact phrase
`AUTHORIZE JETCLASS2 CMS PROXY CAMPAIGN ECC RECOVERY`.

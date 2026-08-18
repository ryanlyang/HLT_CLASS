# HCWDL-MHPE D066 Schedule Screen (300k)

Status: implementation-authoritative validation study.

This plan defines a paired optimization study for one question raised by the
completed MHPE ladders: does the apparent advantage of a nearby KD teacher
depend on optimization duration or peak learning rate? It is additive. It
does not alter, relabel, or resume any completed MHPE campaign.

## Scientific question

The completed `C10P90_300K60` campaign showed that some D066 students selected
checkpoints well after pass 20, while the full-data campaign was limited to 20
passes. The screen tests whether the teacher-horizon ordering is stable under
different optimization schedules before spending full-data GPU time.

The three paired D066 students in every schedule cell receive the same exact
D066 input view and differ only in their immutable teacher targets:

- `U000`: distant projected-offline root;
- `U050`: intermediate structural teacher;
- `U100E`: nearest structural endpoint ensemble.

All students use fresh but paired initialization, sampler, dropout, repair,
and optimizer seed domains. Every student uses unweighted `0.10 CE + 0.90 KD`
at temperature 2. The student architecture, D066 coordinate, AdamW epsilon,
weight decay, batch sizes, 5% warmup, cosine decay, minimum-LR fraction,
validation cadence, and macro-AUC checkpoint-selection policy are imported
unchanged from the authenticated source campaign.

## Exact grid and graph

The Cartesian grid is:

```text
passes = {20, 30, 40, 60, 80}
peak LR = {3e-4, 1.5e-4, 1e-4, 5e-5}
teachers = {U000, U050, U100E}
```

This produces 20 schedule cells and exactly 60 independent fresh fits. There
are no training edges among screen fits. All 60 GPU jobs become runnable at
submission time, subject only to scheduler capacity. One CPU aggregate waits
for all fits and one completion task waits for the aggregate. A poor metric is
a valid result and never suppresses another fit.

## Population and validation partition

The only permitted source is a completed, authenticated
`C10P90_300K60` campaign with 300,000 training rows, 100,000 validation rows,
and sealed 100,000 final-test rows. The screen reuses its immutable U000 and
U050 FP32 train logits and U100E/T2 train probability targets by hash. It does
not rebuild teachers or particle datasets.

The 100,000 validation identities are partitioned once by a frozen SHA-256
rank into:

- 50,000 checkpoint-validation rows, evaluated after every pass and used by
  the standard macro-AUC-first checkpoint selector;
- 50,000 schedule-scoring rows, evaluated exactly once using the selected
  checkpoint and never consulted by training or checkpoint selection.

The partition uses only authenticated jet identity and source/entry lineage;
it reads no label when assigning rows. Both subsets are disjoint, exhaustive,
immutable, and source-count authenticated. Final-test access is forbidden.

## Ranking and interpretation

Schedules are ranked by:

1. highest U100E schedule-scoring macro AUC;
2. lowest U100E schedule-scoring cross entropy;
3. largest `AUC(U100E) - AUC(U000)` on the scoring split;
4. fewer passes;
5. lower peak learning rate;
6. lexical schedule identity.

Every aggregate also reports `U050-U000` and all three absolute scoring
metrics. Ranking selects an optimization schedule, not a scientific winner
among teacher horizons. The predeclared next step is to confirm the top three
schedules with complete, seed-consistent full-data triplets. That confirmation
is outside this campaign and may not reuse screening-seed students as final
evidence.

## Runtime and storage

Each GPU fit requests 8 CPUs, 96 GiB RAM, six hours, and one GH200. CPU jobs
request 4 CPUs, 32 GiB, and one hour. Each worker builds the 300k train,
50k checkpoint-validation, and 50k schedule-scoring D066 views once in RAM.
It reads an authenticated durable target bank and writes only checkpoints,
training reports, a compact runtime report, and task attestation. Repaired
particle datasets remain non-durable.

Every GPU task is checkpointable, receives `USR1` 120 seconds before its
limit, and uses the existing exact optimizer/RNG resume path. A source-pinned
failed/downstream recovery preserves successful fits and may never change
the grid, data, targets, seeds, loss, or resources.

## Operational authority

The completed 300k/60-pass source already exercised the same D066 builder,
loss, target readers, training engine, checkpoint selection, and Tigris GPU
worker. Therefore this schedule-only additive study permits an explicitly
authorized carried-evidence waiver instead of requiring another standalone
smoke. The waiver does not weaken source authentication, dry-run, clean
source, exact GH200 request, or live-submission authorization.

The implementation is accepted only after focused and complete local tests,
all CLI help checks, source compilation, contract-version checks, and
`git diff --check`. No implementation action authorizes Slurm submission;
the human must supply both creation and submission phrases.

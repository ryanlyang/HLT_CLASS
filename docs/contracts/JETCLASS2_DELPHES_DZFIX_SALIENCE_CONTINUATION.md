# JetClass2 dz-fix salience continuation contract

This contract defines the source-pinned continuation from a recovered
`TRAIN_500K` bottleneck-readiness profile to a three-spine production dry run
for the `20260918_dzfix` snapshot. It does not authorize the final production
fits.

## Parent and dependency rules

`JETCLASS2_DELPHES_DZFIX_SALIENCE_CONTINUATION_SPEC/v1` binds the immutable
bottleneck readiness specification, the immutable readiness-recovery
specification, its live exact-ID ledger, the replacement profile job ID, the
dataset root, the continuation source commit, and a fresh isolated output
root. The initial launcher has an exact `afterok` dependency on that profile
job. There is no polling.

The continuation then advances through these success-only barriers:

1. authenticate the recovered bottleneck runtime profile;
2. build and live-submit the linear, quadratic, and quadratic-core25 salience
   assignment foundations, with 480-minute assignment shards;
3. wait for all three immutable foundation locks;
4. live-submit the registered four-way U100 screen (bottleneck context plus
   the three salience candidates);
5. wait for the screen completion and selection locks;
6. create the selected DIRECT/COARSE/DENSE production campaign and its full
   30-task canonical dry-run ledger.

Any failed parent leaves the downstream launcher dependency unsatisfied. A
new exact recovery is required; the continuation never treats path existence
or a failed job as success.

## Scope and storage

The continuation submits three non-scientific matching foundations and the
four registered U100 screen fits. It creates, but does not submit, the 16-fit,
12-reducer production DAG. ULTRADENSE is absent. Final-test particles and
predictions remain sealed. Particle views and dense matching state are not
published; foundations retain only compact assignment artifacts, and the
screen/production retain their ordinary selected checkpoints, reports, and
probability banks.

The reusable implementation is in
`src/hlt_classification/jetclass2_delphes/dzfix_salience_continuation.py`.
The thin CLI and worker are
`scripts/jetclass2_delphes_dzfix_salience_continuation.py` and
`sbatch/run_jetclass2_delphes_dzfix_continuation.sh`.

## Artifact families

- `JETCLASS2_DELPHES_DZFIX_SALIENCE_CONTINUATION_SPEC/v1`
- `JETCLASS2_DELPHES_DZFIX_SALIENCE_CONTINUATION_PLAN/v1`
- `JETCLASS2_DELPHES_DZFIX_SALIENCE_CONTINUATION_RECEIPT/v1`
- `JETCLASS2_DELPHES_DZFIX_SALIENCE_CONTINUATION_COMPLETE/v1`
- `JETCLASS2_DELPHES_SALIENCE_READINESS_SPEC/v2` for non-default bounded
  assignment walltimes; the historical 240-minute form remains v1.

Every artifact is content-hashed and source/parent bound. Live submission uses
durable exact-DAG intents and ledgers. Re-entry may reuse an identical completed
stage, but cannot silently change parents, resources, source, or scientific
scope.

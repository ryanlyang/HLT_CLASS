# LUKA_SIXCLASS_TAIL_HYDRA/v1

Authority: [active continuation plan](../plans/LUKA_SIXCLASS_TAIL_HYDRA_PLAN.md).
This is a new execution/relocation namespace. Scientific recipe, population,
training reports, target banks, view proofs, startup and metrics retain the
unchanged `LUKA_SIXCLASS_BALANCED_OSCAR_* /v1` contracts. New fits bind those
artifacts to the **Hydra spec hash**. Historical imported artifacts retain their
original spec hash. They are never rehashed with substituted local paths.

All hashed artifacts use canonical JSON, schema_version 1, explicit parent
hashes, development-validation scope, final-test sealed/accessed=false, and
unverified event independence/old-trial overlap flags. Reuse requires more than
path existence. No runtime imports from `Fresh_check` are introduced.

## Bundle and transfer

`BUNDLE/v1` binds the original spec and frozen population hashes, original and
export source snapshots, pinned donor manifest, four exact teacher receipt
hashes/jobs and successful accounting, and sorted unique relative payload refs
with size and SHA256. Fields explicitly distinguish epoch state authentication
at the exporter from absent state payloads at the receiver.

Allowed stages are exactly OFFLINE/M0HLT/CONCAT/UE100 with source jobs
9198207/9198208/9198209/9198210. Ref resolution rejects escapes, symlinks,
unregistered paths, size/hash mismatches and inconsistent original refs.
Original fit parents, accepted jobs, 20 committed epoch metadata rows,
selected report, integrated acceptance and bank semantics must agree.
Bank payload rows must match replayed canonical TRAIN identities/labels.
Target shapes are N x 128, probabilities N x 6 at T=2, with the original
TRAIN-only standardization and immutable FP32 target arrays.

`TRANSFER_COMPLETE/v1` is an un-hashed operational completion record bound to
the hashed bundle, exact file/byte counts and `all_files_verified=true`.
The receiver hashes each file itself; the marker alone never authenticates
later file reuse. It serializes transfer writers, rejects archive extras,
links, traversal and duplicates, and publishes each verified file atomically.
An interrupted transfer can repeat the same stream; differing existing files
fail without overwrite. It does not delete original data or older bundles.

## Registration, submissions and local results

`SPEC/v1` parents are the original Oscar spec hash and the bundle hash. It
records local import/data/source/runtime paths, pushed clean source snapshot,
exact donor files/environment/backend, frozen population, original recipe,
two nodes, imported receipt hashes, fixed L40 resources and quota reservation.
Protected values are compared with the active recipe, not merely with a
self-consistent content hash. The two tasks are `train_DE050` and `train_DE000`.

`PLAN/v1` binds spec, exact commands, dependencies, completed-stage reuse and
storage check. It declares two remaining scientific fits at first submission
and zero preflight jobs. `SUBMISSION_INTENT/v1` is written before `sbatch`;
`SUBMITTED/v1` binds its exact command, plan, task and returned job ID. A writer
lock serializes submissions. Missing/ambiguous acceptance is never retried
automatically. No job cancellation or mutation API exists in this workflow.

`FIT_BINDING/v1`, `EPOCH_COMMIT/v1` and `RESULT/v1` distinguish Hydra execution
from Oscar execution. Fit parents include the current spec/population and all
teacher receipts, including imported receipts and the new Hydra DE050 receipt
for DE000. An epoch's full state payload precedes its immutable commit marker.
The result is published last, after selection reload, predictions, metrics and
DE050's new bank. `FAILURE/v1` retains the failed attempt's stage/job/message.
`RESULTS/v1` compares imported and new AUC-selected checkpoints with imported
six-class balanced anchors, without checkpoint reselection or final-test access.

The fresh two-fit path uses unchanged scientific code; only artifact location,
execution provenance, Slurm site/resources, submission DAG and imported parent
authentication differ. Inference is HLT-only only for M0HLT and final DE000;
other ladder rungs and imported OFFLINE/CONCAT controls are privileged.

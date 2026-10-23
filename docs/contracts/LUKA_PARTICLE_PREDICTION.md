# LUKA_PARTICLE_PREDICTION / v1

Authority: [plan](../plans/LUKA_PARTICLE_PREDICTION_PLAN.md).

All new artifacts carry canonical content hashes, exact parents and exploratory
ALLJETS flags. RECIPE fixes the block7 output, FP32 shared predictor, TRAIN-only
token-weighted channel normalization, per-jet SmoothL1, exact aligned/Hungarian
arms and fixed30 schedule. Teacher and historical references retain original
contracts; they are authenticated in their original pinned worktrees.

SPEC binds clean committed source, teacher, completed U100-w5, seeds, split,
prepared cache, partition/debug resources and the three-job DAG. PLAN/CLAIM and
the exact submission journal bind every job, afterok dependency and explicit
authorization. No mutation of existing artifacts or scheduler jobs.

BANK binds native teacher checkpoint, prepared data, TRAIN identities/labels,
offsets and FP32 token payload hash; FP64 mean/scale and raw payload are exported
once and standardized at batch read. Padding is excluded before finite tests
and reductions. Equal counts without authenticated persistent-slot lineage are
insufficient. Scratch payloads may be retired; retained BANK metadata does not
claim the payload can still be replayed.
Only uniquely-owned `/tmp` directories on verified node-local disk are allowed;
reject Ceph/NFS/tmpfs and never fall back to home storage.
Export is deterministic FP32 with its backend restored afterwards. Both fits
must regenerate precisely the preflight's payload/index hashes before training;
results also verify these retained descriptors agree across arms.

PREFLIGHT binds actual environment/GPU, exact job, parity, maximum-support
stress, both technical fits/native checkpoint roundtrips, peaks, setup/export
and per-epoch timings and conservative <=1440min projections. Science requires
successful accounting AND valid evidence, never file existence alone.

TRAINING_REPORT and FIT bind all30 epochs, original selection, native-only
selected state, full-validation probabilities and removed-head readback.
RESULTS recompute historical/current metrics on identical rows and the same
original50k indices. No validation target fitting, final-test operation,
performance admission threshold or silent partial fit is allowed.

# LUKA_TEACHER_SWEEP_HYDRA v1

Authority: [active plan](../plans/LUKA_TEACHER_SWEEP_HYDRA_PLAN.md).
This additive implementation is pinned to donor
`0ae4f28fbc25df4aef0ddc617a16555f98b02591`. It does not change old SupCon5,
curriculum, A/B/C/R, particle-teacher or shared-message contracts.

New content-hashed `LUKA_TEACHER_SWEEP_<KIND>/v1` artifacts, schema 1:
`RECIPE`, `TRANSFER`, `SPEC`, `STATISTICS`, `EPOCH`, `TRAINING_REPORT`, `FIT`,
`RESOURCES`, `ACCEPTANCE`, `PLAN`, `SUBMIT_CLAIM`, `RUN_CLAIM`, `STARTUP`,
`TASK`, `RESULTS`, `PROGRESS`, and explicitly technical `TECHNICAL_BANK`.
All carry parent hashes and the exploratory-data disclosure flags. Existing
OFFLINE target banks retain `LUKA_EMBEDDING_AUX_BANK/v1` semantics and exact
selected-checkpoint SHA parents. Exact-DAG journals retain the shared HCWDL v2
contract; journal identity is the content hash of the exact command plan.

## Inputs and target boundaries

- Exactly 299234 TRAIN and 74895 validation jets, original ordered identities,
  native features, original full30/batch128 schedule. Final test remains sealed.
- Group A is replayed inside its original pinned worktree. C0 reuse verifies
  original files, source and same-seed report. The original transfer bundle and
  environment remain fixed; no substitution based only on pathname existence.
- The new delta contains only native OFFLINE TRAIN and validation cache shards.
  Every hash/size/path must equal the corresponding original cache-manifest
  entry already in the base bundle. Validate per-block identities, labels,
  offsets, capacity and prepared fingerprint on load. Wrong-role access fails.
- Export may rebuild missing native OFFLINE caches in RAM from authenticated
  raw files, matching the original deterministic NPZ bytes. Existing corrupt
  files fail. It never writes into original preparation or raw data.
- Target and summary moments use TRAIN only, population standard deviation,
  FP64 accumulation and floor 0.001. Masked-set slots equal maximum TRAIN
  multiplicity; overflow fails rather than truncating.
- Validation OFFLINE is permitted for teacher selection only. Student training
  reads native D000 and TRAIN U100/selected-OFFLINE banks. No particle match,
  identity, label, provenance, raw OFFLINE input or target enters HLT inference.

## Models and selection

Exactly T00-T11 and S00-S11. The plan and `contracts.recipe()` fix all objective
weights, reductions, auxiliary seeds and augmentations. The native teacher and
HLT model architectures stay unchanged. Teacher native weights share the same
initialization and batch order; auxiliary initialization does not advance the
native RNG. S00-S11 use the same U100-w5 initialization/sampler and two separate
embedding heads. All losses are training-only; exported state keys are `mod.*`.

Teacher checkpoints are selected by native full OFFLINE validation AUC and
original tie rules. Student checkpoints use native D000 validation with the
same rules. Clean native state is reloaded and predictions/metrics reproduced
before publishing a completion receipt. Export selected teacher h on TRAIN in
FP32; the projector, decoder and noise are absent. All 30 epochs run unless a
technical failure occurs. Accuracy, loss magnitude or weak target quality are
never admission gates. Nonfinite required tensors/gradients are failures.

## Execution and durability

Hydra `gpus/default/normal`, one typed RTX A6000, 6 CPUs and 32000 MiB. Setup
12h, fit ceiling48h, CPU release/report60min with one requested CPU/4000 MiB.
Allow the recorded Hydra rounding to two allocated CPUs for these CPU tasks.
Null optional Slurm exports require independent controller evidence; never
spoof cluster identity. Workers pin installed software, absolute project path,
`PYTHONNOUSERSITE`, library path, exact accepted job ID and immutable spec.

Setup covers all 12 teacher paths and the common student path with actual
miniature training, native initialization parity, selected reload, target-bank
export, ordered joins, warmed batches, partial batches and longest-support
backward. Report measured timings and host/GPU peaks; accept below 85% host and
88% GPU memory and the fixed walltime cap. Science estimates use a 2x timing
margin plus 900 seconds and a further 15-minute I/O/startup allowance. Students
also run an actual-selected-target miniature; its weights are discarded.

One concrete bootstrap review covers setup, CPU release and all upper-bound
science/report commands. Only walltimes may decrease after measured setup.
CPU release requires successful exact setup accounting, authenticated receipt
and acceptance; it performs a measured dry run and submits all pairs. Each
student has `afterok` only on its teacher, report `afterany` on all science.
Teacher order is interleaved with students; no cross-recipe dependency/cap.

Exclusive claims and append-only exact-ID journals precede mutation. Completed
submission is idempotent. Ambiguous/partial submission stops for inspection;
no blind retry, claim deletion, broad cancellation or reuse of failed roots.
Artifact publication is atomic and immutable. Expected job-owned output
inventories and byte hashes are revalidated before downstream reuse.

The active plan explicitly selects Hydra, superseding generic Tigris site
defaults for this campaign. Local tests establish readiness to submit setup;
the actual Hydra installed-Weaver/data miniature remains required before any
scientific fits are released.

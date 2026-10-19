# LUKA_EMBEDDING_AUX_* / v1

Authority: [implementation plan](../plans/LUKA_EMBEDDING_AUX_PLAN.md).

Every record has schema_version=1, a canonical content_hash, exact parent hashes,
and explicit exploratory/no-independent-test scope. New artifacts are not
interchangeable with attention KD or earlier representation KD campaigns.

- RECIPE fixes teacher tap, predictor, normalization, loss, matched fixed30
  training schedule and initialization policy.
- ANCHOR binds the original fixed30 parent and authenticated M0HLT/OFFLINE
  selected outputs and seeds. Original-source replay and byte-identical native
  CE/model/input kernels are required before use.
- GATE binds source snapshot, anchor, site/resources, population, storage budget
  and recipe. Only one registered technical preflight is allowed.
- BANK binds the teacher checkpoint, original prepared fingerprint and recipe.
  Its TRAIN-only NPZ has exact ordered uint8 `(N,32)` identities, int64 `(N,)`
  labels, FP32 `(N,128)` targets, FP64 `(128,)` mean and scale. Scales are >=1e-3.
  Manifest is committed last; every payload is size/hash checked before use.
- PREFLIGHT binds GATE/BANK, actual scheduler job, environment/GPU, numerical
  and native checkpoint checks, full technical epoch, longest support stress,
  measured resources and a conservative scientific walltime. Technical weights
  may not initialize science. Acceptance requires the exact submission journal
  and COMPLETED/0:0 accounting, not merely a report on disk.
- SPEC binds the accepted report and the one scientific fit. No runtime tuning.
- TRAINING_REPORT records every epoch, CE/auxiliary loss, full validation,
  selection and technical/scientific status. All 30 epochs are required.
- FIT/RESULTS bind selected HLT-only checkpoint, saved full validation
  predictions, original50k indices and recomputed comparison metrics. Reports
  must remain readable without loading the offline teacher into a model.
- PLAN and exact submission journal bind reviewed commands and immutable spec;
  ambiguous or partial submission is never silently retried.

Source snapshots, recipe equality, ordered row joins, file checksums, finite
values and resource checks fail closed. Weak scientific results never do.

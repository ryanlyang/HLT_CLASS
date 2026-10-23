# FullStat hybrid embedding contracts v1

Authority: [active plan](../plans/LUKA_FULLSTAT_HYBRID_EMBEDDING_PLAN.md).
The [CE-only cumulative contract](LUKA_FULLSTAT_CUMULATIVE_EMBEDDING.md) and
[FullStat20 contract](LUKA_FULLSTAT_OSCAR20.md) remain unchanged.

All new records are canonical content-hashed
`LUKA_FULLSTAT_HYBRID_EMBEDDING_KIND/v1`, schema1, explicit parents and original
sealed-test flags. No legacy artifact may be relabelled as a new hybrid fit.

- `RECIPE`: exact stage/input, immediate logit teacher, explicit embedding
  teacher order, isolated predictor seeds, fresh coordinate seeds,20 epochs,
  batch128, CE.25/KD.75/T2, independent total embedding weight5. Five nodes,
  no UE000 embedding target and no final M0HLT target.
- `TARGET_RECIPE`: selected native128-D tap, FP32 eval, TRAIN-only independent
  FP64 population moments with1e-3 scale floor, FP32 targets.
- `KD_RECIPE`: native temperature-two TRAIN probabilities,11 classes, BF16
  CUDA forward and FP32 softmax as in original FullStat reducers. Auxiliary
  predictions never supervise. No temperature-one or untempered logit bank.
- `SPEC`: clean pushed source, original-source `REUSE`, all recipes/nodes/tasks,
  3M/500k population, protected new output, personal quota, per-task walltimes,
  unchanged backend/environment and resource reuse policy. No new preflight.
- `REUSE`: completed original four-teacher chain, exact successful Slurm
  identity, full original native/KD acceptance, selected references/banks,
  environment/GPU and support. All pre-existing donor runtime bytes must match.
  No time or path-existence shortcut. New combination is not newly measured.
  The explicit unimported parallel trial particle-prediction additions may
  coexist in the source snapshot; this exemption never allows an existing
  runtime edit or changed imported dependency.
- `BANK`: selected native embedding checkpoint/report, target recipe and
  prepared parent, exact ordered uint8 Nx32 identities, int64 labels,
  FP32 Nx128 targets and FP64 128-D means/scales. All finite, unique IDs,
  TRAIN role, exact registered teacher/view. OFFLINE retains original sweep
  BANK identity/bytes, imported through an explicit cross-view TRAIN join.
- `KD_BANK`: checkpoint/report/recipe/prepared parents, exact teacher/view,
  TRAIN role, T2,11 classes, same unique ordered IDs and labels, normalized
  finite nonnegative FP32 Nx11 probabilities. Export/readback byte equality.
- `BUNDLE`: exact embedding bank refs/hashes, immediate KD bank ref/hash and
  temperature, prepared identity and all earlier completed TASK hashes. No
  averaging latent axes and no substitution of old/new identically named nodes.
- `STARTUP`: actual head/KD set, three disposable maximum-support updates,
  maximum-validation-support forward, tap parity, finite gradients, exact
  environment/GPU/source and memory bounds. Discard/reset before science;
  no benchmark, technical epoch or full-validation pass.
- `TRAINING_REPORT`: all20 passes, full update/row counts and learning rates,
  raw CE/T-squared KD/per-head losses and weighted terms. Tiny negative
  floating-point KL roundoff is tolerated; nonfinite values fail. Native-only
  selected state, AUC-first checkpoint selection and fresh initialization.
- `FIT`: native selected weights, full-validation predictions/report,
  checkpoint byte roundtrip and native probability/metric readback.
- `TASK`: completion-last after all required outgoing banks, exact accepted
  job/source/spec, all ancestor task hashes, bundle/report, exhaustive output
  refs. Ancestors require exact successful accounting. Every reuse validates
  metadata and payloads; no partial publication admits a descendant.
- `PLAN`: five exact jobs with all prior afterok IDs; no-requeue,
  kill-on-invalid-dep, exportNONE, absolute-project environment helper. Saved
  dry review, exact hash, authorization and exclusive claim before submission.
  Uncertain submission never automatically retries or mutates another study.
- `RESULTS`: authenticated saved probabilities only; pending stages explicit;
  AUC/accuracy and recovery against historical M0HLT/OFFLINE. Same AUC-selected
  checkpoints for R30/R50, signal/(signal+QCD) score, descending ceil-rank and
  ties included. Zero QCD passing is null/censored, never infinite. Intermediate
  richer-input models are labelled oracles; final native HLT is deployable.

Files publish atomically. Paths/timestamps/quality do not prove reuse. Technical
failure blocks descendants; low AUC, low recovery or censored rejection do not.
Final test and unused rows stay sealed throughout. Single-seed adaptive
development evidence does not include training-seed uncertainty.

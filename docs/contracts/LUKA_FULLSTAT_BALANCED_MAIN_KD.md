# LUKA_FULLSTAT_BALANCED_MAIN_KD / v1

Authority: [active plan](../plans/LUKA_FULLSTAT_BALANCED_MAIN_KD_PLAN.md).
Implementation: `src/hlt_classification/luka_fullstat_balanced_main_kd/`.

All artifacts use canonical SHA256 content hashes, schema version 1, explicit
parent hashes and development-validation/sealed-final-test flags. The original
`LUKA_FULLSTAT_BALANCED_LADDER` family remains unchanged.

- `SOURCE`: exact clean pushed commit and tracked production file hashes.
- `REUSE`: original balanced SPEC file reference, five completed native task
  receipts and file inventories, compatible original import-closure hashes.
- `RECIPE`/`SPEC`: one fresh D000, CE0.25/main-DE050-KD0.75 at T2, original
  four-teacher auxiliary losses and sampling; exact resources and references.
- `PLAN`/submission journals/ledger: one `train_DE000`; no teacher, setup or
  separate preflight jobs. Exact authorization and plan hash required.
- `FIT_BINDING`: new spec, reused population, reuse identity and four completed
  teacher receipt hashes. No continuation from the original CE-only experiment.
- `STARTUP_PHASE`/`STARTUP`: real integrated acceptance. `classifier_ce_only`
  must be false and `classifier_ce_kd` true; equality refers to gradients of
  `0.25 CE + 0.75 main_KD` on the deployed classifier parameters.
- `EPOCH`/`EPOCH_COMMIT`: loss means include raw `ce` and `main_kd`; `total`
  reconstructs their coefficients plus four embeddings, four auxiliary KDs and
  weighted contrastive loss. State/report payloads bind to the epoch and fit.
- `TRAINING_REPORT`: 20 contiguous epochs, deterministic selection and restored
  native selected weights. Resume preserves optimizer/RNG/draw cursor.
- `METRICS`/`TASK_RECEIPT`: selected validation probabilities and per-population
  metrics, native checkpoint, complete history, startup/view/binding inventory.
  No outgoing TRAIN bank. Receipt publishes last.
- `RESULTS`: report-derived metrics and balanced-anchor recovery; null recovery
  for censored/nonpositive gaps. Fast inspection does not claim payload replay.
- Recovery registration, claim, launch intent/approval/completion/failure use
  this same new family and preserve immutable evidence. Only explicit recovery
  of this experiment with identical runtime bytes is allowed.

Original population, VIEW_BUILD and BANK artifacts retain their original family
and are validated with the original balanced spec/recipe. Stored probabilities
are native teacher softmax at T2, not temperature-1 values. Identity/label order
must match every eligible student row, including repeated draw occurrences.

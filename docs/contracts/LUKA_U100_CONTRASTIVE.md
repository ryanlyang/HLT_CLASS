# LUKA_U100_CONTRASTIVE / v1

Scientific authority: [paired OFFLINE identification plan](../plans/LUKA_U100_OFFLINE_CONTRASTIVE_PLAN.md).
All new artifacts use `LUKA_U100_CONTRASTIVE_<KIND>/v1`, schema 1, canonical
content hashes, explicit parents and existing ALLJETS exploratory flags.
Historical artifacts retain their original versions and are validated by their
original contracts. They are never relabelled as new contrastive artifacts.

- **RECIPE / SAMPLER:** exactly twenty registered students; original CE1/U1005,
  thirty full epochs, batch128, native HLT inference, five negative strategies,
  temperature0.1, 1,024 unique negatives, three-epoch linear matching ramp.
  S0/S1 seeds and launch order are immutable. Sampler RNG includes the explicit
  training/diagnostic stream as well as epoch, canonical identity and component.
- **SPEC:** original fixed30 anchor and original direct-U100 result-only import,
  exact pushed clean source, all twenty arms, SPORC/debug A100, six CPUs,
  120,000 MiB host memory, four GiB persistent allowance plus one GiB free
  filesystem headroom, twelve-hour setup / twenty-four-hour fit caps. Personally
  available quota is a capacity check, not the amount the campaign must consume.
- **KEYS:** frozen original OFFLINE native pre-classifier width128; TRAIN-only
  population moments in FP64, population standard deviation floored at 1e-3,
  standardized then L2-normalized FP32 keys. Ordered unique uint8 `(N,32)` IDs,
  int64 labels, FP64 HLT pT and int64 real-token multiplicity. Exact teacher,
  source/spec, prepared population, normalization and sampler bindings.
  Finite zero keys are diagnostic; nonfinite targets fail closed.
- **BUNDLE / TEACHER_READBACK:** references the original U100 bank if its payload
  still validates, otherwise a single authenticated re-export. OFFLINE keys are
  exported once. Teacher checkpoints are not retrained. Validation probability
  readback uses rtol1e-5/atol1e-6 and produces no validation retrieval bank.
- **CAPACITY:** all eleven TRAIN class counts, conservative per-class capacity
  proof, availability of exclusion metadata and TRAIN HLT quantile edges.
  Every class must retain at least 410 candidates under the maximum registered
  exclusions; total population must also cover 4,096 after mixture exclusions.
  This is a sufficient admission bound, not adaptive quota relaxation.
  The historical cache provides unique row identities but no authenticated
  event/replica groups. That absence is recorded; distinct events are not inferred.
  The reusable sampler supports authenticated groups and excludes their union.
- **BASELINE_PARITY / PROFILE / RESOURCE_DIAGNOSTIC / ACCEPTANCE:** genuine
  installed-Weaver production-code miniatures for zero and all five samplers,
  using the full TRAIN candidate universe and full candidate counts. The zero
  kernel must match the original U100 selected native tensors and metrics.
  Worst-support batch128 stresses memory; warmed, shuffled representative
  batches estimate runtime including sampling, transfers and optimization.
  Full validation timing and diagnostic cost are included. Raw evidence is
  published before admission. Limits are 85% host and 88% GPU peak memory;
  measured walltimes cannot exceed the reviewed caps. Poor finite AUC is not
  an admission condition. Technical weights are discarded.
- **REVIEW / PLAN / SUBMIT_CLAIM / DISPATCH_CLAIM:** complete setup and prospective
  twenty-fit/report graph, exact authorization and immutable dry review.
  The detached submit-host dispatcher waits for exact successful setup
  accounting, authenticates its receipt, and submits a measured dry-reviewed
  graph within the authorized envelope. Only walltime may decrease. Two
  alternating `afterany` lanes limit this study to two fits; these are resource
  order dependencies, not teacher dependencies. Report follows all terminal
  attempts. Shared setup failure submits no science. No partial retry is automatic.
  Existing exact-DAG journal/ledger contracts remain unchanged.
- **TRAINING_REPORT / EPOCH / TRAIN_PROBE / FIT:** resolved seeds, target parents,
  all thirty epochs and one-based update counts, original learning-rate and
  selection rules, native selected probabilities, independent matching losses,
  diagnostics from fixed TRAIN probes. Probes restore RNG, mode and buffers and
  take no optimizer steps. Selected checkpoints contain neither auxiliary head.
  Native checkpoint reload must reproduce the selected probabilities/metrics.
  Zero arms call the original U100 optimizer/loss loop, never the matching head
  or candidate sampler, and do not open the OFFLINE key payload.
- **RUN_CLAIM / TASK:** exact accepted Slurm job identity, immutable output
  inventory, exclusive ownership. Missing/partial receipts are not reusable fits.
- **RESULTS:** exact saved validation joins and metric replay, historical anchors,
  matched fresh control of the same replicate, overlapping original50k as
  report-only, selected epochs and explicit incomplete rows. Accuracy, class AUC
  and rejection/censoring remain in the saved metric dictionaries. Final test
  is sealed. This screen is not a high-data recovery measurement.

Shared data and old campaigns are read-only. Corruption, wrong lineage,
nonfinite required quantities, source drift and forbidden data access fail
closed. Finite poor scientific performance completes and cannot cancel an
independent registered row.

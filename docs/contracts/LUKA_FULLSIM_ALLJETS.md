# Luka all-eligible exploratory development contract

Authority: [active plan](../plans/LUKA_FULLSIM_ALLJETS_PLAN.md). This is a
separate experiment, not a reinterpretation of legacy SPLITS or sealed access.

## Population and scope

The authenticated inventory has 890 files, 647,935 raw entries and 374,129
eligible paired rows. Cuts, class mapping and source-byte identities are unchanged:
offline jet pT > 200 GeV, matched nonempty paired jets, and QCD exclusively from
the QCD source files. Existing original file assignments are replayed exactly.

- TRAIN: all eligible rows in original train and former final-test files, 299,234.
- Validation: all eligible rows in original validation files, 74,895.
- Final test and unused eligible rows: zero.
- Original 50,000 selected validation identities remain a secondary reporting
  subset, never a separate checkpoint selector. File roles are never searched.
- Class proportions follow each complete file reservoir, without balancing or
  class quotas. Exact global proportions in each split are not promised.

All new top-level artifacts carry `historical_test_evaluated=true`,
`historical_test_repurposed_for_training=true`, `independent_test_available=false`,
`evaluation_scope=exploratory_development_validation`. `final_test_accessed=false`
means there is no final-test operation in **this** experiment; it must never be
interpreted as the historical test remaining untouched. Its rows are now TRAIN.
Reused low-level feature/model contracts retain their original schema; the new
split, prepared data and campaign parents supply the explicit changed scope.

## Artifact versions

Prefix `LUKA_FULLSIM_ALLJETS_`, schema v1: SPLITS, FOUNDATION, PARTICLE_AUDIT,
PREPARED, GATE, CACHE, SPEC, PLAN, SUBMISSION_CLAIM, TASK, RESULTS,
VALIDATION_DIAGNOSTICS. PREFLIGHT uses v2, inheriting role-specific validation
inference stress from the regular v2 gate. Legacy validators reject these types.

SPLITS embeds the immutable original split and reconstructs all memberships from
its authenticated inventory. It records exact class counts, original validation
memberships, original file roles, no unused rows and zero test rows. Both file
paths and file hashes are disjoint across new roles; cross-file event independence
is unverified because source event identifiers are unavailable.

FOUNDATION binds raw inventory, new split and clean source snapshot. PREPARED
binds that foundation, integrated audit, provisional-mm conventions and unchanged
seven-view physical contracts. Assignment identities and labels must replay the
new membership exactly. CACHE binds all per-file blocks and serial input parity.
No truncation, native endpoint change, validation gradient, or context encoder.

## Execution and reporting

Three gate jobs: scalar foundation, eight-worker one-pass durable preparation,
then genuine SPORC A100 preflight. Storage is bounded from authenticated scalar
constituent counts, seven ragged views, file temporaries and 5 GiB reserve. This
filesystem check is not a personal-quota guarantee; review quota separately.

Preflight loads one TRAIN/validation coordinate pair at a time. All seven
coordinates receive full-population technical passes, longest-TRAIN backward
stress, longest-validation inference-only stress, installed-Weaver parity,
checkpoint and TRAIN-bank readback. Same batch256/BF16 recipe and RAM/GPU headroom
thresholds. Measured projections must fit the SPORC debug 24h limit; no automatic
schedule reduction to force admission. Technical models cannot become teachers.

Science: nine fresh single-encoder fits, five TRAIN-only reducers, aggregate,
complete. Model graph, seeds, feature conventions and training recipe equal the
regular campaign. Full 74,895 validation selects each checkpoint. Its saved
temperature-one validation predictions carry exact identities/labels and a
checkpoint hash. Recomputed full and original-50k metrics bind the same checkpoint;
the secondary subset is never used to choose a pass. Predictions never enter KD
banks. Model performance never determines admission or completion.

Each mode requires a saved dry plan and its exact hash plus
`AUTHORIZE LUKA ALLJETS EXPLORATORY EXACT PLAN`. Immutable source and successful
three-job gate accounting are required before science creation. Claims/journals
prevent duplicate submission; interrupted submission is not silently retried.
There is no automatic follow-up, final-test command, old-model import or resume.

Results are single-seed development comparisons. More passes through a larger
training population mean more optimization work; this is not a compute-matched
data-size ablation. Historical-test comparisons are not independent confirmation.

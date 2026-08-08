# PMARD 64-Model Exploratory Test Comparison

Contract family: `v1`

This study implements the user's explicit decision to evaluate every model
from the completed T100 KD sweep and paired schedule follow-up on the pilot's
100,000-jet `final_test` role. It changes that role from an untouched
confirmatory holdout into an **exploratory comparison set**. Results may be
reported descriptively, but they may not be presented as a single-finalist,
unbiased final-test claim.

## Frozen inventory

The specification is created before test-role access and contains exactly 64
models:

- all 36 registered T100 weight/temperature/exposure sweep models;
- all 28 registered paired schedule-follow-up models.

Each row binds its source study, registry index, experiment ID, immutable
training-report hash, selected checkpoint path and hash, scientific axes, and
validation metrics. The follow-up must be a hash-valid child of the supplied
T100 sweep. Missing, extra, duplicated, non-HLT, representation-architecture,
or changed checkpoint rows fail closed.

The two completed studies are historical immutable artifacts. Their embedded
references are validated using their recorded versioned contracts and content
hashes. The exploratory reader does not recompute an old parent campaign's
scientific identity using the current campaign registry, because later
registry additions cannot retroactively alter a completed source-bound run.

## Access and evaluation

Two dedicated locks record the explicit all-model authorization and the
consequent consumption of the holdout. They are separate from, and do not
weaken, the standard PMARD single-finalist locks. Only after both locks exist
may a deterministic class-proportional 100,000-row selection be made from the
`final_test` role.

All 64 models consume the same ordered HLT-only rows. Each GPU worker verifies
the registered report and checkpoint, audits the deployable forward signature,
and computes the standard 15-class PMARD metrics. Logits and labels live only
in RAM long enough to calculate metrics. No prediction NPZ or per-jet output
is published. Every report records the ordered identity hash; aggregation
requires that hash and the exact row count to agree across all 64 models.

The comparison publishes validation and exploratory-test metrics together,
plus test-minus-validation deltas. Any ranking made after viewing these test
metrics is descriptive and post hoc. A future confirmatory claim requires a
new independent dataset or a predeclared external evaluation set.

## Durable contracts

- `hlt_classification_pmard_exploratory_test_spec_v1`;
- `hlt_classification_pmard_exploratory_finalist_lock_v1`;
- `hlt_classification_pmard_exploratory_execution_lock_v1`;
- existing `hlt_classification_pmard_row_selection_v1` with both exploratory
  access-lock hashes;
- `hlt_classification_pmard_exploratory_test_evaluation_v1`;
- `hlt_classification_pmard_exploratory_test_report_v1`;
- `hlt_classification_pmard_exploratory_test_ledger_v1`.

The Slurm graph is authorization, row selection, an uncapped `0-63` GPU
array, then aggregation. A failure in one evaluation prevents an incomplete
aggregate from being mistaken for the 64-model comparison.

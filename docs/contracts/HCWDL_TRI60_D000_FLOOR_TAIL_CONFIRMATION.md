# HCWDL TRI60 D000 Floor-Tail Confirmation Contracts

This one-fit, full-data campaign answers one matched schedule question.  It
compares the existing `P90_H45_LR3E4` D000 fit with a fresh fit that uses the
same D033E probability teacher, exact-HLT D000 view, initialization seed,
C25/P75 loss, temperature 2, batch size 256, data population, and single-GPU
execution.  Only the registered training protocol changes.

The new protocol warms up for passes 1--3, holds 3e-4 through pass 45, decays
rapidly to 1.5e-5 through pass 60, and holds 1.5e-5 through at most pass 100.
Early stopping has a 60-pass minimum, 15-pass patience, and 5e-5 meaningful-AUC
threshold.  Exact improvements still control best-checkpoint restoration; the
threshold controls only the patience clock.

Version-1 content-addressed artifacts use the prefix
`HCWDL_TRI60_D000_FLOOR_TAIL_CONFIRMATION`:

- `REFERENCE_LOCK/v1` authenticates the immutable budget-screen spec, source
  lock, full-data foundation, recipe, and D033E train target manifest;
- `GRAPH/v1` and `NODE/v1` fix both matched protocols and the one new fit;
- `SPEC/v1` and `COMMAND_PLAN/v1` fix the isolated four-task DAG and resources;
- `TRAINING_REPORT/v1`, `SELECTED_CHECKPOINT/v1`, and
  `FINAL_CHECKPOINT/v1` bind the fit, schedule, lineage, and selected weights;
- `CAMPAIGN_COMPLETE/v1` records completion independently of metrics.

The existing P90 report is not a scheduler or training dependency.  It is
required only when printing the final paired comparison.  The campaign never
mutates source artifacts, never accesses final test, has no rolling resume,
and treats poor metrics as a valid scientific result.

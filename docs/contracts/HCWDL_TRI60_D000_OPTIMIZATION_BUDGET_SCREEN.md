# HCWDL TRI60 D000 Optimization-Budget Screen Contracts

The campaign publishes version-1, content-addressed artifacts under the prefix
`HCWDL_TRI60_D000_OPTIMIZATION_BUDGET_SCREEN`:

- `SOURCE_LOCK/v1` authenticates the exact original full-data campaign,
  foundation, recipe, selected source D000 report, D033E stage, and train and
  validation teacher manifests;
- `GRAPH/v1` fixes all 17 conditions, their common seed/domain/loss endpoint,
  and their exact LR and loss schedules;
- `SPEC/v1` fixes the source commit, paths, resources, roles, and task DAG;
- `COMMAND_PLAN/v1` fixes every Slurm command and dependency;
- `TRAINING_REPORT/v1`, `SELECTED_CHECKPOINT/v1`, and
  `FINAL_CHECKPOINT/v1` bind every fit to its graph, source, schedules, and
  restart-from-zero policy;
- `AGGREGATE/v1` contains the imported reference, all completed screen rows,
  validation landmarks, and a non-operative ranking; and
- `CAMPAIGN_COMPLETE/v1` records structural completion independently of the
  scientific result.

Validation fails closed on stale hashes, wrong source selections, a teacher
temperature other than 2, changed input/seed/batch/loss endpoint, missing
checkpoints, forbidden final-test access, or source mutation.  Poor metrics are
valid results and do not fail completion.

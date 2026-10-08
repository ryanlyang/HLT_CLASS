# S3 automatic follow-up contract

Authority: [operational plan](../plans/JETCLASS2_S3_FOLLOWUP_PLAN.md), additive
to [S3 scientific contracts](JETCLASS2_S3_LADDER.md).

New operational namespace `JC2_S3_FOLLOWUP/v1`, schema 1, kinds
`authorization`, `review`, `submitted`. Canonical content hashes and atomic,
non-overwriting publication. Original scientific artifacts are not edited.

Under the original study's separate `science_followup/` directory:

- `authorization.json`: parents = original spec and live gate ledger; binds
  exact original source/project, preflight ID, controller commit/file hashes,
  seven-job policy and no final test.
- `review.json`: parents = authorization, authenticated completed preflight,
  exact canonical dry science plan; records the bounded policy.
- `complete.json`: parents = authorization, exact plan, verified live science
  ledger; contains all seven distinct IDs. `training_complete=false` means
  successful submission, not successful fits or validated scientific results.
- `controller.lock`: kernel-held exclusive advisory lock. Existing complete
  receipts must authenticate; a stale file is not ownership of the live lock.

Slurm accounting success alone is insufficient. Original source and complete
preflight validation, original measured plan generation, canonical dry ledger
and original exact-hash live submitter remain mandatory. The separate controller
never imports scientific modules in its own process: a fresh interpreter
loads them from the preflight's original project. Historical v1 source locks
and `automatic_followup=false` metadata are not rewritten; this authorization
adds the explicitly requested operational continuation externally.

Any missing final live ledger alongside a live submission claim/journal is
ambiguous and blocks automatic retry. Completed submission evidence must match
the regenerated plan, all resolved dependencies and every journal event. No
claim stealing, automatic repair, partition switching, cancellation, checkpoint
modification, extra branch, strength tuning or test inference is permitted.

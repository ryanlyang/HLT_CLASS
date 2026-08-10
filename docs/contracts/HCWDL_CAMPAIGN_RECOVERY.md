# HCWDL primary failed-closure recovery

`HCWDL_FAILED_CLOSURE_RECOVERY_SPEC/v1` is the supported way to continue an
ordinary HCWDL pilot, named-midscale, or production campaign when an execution
defect is fixed in a later source commit. It complements the narrower dense
ladder and interrupted-final recovery contracts.

The recovery binds the canonical parent campaign specification, its complete
original submission ledger, and an immutable Slurm monitor report. At least
one exact original job must be an authenticated retryable failure. The retry
set is exactly the union of every failed task and its downstream dependency
closure. Every dependency entering that closure from outside it must be a
completed, artifact-valid parent.

The corrected source checkout may change execution code only. The parent
campaign's recipe, graph, node identities, domains, teachers, initialization,
seeds, assignments, role populations, resources, output paths, confirmation
registry, and final-test policy remain immutable. Recovery workers dispatch
the parent scientific specification through the corrected checkout and write
separate recovery attestations. Existing rolling checkpoints remain in the
parent output directories and are resumed under their original checkpoint
contracts.

For the observed 1M failure, `D50c` and `D50w` are completed external parents.
`D25c` and `D25w` are independent recovery roots. Their cold and warm
descendants rejoin at screen aggregation, after which the originally
authorized confirmation and sealed-final stages continue normally. No
completed predecessor, matcher assignment, endpoint qualifier, or root model
is rerun.

Required creation phrase:

```text
AUTHORIZE HCWDL FAILED CLOSURE RECOVERY
```

Required submission phrase:

```text
SUBMIT HCWDL FAILED CLOSURE RECOVERY
```

The original pending descendants should be cancelled only by exact IDs from
their original ledger after the immutable failure monitor has been captured.
They must not be released or requeued because they remain pinned to the broken
source commit.

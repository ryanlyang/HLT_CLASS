# HCWDL dense failed-closure recovery

`HCWDL_DENSE_RECOVERY_SPEC/v1` is the only supported way to continue an
immutable ten-point or five-point dense campaign when a runtime implementation
defect is fixed in a later source commit.

The recovery binds the complete original submission ledger and an immutable
monitor report. Every task before the first retry must be completed and have a
valid task attestation. At least one task must be an authenticated retryable
failure. The retry set is exactly that failure plus its downstream dependency
closure. A completed prefix is reused in place and is neither copied nor
retrained.

The new source commit may repair execution code but cannot change the parent
graph, recipe, assignments, imported controls, seeds, resources, node order,
or output root. Recovery workers dispatch the original immutable scientific
specification through the corrected source checkout and publish separate
recovery attestations and a lineage-bound submission ledger. Dense campaigns
remain validation-only and cannot access final test.

The current repair admits every finite `HIGHCOV_SHELL_EXACT/v1` alpha in the
closed interval `[0,1]`, as required by D95, D90, and the remaining dense
rungs. All other PMARD repair families retain their registered legacy alpha
grid.

If a recovery worker exposes a second source defect later in the same closure,
create another recovery identity from a newer clean commit while retaining the
same immutable original ledger and failure monitor. The registered retry set
remains the original failed/downstream closure: workers validate and reuse any
node completed by the earlier recovery, and the interrupted node resumes from
its compatible rolling checkpoint. Pending jobs from the superseded recovery
must be cancelled only through their exact recovery-ledger IDs.

## Measured-resource rescheduling

`HCWDL_DENSE_RESCHEDULE_SPEC/v1` replaces one live-authorized v1 recovery
whose jobs have not begun because its inherited resource envelope is too
large for effective backfill. It is operational only: the original dense
campaign remains the scientific authority, and the failed closure, graph,
recipe, data, assignments, teachers, seeds, output root, and checkpoint
lineage are unchanged.

The v1 profile is frozen from completed Tigris jobs `77534` (`D90c`) and
`77546` (`D95c`). They completed in 8,983 and 9,187 seconds with peak RSS
50,285,376 and 50,287,104 KiB. The replacement `gpu_single` request is eight
CPUs, `96G`, `06:00:00`, and one `gpu:gh200:1`; the CPU aggregate request is
unchanged. The spec binds the exact prior recovery spec and submission ledger,
and its `superseded_jobs` are the prior ledger's exact job IDs. Recursive or
arbitrary resizing is forbidden.

Create and dry-run the replacement before cancelling anything. Then cancel
the prior recovery through `cancel_hcwdl_campaign.py` and its exact ledger,
confirm those jobs have left the queue, and submit the replacement. Never use
job-name cancellation or edit the immutable prior ledger.

Required creation phrase:

```text
AUTHORIZE HCWDL DENSE FAILED CLOSURE RECOVERY
```

Required live-submission phrase:

```text
SUBMIT HCWDL DENSE FAILED CLOSURE RECOVERY
```

Measured-reschedule creation phrase:

```text
AUTHORIZE HCWDL DENSE MEASURED RESOURCE RESCHEDULE
```

Measured-reschedule live-submission phrase:

```text
SUBMIT HCWDL DENSE MEASURED RESOURCE RESCHEDULE
```

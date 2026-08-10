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

Required creation phrase:

```text
AUTHORIZE HCWDL DENSE FAILED CLOSURE RECOVERY
```

Required live-submission phrase:

```text
SUBMIT HCWDL DENSE FAILED CLOSURE RECOVERY
```


# HCWDL interrupted final-evaluation recovery

This contract defines the only supported recovery after a sealed HCWDL
final-test worker has consumed its one-time execution claim but terminates
before publishing `evaluation_manifest.json`.

## Scientific boundary

Recovery is execution-only. It must not select or replace finalists, change a
seed, checkpoint, input domain, test assignment, recipe, metric, or execution
lock. It must not issue a second claim. The existing
`HCWDL_EXECUTION_CLAIM/v1` is reusable only when its execution-lock and exact
final-test-assignment hashes match the frozen campaign.

The recovery source commit may differ from the parent campaign only to repair
execution machinery. All scientific inputs remain the original immutable
artifacts. A monitor report must authenticate the original sealed-evaluation
job as a retryable failure, and no completed evaluation manifest may exist.

## Resume semantics

`HCWDL_FINAL_EVALUATION_MANIFEST/v2` supports an interrupted evaluation:

- an existing per-finalist `HCWDL_FINAL_EVALUATION/v1` report is validated
  against the frozen registry and reused byte-for-byte;
- only missing finalist reports are evaluated;
- an existing report with different lineage fails closed;
- the completed manifest records the exact claim hash and whether the claim
  was created or reused;
- the final test remains forbidden for model selection.

`HCWDL_FINAL_RECOVERY_SPEC/v1` registers exactly two jobs:

1. `sealed_final_evaluation` under the corrected, source-pinned checkout;
2. `aggregate_report` after successful evaluation.

The recovery publishes its own submission ledger and task attestations while
the scientific evaluation and aggregate remain in the original campaign root.

## Required authorization phrases

Creation:

```text
AUTHORIZE HCWDL EXACT INTERRUPTED FINAL RECOVERY
```

Live submission:

```text
SUBMIT HCWDL EXACT INTERRUPTED FINAL RECOVERY
```


# HCWDL interrupted final-evaluation recovery runbook

Use this only when the original sealed final-evaluation job failed after
creating its execution claim and before publishing a completed evaluation
manifest. It does not authorize a new final-test analysis.

The recovery requires a clean worktree pinned to the pushed repair commit, the
original campaign specification and submission ledger, and a freshly queried
monitor report proving the exact original job failed. Create the recovery spec
with `create_hcwdl_final_recovery.py`, inspect a dry-run ledger from
`submit_hcwdl_final_recovery.py`, and submit the same spec with the exact live
authorization phrase.

The worker revalidates all original locks, assignments, finalist reports,
checkpoints, and the existing execution claim before touching data. Successful
completion produces:

```text
<parent>/final_test/evaluation/evaluation_manifest.json
<parent>/reports/campaign_complete.json
<recovery>/submission_ledger.json
<recovery>/attestations/*.json
```

If either recovery job fails, do not construct another spec blindly. Diagnose
the exact job and preserve the recovery ledger. A subsequent recovery is valid
only if it binds the same scientific lineage and authenticates the new failed
attempt.


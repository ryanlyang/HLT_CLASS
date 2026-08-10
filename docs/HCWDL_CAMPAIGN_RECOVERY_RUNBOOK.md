# HCWDL primary failed-closure recovery runbook

Use this procedure when an ordinary HCWDL campaign failed because of a source
defect that is corrected in a later pushed commit. Do not use `scontrol
requeue`: the original workers remain source-pinned to the broken commit.

The procedure requires the canonical parent `campaign_spec.json`, its complete
original `submission_ledger.json`, and an authenticated monitor generated
before the stale jobs are cancelled. Create a clean detached worktree at the
repair commit and a separate recovery root.

```bash
python -s scripts/monitor_hcwdl_campaign.py \
  --campaign-spec "${PARENT_ROOT}/campaign_spec.json" \
  --submission-ledger "${PARENT_ROOT}/submission_ledger.json" \
  --query-slurm \
  --output "${PARENT_ROOT}/recovery/failure_monitor.json"

python -s scripts/cancel_hcwdl_campaign.py \
  --submission-ledger "${PARENT_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "CANCEL HCWDL EXACT IDS"
```

Create and inspect the immutable recovery:

```bash
python -s "${RECOVERY_WORKTREE}/scripts/create_hcwdl_campaign_recovery.py" \
  --parent-campaign-spec "${PARENT_ROOT}/campaign_spec.json" \
  --parent-submission-ledger "${PARENT_ROOT}/submission_ledger.json" \
  --monitor-report "${PARENT_ROOT}/recovery/failure_monitor.json" \
  --recovery-root "${RECOVERY_ROOT}" \
  --project-dir "${RECOVERY_WORKTREE}" \
  --source-commit "${RECOVERY_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL FAILED CLOSURE RECOVERY" \
  --output "${RECOVERY_ROOT}/recovery_spec.json"

python -s "${RECOVERY_WORKTREE}/scripts/submit_hcwdl_campaign_recovery.py" \
  --recovery-spec "${RECOVERY_ROOT}/recovery_spec.json" \
  --output "${RECOVERY_ROOT}/dry_run_ledger.json"
```

Read the printed retry list and the dry-run commands. Completed parents must
not appear. Then submit the same immutable specification:

```bash
python -s "${RECOVERY_WORKTREE}/scripts/submit_hcwdl_campaign_recovery.py" \
  --recovery-spec "${RECOVERY_ROOT}/recovery_spec.json" \
  --output "${RECOVERY_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL FAILED CLOSURE RECOVERY"
```

Recovery task attestations are written under `${RECOVERY_ROOT}/tasks`; model
reports, checkpoints, locks, and final reports retain their canonical parent
campaign paths. If a recovery task exposes another source defect, preserve the
parent and recovery artifacts and create another source-pinned recovery rather
than editing either checkout or releasing stale dependencies.

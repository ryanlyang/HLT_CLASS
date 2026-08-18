# HCWDL-MHPE Full-Data D066 Schedule Screen Runbook

This runbook creates one immutable v3 full-data C25P75 schedule screen, runs a
nonmutating dry run, and submits its 60 independent fits plus aggregate and
completion jobs. It does not require the source MHPE campaign to be complete;
campaign creation authenticates every required teacher product.

Use a clean detached worktree at the exact pushed commit. Discover candidate
sources by authenticating full-data C25P75 campaign specs and selecting the
single candidate whose required U000, U050, and U100E/T2 products are already
complete. Never select a `*300k*` campaign.

Creation phrases:

```text
AUTHORIZE HCWDL MHPE FULL C25P75 D066 20 SCHEDULE SCREEN EXACT SPEC
AUTHORIZE HCWDL MHPE FULL C25P75 D066 SCHEDULE SCREEN CARRIED OPERATIONAL EVIDENCE
SUBMIT HCWDL MHPE FULL C25P75 D066 20 SCHEDULE SCREEN EXACT LEDGER
```

Canonical commands are emitted in the final handoff after the exact commit is
pushed. The generated command plan must show 60 jobs named `hcwschf_00` through
`hcwschf_59`, each with `--cpus-per-task=8`, `--mem=96G`,
`--time=72:00:00`, `--gres=gpu:gh200:1`, and `--signal=B:USR1@120`.

Monitor and cancel only through the campaign-bound ledger:

```bash
python -s "${WORKTREE}/scripts/monitor_hcwdl_mhpe_schedule_screen.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --submission-ledger "${SCREEN_ROOT}/submission_ledger.json" \
  --output "${SCREEN_ROOT}/monitor_report.json"

python -s "${WORKTREE}/scripts/cancel_hcwdl_mhpe_schedule_screen.py" \
  --submission-ledger "${SCREEN_ROOT}/submission_ledger.json" --execute
```

Do not use broad job-name cancellation.

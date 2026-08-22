# HCWDL-MHPE TRI60 CE60 control runbook

This companion queues one independent full-data, 60-pass, exact-HLT CE-only
fit. It does not modify or cancel the source TRI60 campaign.

After committing and pushing the implementation, create a clean detached
worktree at that exact commit. Set `SOURCE_SPEC` to the immutable running
TRI60 `campaign_spec.json`, then create and inspect the control:

```bash
python -s "${CE_PROJECT}/scripts/create_hcwdl_mhpe_tri60_ce_control.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${CE_ROOT}" \
  --project-dir "${CE_PROJECT}" \
  --source-commit "${CE_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL MHPE TRI60 CE60 CONTROL EXACT SPEC"

python -s "${CE_PROJECT}/scripts/submit_hcwdl_mhpe_tri60_ce_control.py" \
  --spec "${CE_ROOT}/control_spec.json" \
  --output "${CE_ROOT}/dry_run_submission_ledger.json"
```

Live submission is separately phrase-bound:

```bash
python -s "${CE_PROJECT}/scripts/submit_hcwdl_mhpe_tri60_ce_control.py" \
  --spec "${CE_ROOT}/control_spec.json" \
  --output "${CE_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL MHPE TRI60 CE60 CONTROL EXACT LEDGER"
```

Inspect only the exact submitted ID:

```bash
CE_JOB="$(python -s - "${CE_ROOT}/submission_ledger.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["jobs"]["train_M0CE60"])
PY
)"
squeue -j "${CE_JOB}" -o "%.18i %.40j %.2t %.10M %R"
```

Completion is the presence of an authenticated
`training/M0CE60/training_report.json` plus its selected and final checkpoint
and task attestation. No final-test command exists.

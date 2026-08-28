# TRI60 D000 Floor-Tail Confirmation Runbook

This is an isolated, one-fit full-data schedule comparison.  It does not
cancel, reprioritize, or depend on any live four-spine or TRI60 job.  Its Slurm
nice value is 10000 so the main campaigns remain ahead of it within the same
account/QOS ordering.

The reference is the existing budget-screen condition `P90_H45_LR3E4`.  The
new fit is `P100_H45_D60_FLOOR_ES15`.  The reference fit need not be complete
to launch the new fit, but both reports must exist to print the comparison.

## Local validation

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe -m pytest -q -p no:cacheprovider `
  tests/test_hcwdl_tri60_d000_floor_tail.py `
  tests/test_hcwdl_tri60_d000_budget_screen.py `
  tests/test_hcwdl_tri100_spine4.py
```

## Tigris creation, dry run, and submission

Use the exact clean, pushed commit.  The established reference screen is:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_tri60_d000_opt_9fdac874_r1/campaign_spec.json
```

Create a detached worktree, run `bash -n` on the worker, create the authorized
campaign, materialize its dry ledger, and then submit with the exact phrases
defined in the campaign module.  The resulting queue contains two short CPU
gates, one single-GH200 fit, and one short completion job.

## Print results

```bash
python -s "${PROJECT_DIR}/scripts/print_hcwdl_tri60_d000_floor_tail_results.py" \
  --campaign-root "${FTC_ROOT}"
```

The printer authenticates both reports and shows completed/selected passes,
accuracy, AUC, macro R50, deltas from P90/H45, and AUC at passes 60, 75, 90,
and 100 when available.

# Reconstruction repeatability validation

Evidence recorded2026-11-14. Implementation is ready to queue Hydra setup.
No new campaign was submitted during implementation. Actual four-card CUDA/
NCCL qualification and measured runtime/memory acceptance remain mandatory in
setup before automatic release; local tests do not establish those results.

## Local execution

Windows Python3.10.19, PyTorch2.5.1+cu121, installed Weaver0.5.3 from
`tmp/r8-followup-weaver`, local RTX2000 Ada Laptop GPU. Commands:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_r24_diagnostic_hydra.py tests/test_luka_r24_curriculum_hydra.py tests/test_luka_r24_gpu_benchmark.py
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_reconstruction_repeat_hydra.py
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -m pytest -v -p no:cacheprovider tests/test_luka_reconstruction_repeat_hydra.py -k 'exact_recipes or stability or full30'
```

- Baseline before changes: **143 passed,4 skipped**,314.15s.
- First new suite:32 passed,1 skipped,1 failed,354.19s. The failed test compared
  control metadata against the reconstruction-followup registry rather than
  the actual original general C0 dispatcher. Corrected the test expectation;
  scientific code/objective did not change for that correction.
- Targeted final recheck: **3 passed**,129.28s, covering the corrected registry,
  stability summaries and full30-epoch distributed training/results/progress
  after the last reporting edit. Across the new suite, **33 distinct tests now
  pass;1 four-CUDA/NCCL test is skipped locally**. Logs are
  `tmp/reconstruction_repeat_baseline.log`, `tmp/reconstruction_repeat_tests.log`
  and `tmp/reconstruction_repeat_recheck.log` (local evidence, not runtime inputs).
- Bash syntax passed for queue/worker scripts and both runbook blocks. Python
  parsing, new-document links and all new-file whitespace checks passed.
  Scoped `git diff --check` passed for edited shared documentation.

## What was exercised

- All18 cells, original R20/R22/R24/R28 coefficients and D20 schedule, fresh
  seed indices4/5/6, and matched native/predictor/decoder/order/dropout starts.
- Independent full-graph versus chunked-VJP/AdamW comparisons for allfive
  reconstruction recipes, including zero-encoder decoder warm-up and weight10.
- Genuine installed-Weaver CUDA two-epoch production miniatures for allsix
  recipes, full native export/reload and R28's actual doubled token memory.
  Compact CPU model miniatures additionally test artifact corruption.
- Four-process CPU/Gloo execution of all24 common-start comparisons with the
  actual decoder and a surrogate native model; full30-epoch paired training
  with the production loop, actual decoder, native-only export, completed
  reporting, provisional epochs and tamper rejection. These do not substitute
  for real four-GPU SyncBatchNorm/NCCL qualification.
- Mocked exact Slurm dry review/submit/release of2 bootstrap +18 fit +1 report
  jobs, dependency-free fits, afterany report, idempotence, wrong authorization,
  failed setup and partial-journal rejection, resource limits, source/parent
  authentication and typed GPU allocation rejection.
- Incomplete seed/control/window coverage, late-window mean and52% streaks,
  selected versus same-epoch deltas, and no inference in reporting.

The production scientific modules used here have no diff between58d0c0ed and
the implementation donor4a61a34c for the relevant native model, reconstruction
config/loss/targets/VJP, optimizer and distributed helpers. Runtime still loads
them explicitly from the preserved58d0c0ed checkout.

## Next execution step

Commit and push the new files, then use the
[runbook](LUKA_RECONSTRUCTION_REPEAT_HYDRA_RUNBOOK.md) to register a fresh root
against benchmark1094dea4 and queue setup plus durable CPU release. Successful
Hydra setup automatically queues the18 fits and report. No new export or
Tigris submission is required for this explicitly authorized Hydra campaign.

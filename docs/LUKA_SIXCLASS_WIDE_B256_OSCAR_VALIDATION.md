# Wide-range batch-256 Oscar validation evidence

Implementation dated 2026-11-23. Scientific authority is the
[active plan](plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md); execution instructions
are in the [runbook](LUKA_SIXCLASS_WIDE_B256_OSCAR_RUNBOOK.md).

## Data evidence

Local scalar-only audit authenticated the original foundation and frozen extended
role population, then selected QCD plus five hadronic signal labels and [200,2600).
Owned population hash:
`57eb7febd6ee6ae2c9729bdcc5f58a4f34dea232d7b1eaf12b7901ca405025d1`.
Eligible TRAIN2,488,148; validation497,954 plus68,078 outside; sealed in-range
test1,227,412. Population and sampling witnesses are under ignored local
`artifacts/audits/sixclass_wide_b256_oscar_v1/`.

Recipe hash:
`69df385334411bd372031c3ad8b7e42714eae5c52695270f23d5452135b47b41`.
Sampling audit hash:
`17e981ab04314cf6c80902e2341640669656ed8eb9e0245663ac380607264965`.
The audit did not read particle branches, perform inference, or evaluate final test.

## Local checks

The existing six-class and three-fit low-LR suites passed **89 tests** before
the new campaign was finalized. Initial sandbox attempts could not read the
local Weaver installation (`weaver.nn` unavailable); rerunning with access to
the isolated pinned Weaver0.5.3 installation resolved that environment issue.

Final full owned suite: **61 passed** in 124.46 seconds, including genuine installed-Weaver FP32
parity, CUDA/BF16 forward/backward and two-epoch miniature/reload for all six
production architectures with synthetic short particle inputs. The host RSS
witness is stubbed in those Windows CUDA tests; GPU memory is actually measured.
This is not full-dataset L40S memory or runtime qualification.

Production `population.load(..., production=True)` replay passed against the
frozen population above. All 85 pinned donor blobs were checked against Git
and a Linux-line-ending fixture; one local historical file has CRLF-only
differences, with no content drift. Python AST parsing, Bash syntax for all five
shell entrypoints, CLI help, owned documentation links and touched-document
whitespace checks passed. Per-epoch full-summary corruption, primary-selection
independence and distinct training/export batch timing are covered explicitly.

Logs: `artifacts/testing/wide_b256_baseline_v4.log`,
`wide_b256_final_v1.log`, `wide_b256_population_replay.log` in that same folder.

Coverage also includes synthetic ROOT view construction with full particle
support, forbidden-role particle-read guards, six-worker publication with fresh
teacher banks, exact main-FC CE gradients, auxiliary gradients/native inference,
duplicate occurrence negatives, optimized/serial/parallel matching equality,
twenty-epoch mini fits, export-only retry and exact epoch-state continuation,
changed-hash corruption rejection, submission journals, locks, dependency DAG,
and real Bash detached-launch lifecycle with mocked scheduler/runtime commands.

Reproduce focused tests (PowerShell, use an installed Weaver environment):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:PYTHONPATH="$PWD\src;$PWD\artifacts\testing\wide_b256_weaver_053"
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -B -s -m pytest -q -p no:cacheprovider tests/test_luka_sixclass_wide_b256_oscar.py
```

## Remote boundary

No Oscar job was submitted and no remote checkpoint/result was modified during
implementation. No real-data Tigris miniature was run for this Oscar-specific
campaign. Actual L40S acceptance is integrated into each science allocation:
installed-Weaver parity, batch256 full-support stress, batch128 export checks,
native-head gradient audit and measured memory/walltime envelope. Failure saves
diagnostics and stops that stage; poor scientific scores never trigger a failure.

Remaining operator steps: commit/push the scoped campaign, supply current HOME
quota, and invoke the detached launcher. This is queue readiness, not a claim
of completed remote production validation.

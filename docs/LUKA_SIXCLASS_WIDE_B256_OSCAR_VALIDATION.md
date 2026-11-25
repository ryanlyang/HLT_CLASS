# Wide-range batch-256 Oscar validation evidence

Implementation dated 2026-11-23. Scientific authority is the
[active plan](plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md); execution instructions
are in the [runbook](LUKA_SIXCLASS_WIDE_B256_OSCAR_RUNBOOK.md).

## November 25 recovery evidence

Supplied Oscar log: CONCAT9544072 failed with CUDA OOM during its full-support
batch256 stress, after25,832s building resident inputs. It requested another
5.43GiB with43.89GiB already in use on a44.39GiB-visible L40S. There are zero
committed CONCAT epochs. OFFLINE9544070/M0HLT9544071 have published results.
The current supplied quota was65.88GiB HOME usage. No independent remote checks
or changes were performed for this repair.

The SPEC/v2 recovery keeps those two results under their original parents and
trains only the four unfinished stages. Recipe hash remains exactly
69df385334411bd372031c3ad8b7e42714eae5c52695270f23d5452135b47b41.
The immutable population and all sampling/optimizer/selection semantics remain
unchanged. The new donor-manifest entry pins the original stride-preserving
pair saved-tensor helpers; no shared donor implementation was edited.

Verification after implementation:

- Existing suite:60 passed,1 reporting failure in318.61s. The failure was a
  local helper shadowing the newly imported recovery module; fixed by an
  explicit import alias. Its regression now passes.
- Initial recovery suite:11 passed in130.55s.
- Expanded recovery and affected regression rerun:15 passed,59 deselected
  in227.71s. This includes genuine Weaver0.5.3 CUDA FP32/BF16 native versus
  offloaded three-step AdamW parity, parameter/input gradients, BatchNorm
  buffers, updated weights/state and evaluation output, plus the optimized
  production worker's two-epoch miniature/reload and short-support batch256
  acceptance. Local GPU: RTX2000 Ada Laptop,8GiB; Linux RSS is stubbed in local
  integrated memory tests. These are not production L40S peak measurements.
- Registration and actual-script scheduler review:4 passed,72 deselected
  in39.51s. All76 distinct tests (61 original plus15 recovery) have passed
  across these regression/focused runs. Real synthetic ROOT witness tests
  prove only one authorized row from each of TRAIN/in-range/outside validation
  is read. A forced early OOM proves the full input loader is never called.
- Tests preserve original JSON bytes, reuse original bank spec parents through
  real tiny tail-worker fits, preserve both anchors in results/progress, reject
  modified recipes/parent bytes/live jobs/committed progress/duplicate recovery
  claims, and exercise actual detached Bash registration/queue calls with a
  mocked scheduler/runtime. No remote submission is part of these tests.
- All23 owned Python/CLI/test files parse; CLI help exposes the explicit parent
  recovery option. All six runbook Bash blocks and worker/launcher shebangs,
  Unix line endings and syntax validate. Owned links and scoped whitespace pass.
- A separate bounded synthetic memory check used the full production architecture,
  batch8,192 particles and three AdamW updates. CUDA peak allocation decreased
  from446,564,864 to258,334,208bytes (about42%). All147 saved CUDA tensors were
  restored during backward. Evidence:
  `artifacts/testing/wide_memory_synthetic_benchmark.json`. This is a local memory
  witness, not a batch256/L40S capacity or throughput prediction; short timing
  values include different warmup effects and are not a speed comparison.

Initial sandbox test attempts could not access the installed Weaver package;
an elevated attempt also hit the existing pytest temporary-directory ACLs.
Runs with that existing package accessible and new isolated workspace test
directories resolved both local-environment issues. Fixture directories are
under `artifacts/testing/wide_memory_*`; no runtime package was installed or
modified by the repair.

The saved scalar inventory gives CONCAT maxima596 TRAIN,467 in-range validation,
531 outside validation. Recovery will authenticate and read the corresponding
real rows, then exercise full support before loading millions of particles.
Full-objective host/GPU/timing acceptance still runs later in the same job.
CPU transfer overhead and whether full-support batch256 fits with92% GPU
headroom must be measured on Oscar. No Tigris or Oscar real-data miniature was
run locally, and no final-test particles or inference were used.

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

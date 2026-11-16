# Reconstruction stability validation

Local validation completed2026-11-15. The implementation is ready to commit,
push and queue its measured Hydra setup. No remote jobs were submitted by
this implementation step. Real four-GTX1080Ti qualification remains mandatory
inside setup before the automatic release can submit scientific fits.

The registered scientific design is the [42-fit stability plan](plans/LUKA_RECONSTRUCTION_STABILITY_IDEAS.md).
Operational commands are in the [runbook](LUKA_RECONSTRUCTION_STABILITY_HYDRA_RUNBOOK.md).

## Focused evidence

Final command, with bytecode and pytest cache disabled:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_reconstruction_stability_hydra.py
```

Result: **45 passed,1 skipped in237.52s**. The skip is specifically the real
four-CUDA/NCCL/SyncBatchNorm qualification; this workstation exposes one CUDA
device. It is not replaced by the CPU/Gloo tests. The unchanged repeatability
donor baseline was also checked before implementation:33 passed,1 skipped.

The focused suite covers:

- Exact14-configuration/42-fit order,15 controls, seeds7/8/9, schedules and
  effective128/256/512 update counts. All42 initial states prove paired native,
  predictor, decoder, sampler and rank-dropout identities across recipes.
- Independent full-graph gradients and AdamW moments against the chunked VJP
  worker; unequal and partial accumulated groups; globally reduced gradient
  cap/projection; zero-norm cases; bounded communication buckets; decoder-only
  step counters with native state/RNG isolation.
- Four-process Gloo comparisons for the gradient interventions, accumulation,
  decoder steps and SOFT. A separate complete70-case qualification simulation
  exercises all14 recipes at positions3,3.01,4.5,8.99,9, complete common-start
  restoration, saved reference digests and tamper rejection.
- Actual installed-Weaver CUDA execution of all14 configurations through
  two-epoch production miniatures, four separately selected native-only
  exports, disk reload, probability replay and metric authentication. These
  use synthetic rows and a fixed test temperature scale; they are not real
  dataset or four-card resource measurements.
- A distinct frozen512-TRAIN-identity calibration test, including temperature
  scale publication, RNG preservation, numerical qualification, invalid-role
  and modified-calibration rejection. CUDA transport stress covers298 slots,
  empty/one/170/298-particle targets and positions3.01,4.5,8.99. Finite
  differences, real/null marginal masses, dual gap, low-temperature hard limit
  and deliberate nonconvergence are checked separately.
- Native EMA example-clock updates, TRAIN-only BN recalibration on copies and
  unchanged live buffers/RNG. A four-process full30-epoch paired synthetic
  test exercises all readouts, their distinct availability/selection, native
  export/reload, authenticated results/progress and late-window reporting.
- Exact45-job automatic submission simulation, full-envelope review, measured
  requests, interleaved independent fits and afterany report. Duplicate release
  is idempotent; failed setup, wrong plan, enlarged GPU requests, ambiguous
  partial submission, wrong allocation and duplicate launcher claims stop.
- Resource/storage coverage and failure injection; fresh protected roots;
  pinned source/environment identities; read-only reporting while future-write
  storage is insufficient; partial-seed denominators and matched readouts.

The expanded calibration test initially exposed slow ordinary Sinkhorn
convergence for a null-mass case. The final protocol retains500 total iterations
and both1e-4 tolerances, with ordinary log scaling for50 iterations followed by
relaxation1.8 when necessary. The final full suite includes that correction.
The scientific objective and temperature schedule are unchanged. Residual/gap
diagnostics retain maxima across chunks rather than averaging away violations.

## Source, storage and static checks

An isolated temporary extraction of the exact historical58d0c0ed `src/` loaded
all16 new runtime modules and instantiated the42-fit recipe and model inventory
against the original science. All27 direct donor import paths exist at that
commit. The original source was not edited or replaced.

The serialized298-slot inventory in that isolated environment requires29GiB
for outputs plus a separate1GiB reserve. This is a registration projection;
Hydra setup measures real output sizes, complete validation, BN recalibration,
longest-support memory and each expensive training phase before accepting it.
No new walltime or AUC measurement is claimed from the historical benchmark.

Python AST parsing, CLI help, Bash syntax, shell LF endings, scoped whitespace
and repository-relative links in the edited documentation passed. Tests and
static checks leave unrelated user changes intact. No contract or scientific
meaning in an existing campaign was changed; this campaign has its own
`LUKA_RECONSTRUCTION_STABILITY_<KIND>/v1` family.

## Remaining execution boundary

The live scheduler dry review, real four-card NCCL/SyncBatchNorm comparison,
full299234/74895 input replay, actual per-card/host memory and walltime checks,
42 genuine setup miniatures, per-fit startup checks and full scientific fits have **not** run
locally. No Tigris miniature is claimed; this active plan targets Hydra.

The runbook registers the exact pushed source; the queue helper reviews it and
submits setup and a durable dependent CPU release. That release authenticates the
successful setup and measured acceptance before submitting42 fits and the
report. Failed technical qualification preserves diagnostics and submits no
science; low finite recovery never fails qualification. There is no independent
test result, performance guarantee or automatic retry.

# Small-data subclass Hydra validation

Implementation evidence, 2026-11-10. Exact internal donor base:
`5571240c8c42fce1103344dff94c88bf8edf13c5`. This is a new additive campaign;
existing runtime, original source worktrees and jobs are preserved.

## Local evidence

Before implementation, the focused donor suites passed **63 tests in70.74s**:
`test_luka_embedding_objective_hydra.py` and
`test_luka_fullstat_subclass_targets.py`.

The new `test_luka_subclass_hydra.py` suite passed **62 tests in77.82s**.
After explicit duplicate-ID and class-label validation tightening, the focused
target-reader rerun passed **8 tests in11.76s** (54 deselected).
Tests used Python3.10, Torch2.5.1, a locally installed Weaver0.5.3 fixture at
`tmp/hydra-weaver-fixture-20261102`, and the local RTX2000 Ada laptop CUDA GPU.

Coverage includes:

- Independent hand-formula checks for all ten objective coefficients, the
  three-epoch ramp, true-class conditional normalization and detached targets.
- All ten complete30-epoch synthetic fits, original checkpoint selection,
  strict native reload, saved-prediction replay and corruption rejection.
- Original U100-w5 complete model/AdamW update parity on CPU and CUDA with
  subclass gradients disabled for this technical check. Auxiliary-head
  initialization preserves the original native model and U100 predictor.
- All ten full-depth8+2-block CUDA forward/backward paths at batch128 and
  partial11, with unchanged native inference after auxiliary-head removal.
- Gradient probes give identical model/optimizer/RNG results to an ordinary
  update; subclass-only gradients reach the shared encoder without directly
  training the original classifier or U100 predictor heads.
- Actual target-building and reading functions on synthetic small-data T00
  records: immutable shared bank, TRAIN-only role, ordered joins, recipe and
  file hashes, and corrupt-source/prototype rejection. Original pure-kernel
  tests cover reproducibility,75/25 geometry, softness calibration and finite
  degeneracy without a quality gate.
- Complete production setup functions and every miniature on synthetic CUDA
  data, plus disposable startup and full scientific fits at weights0.5/100.
  External source/Slurm/site boundaries are mocked in these local tests.
- Resource coverage, memory and walltime rejection, preservation of training
  memory peaks across deployment-only counter resets, and no accuracy gate.
- Exact-ID setup/release/ten-fit/report flow:13 fake scheduler submissions,
  no inter-fit dependencies, no duplicates on completed retry, and failed
  setup or ambiguous partial state submitting no additional science.
- Pinned T00-only parent authentication, source/additive registration checks,
  completed/provisional results without file writes, and shell/Python parsing.

All required runtime/documentation files were checked for syntax and whitespace;
the CLI registry matches the registered ten weights. Runbook Bash blocks parse,
Markdown links resolve, and pre-existing unrelated edits remain intact.

Local command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -x -p no:cacheprovider tests/test_luka_subclass_hydra.py
```

## Remote acceptance still belongs to setup

No research training, inference, commit, push or Slurm submission was performed
by this implementation task. Synthetic results do not estimate physics gains,
A6000 throughput or campaign start times. No real Tigris/Hydra miniature is
claimed. The active plan uses installed-Weaver A6000 qualification for this
Hydra-only campaign rather than the legacy Tigris-only workflow.

After the scoped commit/push and initial queue authorization, setup must
authenticate the original small-data T00 and C0, build the real TRAIN target
bank, pass original native/U100 parity and historical numerical replay, run
every actual-target miniature, and accept measured resources. Only then can
the dependent CPU release submit the ten independent science fits. A finite
poor accuracy or auxiliary diagnostic is a result, never a reason to skip an
arm. Results use development validation and an overlapping original50k subset;
final test remains sealed.

Exact next step: follow the [runbook](LUKA_SUBCLASS_HYDRA_RUNBOOK.md).

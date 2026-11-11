# Small-data subclass Hydra validation

## Separate GTX 1080 Ti benchmark

2026-11-10: added a bounded benchmark in its own contract namespace, plan and
[queue runbook](LUKA_SUBCLASS_GPU_BENCHMARK_RUNBOOK.md). It cannot release the
scientific sweep. Existing BF16 production tests passed73/73 in72.29s before
the benchmark changes. The combined suites passed93 tests with1 skip in80.51s;
the final benchmark-only suite, including two new phase-failure orchestration
cases, passed22 tests with1 skip in17.78s.

New evidence covers global padding and odd-batch gradient weighting, a real
two-process CPU Gloo comparison, actual-target payload schema/hash/role rejection,
independent benchmark identity, exact two-card allocation checks, full dry/live
submission idempotence, partial-claim rejection, sequential phase execution
including one-card failure, and honest reporting of failed phases. Local
installed-Weaver checks exercise FP32 forward/backward/AdamW and full128/partial
GPU batches. Two-GPU NCCL/SyncBatchNorm parity is implemented as a test and as
part of the real benchmark, but is skipped locally: this workstation has one
RTX2000 Ada laptop GPU. No actual 1080 Ti performance measurement exists yet.

Python/shell syntax, all32 required files, two benchmark Bash examples, new
Markdown links/fences and whitespace passed. No donor runtime was changed;
donor/source details are in LEGACY_SOURCE_MAP. Nothing was committed, pushed
or submitted remotely. Real Brown benchmark completion remains the next step;
it would still not qualify a new scientific FP32 profile by itself.

## GPU-profile execution v2

2026-11-10 follow-up to pushed c8e15195: the user's scheduler evidence shows
A6000 maintenance drain; L40 and L40S are schedulable types but fully allocated
in the supplied snapshot. Original scheduler dry review submitted no jobs.
New registrations can select a6000, l40 or l40s, with fresh v2 receipts and
authorization. Scientific RECIPE and TARGET_RECIPE hashes remain byte-identical
to c8e15195. No objective, precision, batch size or epoch change.

Before edits, **62 tests passed in77.93s**. After edits, the focused suite
passed **73 tests in78.48s**, including all existing synthetic CUDA/Weaver
checks. Added coverage validates registration and automatic13-job release for
all three profiles, real controller/device check code with mocked evidence,
wrong-device/compute-capability/BF16/allocated-GPU rejection, fresh-profile
enforcement, v1 execution rejection and preserved scientific recipe hashes.
No local identity mock is evidence of real L40/L40S qualification. Python
syntax, both shell scripts, three runbook Bash blocks, links and whitespace pass.

No commit/push or remote jobs were performed. Follow the updated runbook for
one new execution. Its actual chosen GPU must pass the existing native replay,
full model/update parity, real-target miniatures and measured resource checks
before the durable CPU release submits science. Historical A6000 C0 remains the
reference; numerical replay is not a proof of identical cross-GPU training.

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
A6000/L40/L40S throughput or campaign start times. No real Tigris/Hydra miniature is
claimed. The active plan uses installed-Weaver selected-GPU qualification for this
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

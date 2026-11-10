# U100 embedding objective validation

Implementation evidence is recorded here; real Brown qualification belongs
to the campaign's setup job. No local test establishes scientific benefit.

Baseline before changes: 116 donor summary-token/curriculum tests passed in
123.31s. Final focused suite: **57 passed in 80.33s** on 2026-11-10.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -x -p no:cacheprovider tests/test_luka_embedding_objective_hydra.py
```

Coverage includes:

- All eight objectives, centered-class invariance, projector/nullspace behavior,
  Fisher diagonal against explicit derivatives, bounded geometry, degenerate
  cases, fixed gradient-RMS calibration and detached targets.
- Hand-checked conflicting/nonconflicting gradient projection, untouched
  classifier/predictor gradients and unused-parameter behavior.
- Original U100 initialization, forward, complete model update and AdamW-state
  parity on CPU/CUDA; all eight full-depth CUDA full128/partial11-jet updates.
- Eight full30 synthetic fits with native export/reload, authenticated epoch
  history and target lineage. Poor finite auxiliary scores still complete30.
- Production geometry and evaluation-bank build/readback, corruption/role
  rejection and validation diagnostics preserving native probabilities/RNG.
- Production setup functions and all eight miniature/measurement paths on
  synthetic CUDA inputs. Scientific R4/R8 workers execute discarded startup
  miniatures followed by fresh full30 fits and exact-ID receipts. Memory
  regressions verify earlier training peaks survive deployment counter resets.
- Complete bootstrap and automatic science/report submission with exact IDs,
  bounded dry review, dependency/order checks, failed setup rejection,
  completed-submission idempotence and ambiguous partial-submission rejection.
- Registration/source drift checks and read-only completed/provisional reports.

Python syntax, standalone shell scripts and runbook shell blocks are checked
locally, as are required file inventory, relative documentation links and
whitespace. CLI registration is inspected without invoking a scheduler.

Environment: Windows Python3.10 `tagging-hlt`, Torch2.5.1, the local Weaver0.5.3
fixture, NVIDIA RTX2000 Ada Generation Laptop GPU. CPU tests use one thread.
Full30 fixtures use a reduced-depth model and 22 synthetic jets; the separate
full-depth CUDA checks retain eight particle/two class blocks but use short
synthetic particle sequences. Setup/worker tests mock external source/Slurm,
site GPU identity and host memory boundaries. These are not installed Brown
environment checks, real A6000 resource measurements or scientific results.

Mandatory real setup: exact pushed clean source, authenticated existing input
and C0 lineage, frozen U100 encoder/validation transfer, full-population TRAIN
geometry/calibration and evaluation target creation, native parity/replay,
all eight production miniatures, warmed update/diagnostic measurements and
memory/runtime acceptance. Only that successful setup releases science.

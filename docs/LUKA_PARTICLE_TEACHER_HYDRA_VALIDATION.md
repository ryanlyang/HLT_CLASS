# Teacher particle prediction: validation and remaining execution

Implementation baseline: `cc55765a5e6a80ed04b660d99dc182a4aaf4b017`.
New runtime is additive under `luka_particle_teacher_hydra`; existing runtime,
campaigns and saved artifacts are preserved. Contracts are
`LUKA_PARTICLE_TEACHER_HYDRA_*/v1`. Donors are recorded in
[LEGACY_SOURCE_MAP.md](LEGACY_SOURCE_MAP.md).

## Local evidence, 2026-11-05

Environment: Python 3.10.19, Torch 2.5.1, Weaver-core 0.5.3 from the installed
fixture `tmp/hydra-weaver-fixture-20261102`. CUDA used an NVIDIA RTX 2000 Ada
Generation Laptop GPU. These are synthetic local tests, not Hydra throughput
or real-data scientific results.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_particle_teacher_hydra.py
```

Result: **57 passed in 48.10s**. Coverage includes:

- Extraction from the active privileged `teacher_m16.oracle` path at blocks 6/8,
  preserving HLT particle identities and masks. Changing the privileged message
  changes targets; identical extraction reproduces hashes and statistics.
- Frozen teacher/encoder, TRAIN-only channel statistics, ragged slot alignment,
  per-jet weighting, masked NaN padding, particle permutation and centered loss.
- Exact native FP32 inference and initialization parity against AuxiliaryParT,
  preserved RNG, gradient flow from prediction heads into the encoder, removed
  auxiliary heads at deployment and rejection of privileged model inputs.
- All twelve production objectives with checkpoint publication/reload, plus
  production-width CUDA full/partial batches for control, multi-depth and
  centered variants. A full 30-epoch synthetic fit verifies selection/history.
- Exact setup/science worker paths with scheduler and data boundaries mocked;
  actual training/publication kernels, disposable warmup and fresh measured
  miniatures, unchanged initialization between warm and measured fits.
- Content-hash/target/receipt failures; resource projection from warmed timings;
  runtime/RAM violations; poor finite accuracy is never an admission gate.
- Independent science jobs, exact setup/release/report dependencies, durable CPU
  release, completed-stage idempotence and refusal of ambiguous partial submits.
- Hydra's one-requested/two-allocated CPU case and wrong cluster/job/resources;
  original-worktree teacher proof; immutable source/spec checks; read-only epoch
  reporting with same-seed controls.

Before editing, the three related suites produced 123 passes and one failure:
the existing CUDA particle-prediction test required `CUBLAS_WORKSPACE_CONFIG`
and that initial command omitted it. Rerunning that test with the variable set
passed (1 passed in 20.61s). No donor code was changed for this environment issue.
The worker exports this variable before starting Python.

Final related regression run with the CUDA variable set:

```powershell
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_shared_message_hydra.py tests/test_luka_cross_message_hydra.py tests/test_luka_particle_prediction.py
```

Result: **124 passed in 136.32s**. Together with the new focused suite,
**181 tests passed**. Both Bash entrypoints passed syntax checks; CLI help,
Python AST parsing, all18 required files, new documentation links and whitespace
checks passed. The earlier shared-message CPU fix and other existing dirty files
were preserved; this campaign did not edit any donor runtime.

## Real execution boundary

The existing A6000 parent qualification and completed teacher_m16/job145410 are
authenticated by their own original pinned worktree. Local evidence does not
replace a real miniature of the new path. The user's explicit Hydra request
specializes the repository's generic Tigris execution target to Hydra A6000;
the requirement for real production miniatures is retained.

The new GPU setup builds all TRAIN targets, records their exact hashes and
normalization, and runs warmed production miniatures for all twelve arms with
real worst-multiplicity full/partial batches. Actual preparation peaks and fit
timings must fit the registered envelope before the CPU release submits science.
Every supervised fit reconstructs the same targets in allocation RAM and checks
them against setup; controls do not load privileged particle views or weights.

No new remote job, training run, or Tigris miniature was executed here. No new
recovery improvement is claimed. The next action is to commit/push only the
new campaign runtime and follow the [runbook](LUKA_PARTICLE_TEACHER_HYDRA_RUNBOOK.md).
Existing Brown data, environment, parent worktree and teacher checkpoint are
required; no new SPORC transfer or teacher fit is needed.

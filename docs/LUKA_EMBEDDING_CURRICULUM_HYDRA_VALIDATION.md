# Embedding curriculum validation

Implementation date: 2026-11-06. Exact donor:
`b5464e73c94b322a81a33032baea18f1615a5a66`.
See the [plan](plans/LUKA_EMBEDDING_CURRICULUM_HYDRA_PLAN.md),
[contract](contracts/LUKA_EMBEDDING_CURRICULUM_HYDRA.md) and
[queue runbook](LUKA_EMBEDDING_CURRICULUM_HYDRA_RUNBOOK.md).

## Local evidence

Before changes, focused donor selection/registry/release checks passed:
48 passed, 35 deselected in 14.29 seconds. The initial complete new curriculum
suite passed 60 tests in 54.06 seconds, including CPU runs of all twenty full
30-epoch configurations and CUDA production-model transitions. The final
expanded curriculum plus reconstruction-release regression suite passed
**92 tests in 56.01 seconds**, including genuine local CUDA execution (no CUDA
cases skipped).

Static verification passed for all 18 required new files: Python AST, Bash
worker/queue and runbook block syntax, PowerShell runbook/test block parsing,
local documentation links, whitespace and the CLI's twenty-row registry.
Scoped tracked diffs confirm the pre-existing shared-message edits were not
part of this implementation. README, handoff and source map record the addition.

The focused suite checks exact clocks and distinct rows, all four optimizer/LR
transition combinations, full model/AdamW/RNG restoration, embedding-best versus
last selection, post-pretraining classifier selection, zero-weight decay
exclusion, native-only deployment and replay, TRAIN/validation role separation,
frozen TRAIN normalization, checksum/lineage rejection, warmed resource
projection, source inventory, exact Slurm journals, automatic bounded release,
failed admission and duplicate prevention after a lost scheduler response.

Local CUDA cases use the full native ParT configuration and 128-row plus
non-singleton partial batches with synthetic short particle lists. They exercise
E01, E12 and E16 transitions and original U100 one-update parity. They establish
local execution correctness, not full-population Hydra timing or memory.

Reproducible focused command from this Windows checkout:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider `
    tests/test_luka_embedding_curriculum_hydra.py tests/test_luka_reconstruction_release.py `
    --basetemp=tmp/curriculum-validation-fresh
```

The existing local Weaver fixture is a test dependency, not an authoritative
substitute for the installed Brown environment. Choose a fresh temporary test
directory for a new validation run.

## Required Hydra evidence

No remote campaign was registered or submitted during implementation. No full
dataset training, authoritative installed-Hydra replay, Hydra measurements or
Tigris miniature has run for this new campaign. The requested target is Brown
Hydra, with the runtime checking its pinned installed environment and A6000.

Before releasing science, genuine setup must authenticate all inputs and C0,
build the evaluation-only validation bank, prove installed-Weaver FP32 and
one-update model/optimizer parity, complete all nineteen warmed and measured
miniatures through production transition/export paths, and qualify memory and
walltime. The durable CPU job checks successful setup accounting, immutable
receipts and the exact measured plan before submitting independent science.
There is no scientific quality threshold. Failures preserve diagnostics and
cannot silently widen limits, change weights or train on validation targets.

Local completion means ready to commit/push and queue that real bootstrap.
Scientific success and measured Hydra suitability remain unclaimed.

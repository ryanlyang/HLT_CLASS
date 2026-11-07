# SupCon-5 validation boundary

The [two-fit plan](plans/LUKA_SUPCON5_DEBUG_PLAN.md) is implemented in the additive
`luka_supcon5` package, CLI, queue launcher and Slurm worker. No existing runtime
is edited. Exact donor: `00fbb92641553e8817b577b8765796e6618afc13`; reused/adapted
functions are recorded in [LEGACY_SOURCE_MAP.md](LEGACY_SOURCE_MAP.md).

## Local evidence

Before editing, the relevant supervision-depth teacher/SupCon focused checks
passed: 7 passed, 46 deselected. The initial new focused suite passed 18 tests,
including CPU and local CUDA teacher-to-student training, target export,
detached target routing, original U100 head parity, corruption rejection,
resource admission and dispatch failure handling.

Final focused suite: **25 passed in 24.39s**. This adds:

- Full native eight-particle-block/two-class-block CUDA updates at physical
  batch 128 and partial batch 11, with strict native export parity. Synthetic
  particle support is small; actual longest-support stress remains remote.
- Full30 teacher/student production-worker execution with selected native
  target export, real target-bank readback, startup miniature, task receipts,
  completed report and epoch histories; scheduler/site inputs are simulated.
- The real setup measurement code on synthetic CUDA data, including both
  miniatures, warmed steps, acceptance replay and retained failure diagnostics.
  A100 identity and Linux memory readings are simulated in this local check.
- Exact-DAG journals, resolved teacher/student/report dependencies, duplicate
  protection, ledger tampering, source compatibility and poor finite accuracy.

Focused donor regression selection: **57 passed, 94 deselected in 194.41s**,
covering the new suite as it then stood and applicable supervision-depth,
contrastive and pair-correction model/source/journal checks. The two subsequently
added setup/source tests passed in the final 25-test suite above.

Required 17-file inventory, Python AST, CLI help, Bash worker/launcher/runbook
syntax, runbook PowerShell syntax, relative documentation links and scoped
whitespace checks passed. README, handoff and donor-map entries are additional
documentation changes. No unrelated dirty work was staged or modified.

Local test execution uses Python from `tagging-hlt` and the existing local
`tmp/hydra-weaver-fixture-20261102` Weaver fixture. These are not the authoritative
installed SPORC environment or actual cluster inputs.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_supcon5.py
```

## Required remote evidence

No remote registration, GPU qualification, job submission or scientific result
is claimed. A read-only SSH attempt was denied at port22 by the current execution
environment; live debug capacity was not verified. Queue instructions are in
the [runbook](LUKA_SUPCON5_DEBUG_RUNBOOK.md).

The campaign intentionally runs on SPORC debug, as specified by its active plan.
Its real setup must establish installed-Weaver/environment parity, full native
initialization/reload equivalence, successful production-worker miniatures,
longest-support and partial-batch backpropagation, warmed timings and memory.
Acceptance must match the persisted diagnostic and remain within the reviewed
resource envelope before science is released. The real student separately
qualifies its selected-teacher targets before restarting from fresh weights.

No numerical teacher/student quality gate is imposed. Poor finite accuracy or
recovery is a scientific outcome. This single-seed, two-fit screen compares the
combined intervention against saved U100-w5; it cannot isolate the SupCon
coefficient without a newly matched CE-only teacher/student comparison.

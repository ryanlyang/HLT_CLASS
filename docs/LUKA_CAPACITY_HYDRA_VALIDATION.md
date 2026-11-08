# HLT capacity and complementarity validation

Local implementation evidence, 2026-11-08. This is readiness to queue real
Hydra setup, not a claim of remote execution or improved scientific recovery.

## Local environment and checks

Python: `C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`.
Torch2.5.1; NumPy2.0.1; sklearn1.7.2; SciPy1.15.2; Weaver0.5.3 fixture at
`tmp/hydra-weaver-fixture-20261102`. CUDA device: NVIDIA RTX2000 Ada Generation
Laptop GPU. Source bytecode writes were disabled. CUDA used
`CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Baseline selective suite: **73 passed in75.08s**. Final combined command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -m pytest -q -p no:cacheprovider tests/test_luka_capacity_hydra.py tests/test_luka_selective_teacher_hydra.py
```

Result: **130 passed in150.32s**. Subsequently added CPU rounding and batched
exact-ID accounting checks: **5 passed in7.08s**. Thus **135 distinct tests**
(62 capacity,73 selective) passed; no CUDA skips. CLI help, Python compilation
and Bash syntax also passed. See the runbook for production environment use;
the local fixture path is not a Brown environment override.

## Covered behavior

- Complete fixed2x3x2 matrix, same initialization/shuffle seeds within each
  comparison, exact loss terms and bounded16-job plan.
- Native128 dual-predictor state, losses and one optimizer update agree exactly
  with original selective S2IT. Predictor creation preserves encoder RNG.
- Every cell: actual Weaver CPU forward/backward, detached targets, encoder
  gradient routes, inference without heads/targets; full30-epoch fit with
  reduced block counts but registered128/256 widths; selected checkpoint,
  immutable epoch validation and width-aware export/reload.
- Both production widths and all three objectives: full eight particle/two
  class blocks on CUDA, batch128 and partial11 backward, native reload parity.
- Actual CUDA production miniatures for CE, U100, and U100+IT. Complete setup
  worker path across all12 cells using reduced-depth actual models, real CUDA
  timing/backward/replay/files and simulated Slurm/environment/A6000 identity
  and host-memory envelope. This is explicitly not a real A6000 measurement.
- Authentic selected IT export/bank validation, ordered TRAIN identity/label
  join, validation-role rejection and corrupted-payload rejection. CE-only
  never opens target banks. TRAIN/validation overlap is rejected.
- Exact identity alignment, duplicate/missing/label mismatch rejection; fixed
  probability ensembles, complementary and identical prediction fixtures,
  paired deterministic bootstrap, tied weighted AUC agreement with sklearn,
  finite poor predictions accepted without quality gates.
- CPU analysis artifact publication/readback, completed/provisional reports,
  missing controls, matched capacity interaction calculations and deployment
  profile integrity. No reference/oracle may become an ensemble member.
- Original parent authentication in its own pinned source process, sentinel
  checking, immutable registration round trip, changed target detection, and
  rejection of edits to donor runtime.
- Full dry/bootstrap/automatic science release with simulated scheduler:
  exact setup/release IDs, twelve independent fits, afterany report, idempotent
  completed ledgers, failed setup and ambiguous claims preventing submission.
- CPU guard accepts observed one-requested/two-allocated Hydra rounding but
  rejects extra CPUs or GPU resources. Status lookup uses only accepted IDs.

## Remaining production boundary

Commit/push the scoped additions, register the existing completed selective
parent on Brown and run the [queue procedure](LUKA_CAPACITY_HYDRA_RUNBOOK.md).
Real setup must authenticate the original environment/data/targets, pass
installed-factory parity and original native BF16 replay, run every actual
target miniature, measure all12 profiles and satisfy the resource envelope.
Only then does the durable release submit science. No job, commit or push was
performed locally by this implementation session. No final-test inference was
performed. Parameter/latency and scientific conclusions await real outputs.

The active [plan](plans/LUKA_CAPACITY_HYDRA_PLAN.md),
[contract](contracts/LUKA_CAPACITY_HYDRA.md), and
[donor map](LEGACY_SOURCE_MAP.md) record the scope and source lineage.

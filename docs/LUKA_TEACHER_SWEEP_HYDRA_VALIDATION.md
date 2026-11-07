# Teacher-sweep validation

Status: locally validated and ready to queue the real Hydra setup after source
commit/push and the native OFFLINE input delta transfer. This is readiness to
attempt qualification, not a claim that the24 scientific fits have qualified.

## Sources and contracts

All implementation is additive in `luka_teacher_sweep_hydra`, three scripts,
the focused test file and four campaign documents. README, HANDOFF and
LEGACY_SOURCE_MAP record the new work. No existing runtime was changed.
Donor commit: `0ae4f28fbc25df4aef0ddc617a16555f98b02591`. The donor map enumerates
source files and adaptations. All new artifact contracts are v1; clean TRAIN
target banks reuse `LUKA_EMBEDDING_AUX_BANK/v1` without changing its semantics.

## Local evidence

Environment: Python in `tagging-hlt`, repository Weaver fixture
`tmp/hydra-weaver-fixture-20261102`, CPU and NVIDIA RTX2000 Ada Generation Laptop
GPU (8GiB). `PYTHONDONTWRITEBYTECODE=1`, fixture plus `src` on PYTHONPATH, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8` were set.

Before editing, focused SupCon5/reconstruction-release tests passed51 checks
in63.77s. Final command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe' -B -m pytest -q -p no:cacheprovider tests/test_luka_teacher_sweep_hydra.py tests/test_luka_supcon5.py tests/test_luka_reconstruction_release.py --basetemp '.tmp/teacher-sweep-final' -x
```

Result: **116 passed in199.43s** (65 new campaign cases and51 donor/release
regressions). Tests cover:

- Twelve distinct teacher objectives; preserved native initialization and HLT
  deployment parity; exact SupCon5 donor formula and weighted-gradient scaling.
- Local augmentation RNG, token masks, partial batches, permutation-invariant
  physical summaries and unordered reconstruction, and correct gradient routes.
- All twelve teacher and common student paths with the full native8+2-block
  architecture, physical batch128, partial batches and native reload on CUDA.
  A separate masked decoder case uses298 output slots and batch128 with backward.
- Each teacher's two-epoch fit, selection, native checkpoint reload and TRAIN
  embedding export. CPU/CUDA full30 synthetic teacher-bank-student pipelines.
  A poor finite teacher completes instead of being rejected by accuracy.
- TRAIN-only statistics/banks, exact row joins, disjoint TRAIN/validation,
  coordinate/role restrictions, population mismatches and corruption rejection.
- Native OFFLINE cache export from synthetic ROOT inputs, relocation, exact
  historical shard bytes and prepared fingerprint validation. Historical
  source-authentication boundaries are mocked; payload validation is exercised.
- The setup measurement pipeline across all13 profiles, acceptance and saved
  failure diagnostics. This test uses synthetic rows, smaller fixture model
  dimensions and a mocked A6000 identity, not a real Hydra measurement.
- Production teacher/student workers with selected real synthetic teacher
  targets, startup miniatures, receipts, read-only progress/results and CPU
  report; mismatched job ownership is rejected. Allocation/registration
  boundaries are mocked, while fitting/export/report validation execute.
- Actual exact-DAG journal machinery with a simulated Slurm boundary: setup,
  durable release,24 fits and report; only own-teacher student dependencies;
  failed-setup exclusion; bounded resources; completed-ledger idempotence;
  partial/ambiguous submissions fail without submitting duplicates.
- Donor CPU-allocation tests cover Hydra's observed two-allocated/one-requested
  CPU representation and prevent GPUs from entering the release/report request.

Static checks cover the complete required-file inventory, Python AST/CLI,
shell scripts and all runbook Bash/PowerShell blocks, local documentation links
and whitespace. The CLI review prints the full authorized24-fit upper-bound
plan alongside the bootstrap dry-run ledger before confirmation.

## Remote qualification still required

No remote transfer, registration, inference, training or submission was run.
No existing campaign job or artifact was changed; no source was committed or
pushed. Real installed-Weaver/Hydra parity and real-data throughput remain
unverified. No new Tigris miniature was run: this campaign's active plan selects
Hydra A6000 and requires its genuine setup instead.

The real setup verifies original input lineage, exact installed environment,
row joins, native parity, production miniatures, target export, full validation
timing, representative/largest-support backward passes, host/GPU memory and
measured walltime. It must pass all technical checks inside the unchanged
32GB/48-hour fit envelope before the CPU release automatically submits science.
Each student then checks its selected teacher's actual TRAIN bank and runs a
disposable startup miniature. Miniature weights never initialize science.

GPU availability, user quota and scientific gain are not established locally.
Single-seed development validation with historical recovery anchors and an
overlapping original50k slice is exploratory. The final test remains sealed.
Use the [runbook](LUKA_TEACHER_SWEEP_HYDRA_RUNBOOK.md) for the exact next steps.

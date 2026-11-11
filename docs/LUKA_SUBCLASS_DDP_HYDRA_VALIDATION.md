# Two-GPU subclass validation record

Source baseline/donor: `1d83d69d4339599d0cf8c59a8b5cafe52704cd81`.
Implementation is additive under `luka_subclass_ddp_hydra`; original scientific
and benchmark runtime files remain unchanged. Scientific/operational authority
is the [active plan](plans/LUKA_SUBCLASS_DDP_HYDRA_PLAN.md).

## Supplied Brown evidence

The user reported job160948, `lukascb_probe`, COMPLETED0:0 in17m24s. Its output
reports two-GTX1080Ti FP32 S0p5/S100 updates at0.168/0.164s, train-only
projections3.27/3.20h, longest-support success and peaks6.61/7.20GiB. Single-card
updates were0.246s with longest-support failure at10.47GiB. Short deterministic
one/two-card parity passed at16/15rows. These are user-supplied benchmark
observations, not a locally reproduced or authenticated production acceptance.

## Local implementation checks

Before changes, the original subclass and benchmark suites passed95 tests
with1 skipped in88.31s. The skipped test requires two CUDA GPUs and NCCL.
Combined final verification: **128 passed,2 skipped in102.70s**. The new
production suite contributes33 passing tests and one two-GPU skip; the original
subclass/benchmark suites contribute95 passes and one two-GPU skip.

Evidence includes exact original FP32 model/AdamW control parity on CPU and
CUDA; diagnostic RNG/buffer/update noninterference; real two-process CPU Gloo
updates with an odd15-row split; full30 runs at C0 and S100 with267 TRAIN rows
(global128,128,11) and129 validation rows (including an empty local tail);
ordinary native selected-checkpoint reload; duplicate/missing/corrupt validation
rejection; full-depth CUDA128/98/11-row updates; strict memory/timing/device
rejection; two-GPU controller checks; exact fourteen-job auto-release and
duplicate/ambiguous/failed-setup rejection; fresh-control reporting and additive
source completeness. CPU full30 tests use a small synthetic network to exercise
collectives/orchestration. Installed-Weaver tests separately exercise the native
model and exported checkpoint path.

All22 required files exist. Python AST, shell scripts and embedded Bash queue
blocks, new documentation links and whitespace checks pass. CLI registry imports
successfully. The original frozen target recipe hash is unchanged, and the
source guard rejects incomplete source and modifications to pinned donor runtime.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -x -p no:cacheprovider tests/test_luka_subclass_ddp_hydra.py tests/test_luka_subclass_hydra.py tests/test_luka_subclass_gpu_benchmark.py
```

Local installed-Weaver testing uses Python3.10/PyTorch2.5.1, the existing
Weaver0.5.3 fixture and one RTX2000 Ada laptop GPU. CPU Gloo collectives test
the distributed orchestration; they do not validate CUDA SyncBatchNorm.

## Production qualification

No new research run, commit, push or Brown submission is performed by this
implementation task. Real two-GTX1080Ti production setup remains pending.
The queue helper submits that setup and its durable dependent release together.
Release requires the complete setup TASK receipt, exact successful Slurm
accounting, all eleven production miniatures, replay/parity, longest-support
memory and recomputed measured acceptance. It cannot reuse the benchmark as
science acceptance. Poor finite accuracy never blocks a registered fit.

The [runbook](LUKA_SUBCLASS_DDP_HYDRA_RUNBOOK.md) contains scoped commit/push,
fresh Brown registration, automatic release and read-only results commands.

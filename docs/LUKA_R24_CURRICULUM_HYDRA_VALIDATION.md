# R24 curriculum validation record

Implementation date:2026-11-13. Local queue readiness and remote scientific
qualification are separate. No remote job, commit or push is performed by the
implementation task.

## Scope and sources

New package `src/hlt_classification/luka_r24_curriculum_hydra/` contains the
twenty-cell registry, separate encoder/decoder schedules, immutable provenance,
FP32 distributed kernel, full training/export, resource qualification, exact
automatic release, worker and saved-results readers. Thin CLI, queue script,
Slurm wrapper, focused tests, active plan, v1 contract and runbook accompany it.

Scientific runtime donor: `58d0c0edf5f41cff2398fb168fc16fc92e169294`.
Benchmark/adaptation donor: `1094dea4ccfb36d08e29f740a97d90fc770335ec`.
Reused source files and changes are listed in `LEGACY_SOURCE_MAP.md`. Original
campaign and benchmark code are untouched. Unrelated dirty workspace files
are preserved.

## Evidence

Before edits:95 passed,4 skipped across the original R24 benchmark, batched
reconstruction and two-GPU subclass execution suites. Skips require local
multi-GPU CUDA/NCCL.

New focused verification covers all20 schedules; independent full-graph
gradient/AdamW comparisons; native initialization and warmup isolation;
actual four-process Gloo uneven-batch gradients; real installed-Weaver FP32
CUDA; miniature native checkpoint export/reload; corruption rejection;
full30 orchestration; exact bootstrap/release/science/report journals;
resource limits and incomplete-profile rejection.

Final evidence: 148 passed, 5 skipped in the combined new/donor regression run;
two subsequently added registration/file-inventory checks also passed. Total
coverage is 150 passing tests and 5 hardware skips (55 new passing tests, 1 new
four-CUDA skip). The full30 test uses four actual Gloo processes with a small
model, preserving uneven global-batch reduction, schedules, publication and
native export. The installed-Weaver CUDA test uses the full native model and
real decoder; it checks initialization and native/AdamW isolation during
decoder preparation. Synthetic inputs do not certify real-data performance.

The additive driver imported all modules and built the full model against a
separately archived58d0c0ed scientific tree. Both shell scripts and the runbook
Bash blocks passed syntax checks. New Python parsed, new Markdown links
resolved and whitespace checks passed. Consumed original scientific,
benchmark and distributed-helper modules have no working-tree changes.

Reproduce the focused plus donor regression checks locally:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_r24_curriculum_hydra.py tests/test_luka_r24_gpu_benchmark.py tests/test_luka_reconstruction_batched.py tests/test_luka_subclass_ddp_hydra.py
```

The user supplied completed Hydra benchmark173643: four GTX1080Ti cards passed
parity and full-support stress, projected12.04h total and48.18 active GPU-hours
for R24, versus10.98h observed A6000 elapsed. These are provided remote benchmark
results, not a local measurement or a completed new curriculum campaign.

Real four-card SyncBatchNorm, all20 production miniatures, full-data replay,
startup/export resource measurements and exact scheduler review remain required
on Hydra. They are implemented in setup and gate automatic release. The local
machine has one CUDA GPU; it cannot certify that remote production stage.

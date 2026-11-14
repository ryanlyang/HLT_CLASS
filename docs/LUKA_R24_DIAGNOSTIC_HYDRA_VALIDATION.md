# R24 diagnostic implementation validation

## Qualification v2 follow-up, 2026-11-13

Hydra job 174442 under 519d672f failed the original independent-trajectory
buffer tolerance. By update 12 the maximum was .001167685 (P0_C) and
.001259416 (P0_R), above 1e-4; all other registered metrics passed. This is
remote evidence of a failed qualification, not a failed scientific result or
proof that buffer drift is harmless. The old log lacks named distributed
buffers and eval-probability comparisons. A6000 jobs 174465–174470 were released
successfully under the old independent lane (user-supplied scheduler output).

The revised protocol saves full serial pre-update model/optimizer states and
tests each of 12 transitions from that exact common state at the original
tolerances. It separately retains the free 12-update trajectories, per-buffer
errors/scales and eval-probability differences. Finite trajectory drift is
observational; nonfinite metrics, integer counter mismatches and unsynchronized
rank buffers still fail. This is an explicit policy/schema change, not a
tolerance increase or a retroactive pass for 174442.

Local baseline before changes: **40 passed** in 105.55s. During implementation,
the existing 40 passed with the revised protocol in 135.95s, and nine new
focused checks passed in 11.99s. New checks cover BN running-stat/counter and
Adam-state restoration without aliasing, named buffer metrics, strict versus
observational thresholds, invalid state rejection, Pascal-only release and
rejection of v1 qualification artifacts. The four-process/Gloo protocol test
checks both 24-step record sets and hashed common-start checkpoints. The Gloo
surrogate has no CUDA SyncBatchNorm and is not hardware parity evidence.

Final combined diagnostic/curriculum/benchmark regression run:
**143 passed, 4 skipped in 241.77s**, including 50 diagnostic tests. Skips
require two/three/four CUDA GPUs and NCCL. The installed-Weaver single-GPU
CUDA test and original/new miniature exports passed locally. This verifies
the new protocol and submission implementation locally, not its acceptance
on Hydra; the genuine four-card check still runs in the new setup.

The runbook includes a fresh Pascal-only retry. Syntax checks pass for 13
Python files, both shell wrappers and all three Bash runbook blocks. The
historical trailing blank line in resources.py is removed. No new Hydra job
was submitted, no A6000 job modified, and no Tigris validation was performed.
Actual 4x1080Ti setup remains required before the eight fits can release.

## Initial implementation evidence

Implementation date: 2026-11-13. This file records local evidence, not a
claim that the new Hydra jobs have run.

Baseline before implementation: curriculum and GPU benchmark suites passed
93 tests; four hardware-only NCCL/multiple-CUDA tests were skipped locally.

The focused diagnostic suite covers the 14-cell matrix, independent random
streams and tensor hashes, actual training/reload and artifact tampering,
two independent automatic-release lanes, exact measured command envelopes,
resource rejection, duplicate submission prevention, and a real four-process
CPU/Gloo update sequence crossing reconstruction activation. An installed
Weaver CUDA test exercises the BF16 native route and original/new fits.

Final local evidence:

- Combined diagnostic/curriculum/benchmark run: **132 passed, 4 skipped** in
  184.12s. The additional complete qualification-protocol/reference-tampering
  test then passed in 33.42s: **133 passed total**, including 40 diagnostic tests.
- Four-process Gloo tests cover twelve sequential uneven-batch updates,
  the complete qualification snapshot/receipt flow, and paired 30-epoch
  training/export/progress on a small surrogate model. They do not substitute
  for CUDA SyncBatchNorm qualification.
- Actual installed-Weaver CUDA fits cover OB_C, OB_R, NB_C, NB_R and NF_R
  miniatures and selected native reload. The new BF16 route exactly matches
  the donor native forward on the same input/model. Test device was an NVIDIA
  RTX 2000 Ada Generation Laptop GPU, not Hydra A6000/Pascal.
- Local environment: Python3.10.19, torch2.5.1/CUDA12.1, Weaver-core0.5.3,
  ParticleTransformer source SHA256
  `44979aea4f53e48246eb1ada35be39d504e44f94b9f0bb69b946a79b4d14ae1d`.
  This is local evidence; production still checks its immutable environment.
- Source audit confirms the reused scientific next-supervision primitives,
  cross-message training and distributed helper are unchanged between pinned
  58d0c0ed and implementation donor a549fef1.
- Python source parsing, both shell wrappers and all Bash runbook blocks pass.
  Focused whitespace checks pass. Historical scientific source files have no
  local diff. Unrelated existing worktree changes were preserved.

Reproduce the focused suite in the installed-Weaver local environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_r24_diagnostic_hydra.py tests/test_luka_r24_curriculum_hydra.py tests/test_luka_r24_gpu_benchmark.py
```

The four skips require two/three/four CUDA GPUs and NCCL. No remote Hydra or
Tigris miniature was run locally. The authorized site for this campaign is
Hydra; its two production setup jobs enforce those missing checks.

Hydra-specific qualification remains a required part of submission: exact
pushed clean source and environment, actual 4x1080 Ti and 1xA6000 allocations,
authenticated full data, real miniatures, full validation and measured time/
memory. No new Hydra jobs or scientific results are produced by local tests.

# R24 diagnostic implementation validation

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

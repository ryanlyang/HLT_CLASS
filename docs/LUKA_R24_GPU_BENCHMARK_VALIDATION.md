# R24 GPU benchmark validation

Focused local suite: **84 passed,4 skipped**,54.41seconds. The skips require
two/three/four CUDA GPUs and NCCL (three new checks, one prior benchmark check).
No Brown measurements or GTX1080Ti/NCCL parity results are claimed yet.

Evidence includes actual2/3/4-process Gloo gradient SUM with unequal shards and
parameters split across bounded buckets; independent full-graph versus chunked
VJP/AdamW comparisons at zero/partial/full reconstruction weights; original
FP32 native step parity; one local CUDA production ParT+decoder step and native
validation; authenticated batch publication/corruption; both-source admission;
exact/idempotent submission, allocation rejection and continuation after failed
profiles. Bash syntax checks passed for both shell scripts; git diff whitespace
check passed. A first combined run exposed test-process solver-thread state
leaking into donor tests; the fixture now restores it and the passing combined
suite includes those donor tests. Remote phases already run in separate processes.

Files belonging to this change:

- `src/hlt_classification/luka_reconstruction_benchmark/` (six files)
- `scripts/luka_r24_gpu_benchmark.py`
- `scripts/queue_luka_r24_gpu_benchmark.sh`
- `sbatch/run_luka_r24_gpu_benchmark.sh`
- `tests/test_luka_r24_gpu_benchmark.py`
- `docs/plans/LUKA_R24_GPU_BENCHMARK_PLAN.md`
- `docs/contracts/LUKA_R24_GPU_BENCHMARK.md`
- `docs/LUKA_R24_GPU_BENCHMARK_RUNBOOK.md`
- this validation file
- relevant additions to README, HANDOFF and LEGACY_SOURCE_MAP (preserve their
  unrelated existing edits).

Baseline:46passed,1multi-GPU/NCCLskip in reconstruction-batched and prior
subclass-benchmark tests. Local command uses the cached installed-Weaver0.5.3
tree at `tmp/r8-followup-weaver` and the `tagging-hlt` Python environment:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH="$PWD/src;$PWD/tmp/r8-followup-weaver"
& 'C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe' -m pytest -q -p no:cacheprovider tests/test_luka_r24_gpu_benchmark.py tests/test_luka_reconstruction_batched.py tests/test_luka_subclass_gpu_benchmark.py
```

Remote acceptance is the requested benchmark itself: authenticated actual
targets at original K, native BatchNorm versus SyncBatchNorm,2/3/4 GPUs,
uneven batches, exact matching/VJP/AdamW and full-support memory. This bounded
Hydra plan explicitly supersedes generic Tigris qualification for this task;
it does not qualify a scientific Pascal campaign.

Next step: commit/push these files, then run the Brown queue block in the
[runbook](LUKA_R24_GPU_BENCHMARK_RUNBOOK.md). No job was submitted locally.

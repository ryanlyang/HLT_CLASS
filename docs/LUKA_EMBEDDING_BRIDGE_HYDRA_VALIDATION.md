# Embedding bridge / response / region validation

Implementation completed locally on 2026-11-10 for queueing real Hydra setup.
No Brown production setup, remote export, commit, push or research submission
was performed here. No result or runtime improvement is claimed.

Scientific authority: [plan](plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md).
The [runbook](LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md) queues real two-GTX1080Ti
qualification and automatic release; [contract](contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md).
The exact internal donor baseline and reused files are in the
[source map](LEGACY_SOURCE_MAP.md): `45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08`.

## Implemented scope

- All 30 scientific fits and 22 native HLT endpoints; R0/P0 alias fresh C0.
- Three assistant ladders in unchanged U100 axes; shared six-condition TR;
  independently encoded physical-crop TP; all nine R and nine P objectives.
- Original assignments and native frontend; authenticated physical export,
  quarter-view preparation, unchanged TRAIN U100 arrays, evaluation-only
  targets and frozen TRAIN statistics in typed memory-mapped stores.
- Two-rank FP32/SyncBatchNorm execution, true parent dependencies, genuine
  production setup/startup, resource admission and automatic release/report.
- Epoch losses, throughput, memory, target coverage, fixed-panel diagnostics,
  checkpoint selection/replay, stripped native export and read-only contrasts.

## Local evidence

Python: `C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`.
Tests used the repository's existing installed-Weaver fixture at
`tmp/hydra-weaver-fixture-20261102`, one thread per numerical library,
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and CUDA deterministic
workspace configuration `:4096:8`.

1. Baseline donor suite before implementation: **106 passed, 1 skipped**.
2. Combined campaign, two-GPU subclass donor and selective-teacher regression
   suite: **151 passed, 2 skipped in 111.99s**. Both skips require two CUDA
   GPUs/NCCL. This also exercised the donor's real two-process CPU Gloo checks.
3. After final reporting, CPU admission and epoch-observation additions, the
   complete new campaign suite: **48 passed, 1 skipped in 38.18s**. The skip is
   the genuine two-CUDA/NCCL producer pipeline and all-family SyncBN test.
4. After adding measured producer-chain cost reporting: **6 focused tests
   passed in 8.19s**, covering chain accounting, historical references,
   CPU measurements, automatic release and failed-setup handling.

Static checks passed for all 28 campaign files, Python syntax, both shell
scripts, all three Bash runbook blocks, 43 local documentation links and all
58 recorded donor paths at the exact baseline. New files have one final
newline and no trailing whitespace. The CLI help loads successfully.

Reproduction of the final campaign suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONNOUSERSITE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -p no:cacheprovider tests/test_luka_embedding_bridge_hydra.py
```

Meaningful campaign checks include:

- Exact original U100 AdamW loss, native state and optimizer-update parity;
  auxiliary heads absent from native reload; every loss reaches the encoder.
- Actual two-epoch native fits for all eight producers, target publication,
  reload and consumption by all final families; full30 C0/R5/P5 fixture fits.
- All original coordinate replays and new quarter views through the real
  physical frontend; changed original assignments rejected despite a newly
  valid outer shard hash.
- Response algebra and scale floors; crop nesting/permutation/isolation;
  valid-region/scale/jet reductions; source-bound targets and controls.
- Technical-bank rejection by scientific readers, corruption/role/hash
  checks, checkpoint selection, exact DAG and duplicate/ambiguous submission
  refusal; CPU/GPU/time/coverage failures preserve diagnostics.
- Full-depth eight-particle/two-class-block **one-GPU** FP32 forward/backward
  for C0, R8, P7, TR and TP with global batch128 and tails98/11, including
  observation gradients. This local card is an RTX2000 Ada laptop GPU,
  not a GTX1080Ti and not a two-rank acceptance run.
- CPU allocation rounding for release/report; measured CPU report kernels
  and admission arithmetic; optional S2IT/S6 ownership/corruption checks;
  missing fresh C0 never replaced by historical BF16 C0; producer-chain
  costs include all assistants and distinguish missing cost evidence.

Synthetic training fixtures reduce depth for fast integration checks; the
separate CUDA cases retain production depth. Neither uses research data.

## Required remote evidence

The runbook pins clean pushed source, exports original physical rows and
U100 weights on SPORC, verifies transfer bytes and available quota on Brown,
then submits setup and durable release. Setup must establish:

- Original view/target/checkpoint reproduction and installed-Weaver parity.
- Genuine two-GTX1080Ti NCCL/SyncBatchNorm loss, gradient, buffer and update
  parity; diagnostic RNG/state neutrality on every scientific family.
- All 30 production miniatures with actual technical parent-bank exports,
  every teacher condition/source/scale, largest support and tail batches.
- Warmed update timings, full validation and bounded serialized export costs,
  summed host/process peaks and per-card peaks under the registered bounds.
- CPU authentication/report measurement and explicit full-campaign estimate.

Each scientific fit additionally authenticates its actual scientific parent
and runs a disposable startup before fresh initialization. Technical weights
and banks cannot be scientific parents. A weak but finite teacher is allowed
to complete and release its dependent students.

Queue readiness is readiness to perform this remote qualification. It does
not mean that its resource estimates, two-rank parity or study outcomes have
already passed on Brown. The next task is the scoped commit/push and runbook;
the implementation changed no existing jobs or donor scientific runtime.

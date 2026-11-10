# Shared summary-token Hydra validation

Local implementation evidence, 2026-11-09. Ready to queue real Hydra setup;
this document does not claim remote execution or a scientific recovery gain.

## Environment and commands

Python `C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe` (3.10.19),
Torch2.5.1, Weaver0.5.3 source fixture at
`tmp/hydra-weaver-fixture-20261102`, NVIDIA RTX2000 Ada Generation Laptop GPU.
No bytecode writes. The fixture is a local test aid, not a Brown override.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='tmp/hydra-weaver-fixture-20261102;src'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -B -s -m pytest -q -p no:cacheprovider tests/test_luka_summary_tokens_hydra.py
```

Complete suite result: **48 passed in58.03s**, no skips. Two additional
production-depth FP32 qualification tests (both seeds) then passed in10.03s;
**50 distinct campaign tests passed**. Donor suite before changes:
`tests/test_luka_capacity_hydra.py`, **62 passed in84.77s**. These are112
distinct passing tests across three invocations. New campaign CLI registry,
Python compilation and queue/worker Bash syntax are checked separately.

The initial batched-query check exposed FP32 GEMM shape-dependent differences
around1.5e-6. Registration explicitly uses rtol2e-5/atol5e-6 and records
maximum differences; mathematical value/gradient tests independently verify
the calculation. The native BF16 historical AUC tolerance remains5e-5.
No accuracy threshold was relaxed. An early combined local test process
terminated without a traceback; isolated CUDA checks passed and the complete
thread-limited final command above completed successfully.

## Verified behavior

- Fixed A-E x two matched seeds, ten independent science tasks, setup/release
  bootstrap and afterany report (13 jobs). Dry review binds all commands;
  measured walltimes may shrink but resource/task changes are rejected.
- Ordinary native CE and native U100+IT reproduce donor initial weights,
  losses and one optimizer update exactly. Extra query/head creation preserves
  the native RNG stream. C/D/E share initial native states; D/E share every
  initial parameter and differ only in predictor routing.
- Actual installed full-depth deployed parameter counts: one summary
  **2,143,615**, three summaries **2,146,687**, difference **3072** (~0.143%).
  No cloned pooling weights. An independent batched-query reference checks
  outputs and encoder/pooling/query gradients. Changing one extra query cannot
  affect the other summaries. Particle encoder executes once; padding is inert.
- Auxiliary gradients reach only the declared pooled summary and shared
  upstream weights, not classifier weights or frozen targets. CE reaches the
  expanded classifier; zero initial extra-query CE gradients are allowed.
  A/C never load target banks. Wrong target roles, identity joins, corrupt
  payloads, forbidden targets and wrong architecture reload fail closed.
- Every cell runs a complete30-epoch CPU fit with reduced block counts but
  registered width128, followed by selected checkpoint, immutable epoch,
  metric and deployment reload checks. All five configurations additionally
  run full eight-particle/two-class-block CUDA backward at batch128 and partial11,
  plus native export/reload parity. CUDA miniatures exercise CE/shared/separate
  routes with production fit functions.
- Both seeds additionally pass production-depth, batch128 CUDA FP32 factory
  and independently batched pooling equivalence with registered tolerances.
- Complete CUDA setup worker across all ten cells: replay, real model
  forward/backward, warmed timing, partial/longest inputs, miniatures, resource
  artifacts and recomputed acceptance. This uses synthetic data/targets and
  reduced depth; Slurm, A6000 identity and Linux host-memory envelope are
  simulated. It is not a remote resource measurement.
- Original IT export/normalization and bank validators, TRAIN-only joins,
  original-worktree subprocess authentication, immutable registration/source
  guards and substituted-target rejection.
- Simulated exact-ID bootstrap and automatic science release, completed
  submission idempotence, failed setup and ambiguous partial-claim rejection;
  observed Hydra one-requested/two-allocated CPU rounding retained.
- Same-seed contrasts with missing controls left null, two-seed descriptive
  summaries, completed/provisional reports and saved-prediction validation.
  Read-only commands leave all fixture bytes unchanged. Poor finite quality
  does not gate release. Resource ceilings remain enforced.

## Remote boundary and next execution

No commit, push, remote inference or job submission was performed here. Use
the [runbook](LUKA_SUMMARY_TOKENS_HYDRA_RUNBOOK.md) to commit/push only this
campaign and register a fresh output root against the completed selective IT
parent. No new teacher fit or SPORC transfer is required.

The real setup must authenticate pinned installed Weaver, original HLT/U100/IT
data and targets, replay native BF16 validation, verify full production pooling
and factory parity, run actual-target miniatures and measure every A6000
profile within the registered envelope. Only successful setup and accounting
allow the durable CPU release to queue science. All scientific conclusions,
production latency assessment and recovery gains remain unmeasured.

The active [plan](plans/LUKA_SUMMARY_TOKENS_HYDRA_PLAN.md) selects Hydra
qualification for this campaign; local evidence does not claim Tigris or
Brown qualification. See the [contract](contracts/LUKA_SUMMARY_TOKENS_HYDRA.md)
and [donor map](LEGACY_SOURCE_MAP.md) for schema and source lineage.

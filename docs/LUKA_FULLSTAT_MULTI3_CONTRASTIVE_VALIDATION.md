# FullStat Multi3 contrastive validation

Local development evidence, 2026-11-04. No Oscar jobs were submitted and no
research data, checkpoints or final-test data were opened by this implementation.

Environment: Python 3.10, PyTorch 2.5.1, installed Weaver 0.5.3 in an isolated
temporary target directory, CPU and local CUDA (RTX 2000 Ada laptop GPU).
Tests disabled bytecode writes and used CUBLAS_WORKSPACE_CONFIG=:4096:8;
the final focused runs also explicitly set PYTHONNOUSERSITE=1. An initial sandbox run
could not read the isolated Weaver installation; tests were rerun with access.
Two baseline CUDA cases initially lacked the required CUBLAS setting; both
passed after setting it. These were test-environment failures, not source fixes.

## Test results

- Combined new campaign, donor contrastive and export-recovery regression:
  **95 passed in 189.80s**, before adding CPU process parallelism. Two existing
  Weaver fusion-mask deprecation warnings were emitted.
- Final campaign: **37 passing cases**. The final full run had 36 passes and
  one negative-test fixture that no longer exceeded the increased 48-hour
  allocation. Its actual installed-Weaver startup had passed. After increasing
  that deliberately oversized fixture, the focused rerun passed in 33.86s;
  no production code changed for this correction.
- Python AST, forbidden-import and Bash syntax checks passed. Runbook Bash
  blocks, local document links and whitespace checks passed.
- Queue compatibility follow-up after the user's Hydra commit `73abecfd`:
  **11 source/registration cases passed in 6.31s**. The execution baseline was
  advanced to that exact commit after verifying that its runtime delta contains
  only Hydra additions; no scientific donor file changed. New tests retain
  rejection of later changes outside this campaign. Scientific baseline is
  still `a40ae47`; this does not alter losses, weights, model or sampling.

Focused reproduction command, with the environment above:

```powershell
python -m pytest -q -p no:cacheprovider tests/test_luka_fullstat_multi3_contrastive.py
```

## Scope

- Reference versus optimized full 1025-candidate mix sampling on CPU/CUDA:
  exact indices and evidence, canonical ties, repeated anchors, remainder
  batches, known event/alias exclusions, three persistent CPU processes and
  bounded asynchronous lookahead. Separate diagnostic RNG streams remain exact.
- Identical original Multi3 parameters and RNG after matching-head creation;
  zero-weight objective/gradients reproduce the historical kernel exactly.
  Nonzero matching gradients reach the encoder and independent head; frozen
  keys have no gradients and the native classifier gets CE alone.
- Native deployment excludes every auxiliary parameter and remains unchanged
  when discarded predictor parameters are deliberately filled with NaNs.
- Complete 20-epoch synthetic CPU/CUDA training and workers, every epoch state,
  full/native selected checkpoint publication, prediction replay and tampering
  rejection. A resumed fit reproduces uninterrupted metrics, losses and selected
  weights exactly; a simulated post-fit publication failure recovers without
  any further optimizer update.
- Full installed-Weaver architecture on local CUDA through the actual integrated
  startup path: installed parity, full-candidate stress, native reload, 64
  complete measured updates and validation timing. Synthetic 6006-row bank;
  this is not a 3M-row Oscar resource acceptance. Child peak RSS is measured
  through Linux resource accounting in production; the Windows test cannot
  certify that host envelope.
- All four actual frozen-bank validators, exact ordered joins, pure-OFFLINE key
  transform, corrupt payload rejection and wrong-role refusal.
- Exact single-job dry/live journals, authorization, scheduler rejection,
  ambiguous submission, duplicate launch prevention and recovery claims.
- Exact Oscar allocation and historical-source checksum checks; historical
  selected-epoch metadata is not mistaken for a file reference.
- Real Bash detached-launch smoke test: the foreground tail can exit before
  the background task completes. Windows uses a setsid stub; Linux SSH-session
  survival is provided by the production nohup/setsid/stdin-redirection path,
  not independently demonstrated by this Windows test.

## Performance evidence

A local full-batch sampler probe used 110,000 synthetic 128-dimensional keys,
11 classes, 128 queries, 4096 hard proposals and 1024 negatives. Eight repeated
samples after warmup measured approximately 0.4410 s/batch for the reference and
0.1395 s/batch for the optimized sampler: **3.16x sampling speedup**. Candidate
indices and evidence agreed exactly. The probe uses seed 12 for synthetic keys
and query selection, epochs 3 through 10 for timing, and three spawned CPU
workers. This microbenchmark excludes the ParT training update and uses a
smaller bank than production; it is not a promised end-to-end or Oscar speedup.

Separate full-model synthetic startup checks measured 0.3413 s/update with
serial candidate construction and 0.1389 s/update with the final parallel
path. These short local checks use synthetic token supports and 6006 bank rows,
so they are illustrative, not a production ETA. The allocation ceiling is
48 hours, with actual complete-update timing enforced inside the science job.

The original eight-query FP32 dot reductions are unchanged. Improvements are
batched index uploads, stable GPU ranking, one hard-index readback, vectorized
exclusion partitioning, three persistent CPU workers, bounded candidate
lookahead, and batched loss logging.
There is no reduced batch/support/negative count or lower-precision target bank.

## Remote boundary

Actual source replay against the user's completed Oscar artifacts, queue dry
review, full 3M-bank/longest-support memory and full-update timing still run on
Oscar. The integrated worker persists measurements before accepting the fit.
Local synthetic tests cannot certify remote filesystem health, remaining quota,
scheduler availability or the new objective's generalization performance.

Source, contract and donor inventory are in the [runbook](LUKA_FULLSTAT_MULTI3_CONTRASTIVE_RUNBOOK.md),
[contract](contracts/LUKA_FULLSTAT_MULTI3_CONTRASTIVE.md) and
[source map](LEGACY_SOURCE_MAP.md).

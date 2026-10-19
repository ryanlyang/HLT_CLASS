# FullStat last-rung attention: science-identical execution recovery

Authority: operator request2026-10-19 to accelerate the existing attention fit
without weakening its training, then replace the slow job. This operational plan
supplements, and does not change, `LUKA_FULLSTAT_LAST_ATTENTION_PLAN.md`.

## Fixed science

Reuse the authenticated exported D033 TRAIN bank from attention source
`62c9b2df91eb3bda4fd4df4edc75c5f810a48c55`. Teacher remains the selected original
COARSE_D033 epoch17. Reuse completed M0HLT/OFFLINE/DIRECT_D000/COARSE_D000
references from the original FullStat parent. Train **only ATTN_D000**, fresh
from its original seed; do not export targets again or repeat LOGIT_D000.

Unchanged: exact3M TRAIN/500k validation; paired HLT slots; batch128;
per-epoch global permutation;20epochs; AdamW and warmup/cosine schedule;
BF16 forward/FP32 losses; CE/logit-KD weights/temperature;0.1 attentionKL;
block7, head-mean, pre-dropout, FP32 attention readout and equal-jet/query
normalization; every-epoch full-validation selection. Native single-HLT
deployment is unchanged. Final-test1.5M and105,325 unused rows stay sealed.

No token truncation, length sorting, reduced target precision, dropped heads,
fewer auxiliary batches, altered teacher, changed loss coefficient, or changed
validation cadence is permitted by this optimization.

## Operational changes

- Authenticate every old bank payload/identity/surface as before. Allocate one
  FP32 ragged target buffer plus offsets under an explicit RAM admission limit.
  Hash the exact shard bytes copied. No target files are opened during batches.
- One bounded CPU producer prepares two batches ahead in the **same** sampled
  order. It performs no Torch/CUDA or random operations. Original masks/padding
  and batch membership are retained. Producer failures propagate.
- Keep original forward/loss/backward/finite-gradient guards and AdamW math.
  Compute diagnostics without an extra autograd graph; transfer their six
  scalar values to the host together. Do not approximate attention math.
- Emit progress every128updates, including input wait and epoch-training ETA.
- Commit immutable epoch checkpoints with current weights/buffers, optimizer,
  RNG states, updates, full history and selected-best state. Publish their
  checksum receipt last. Only complete, same-execution epochs can resume.

The old running implementation has no epoch checkpoint: a new fast fit starts
at epoch1, not from its live epoch. No live job/source/spec is modified.

## Acceptance and execution

New `LUKA_FULLSTAT_FAST_ATTENTION_*/v1` execution identities. Existing source
and scientific artifacts are frozen byte-for-byte. Additive donor is
`cfb3a1587a6739b1949be9e65ac80f7ba9f8ead9`; original bank/source replay runs in
its original pinned62c9 worktree. Unrelated embedding-aux work stays untouched.

Oscar L40S,6CPUs,180000MiB, same installed environment/backend. New GPU gate:

1. Load authentic full HLT TRAIN/validation caches and full resident TRAIN bank;
   fail before target allocation if buffers/reserve exceed the admitted envelope.
2. Installed native/readout parity in FP32 and BF16, original/fast miniature
   training-state/loss/validation equality, epoch checkpoint roundtrip, full
   population maximum TRAIN backward and validation forward support stress.
3. Compare original LRU and resident target bytes for32globally shuffled batches.
   Measure their loaders separately (not a claim of end-to-end speedup).
4. Warm up8updates, measure256real-bank globally shuffled optimizer updates in
   eight32-update blocks. Use the slowest block's mean, not the fastest batch.
5. Time a complete500kvalidation pass including metric computation. Derive a
   two-times conservative20epoch walltime plus publication/IO allowance, with
   a12h floor and48h ceiling. Fail resource/timing admission, never model quality.

After successful registered gate accounting and dry review, submit only the
attention fit and its dependent CPU comparison report. No automatic follow-up,
cancellation, restart, deletion, tuning or test inference. The report contrasts
ATTN_D000 with **original COARSE_D000** and labels historical controls as reused.

A finite sample gives an estimate, not a guaranteed runtime. Near-logit-KD
speed is a target to measure, not a pre-acceptance claim. Attention's extra
readout/backpropagation remains intrinsically additional work.

## Validation and handoff

Local CPU/CUDA miniature parity and resume equality are required, alongside
reader corruption/budget/order tests and production-path tests with synthetic
populations. These do not replace a real Oscar gate. See the
[runbook](../LUKA_FULLSTAT_FAST_ATTENTION_RUNBOOK.md).

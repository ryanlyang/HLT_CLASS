# FullStat last-rung aligned-attention comparison

Authorized 2026-10-19. This is a new Oscar experiment; it does not alter the
completed FullStat campaign or the trial ALLJETS/fixed30 attention studies.
Import the authenticated selected regular COARSE_D033 (reported epoch 17) from
FullStat recovery source 9bc3cae0. All parent fits, dependencies, source and
payloads must authenticate in the original pinned worktree. No warm start.

Train exactly two fresh single-HLT D000 ParTs from the same coordinate seeds:
LOGIT_D000 with C25/P75 T2, and ATTN_D000 with the same objective plus constant
0.1 attention KL. Freeze the original FullStat20 recipe: 3M TRAIN/500k validation,
true batch 128, exactly 20 complete epochs, warmup2/cosine, native optimizer groups,
BF16 forward/FP32 losses, same selector. The 1.5M test and 105325 unused rows stay
sealed; event independence and old-trial overlap remain unverified. The earlier
trial's repurposed-test flags, 30 epochs and checkpoint-retirement policy do NOT
apply. Full validation chooses checkpoints; no quality-based gate or extension.

Reuse the implemented block 7 head-mean pre-dropout probability surface and
teacher-to-student attention KL (epsilon 1e-8, real keys/queries, equal jet weight,
native diagonal and own-view pair bias). Teacher is frozen/eval. No target enters
student forward. D033/D000 persistent-slot order is proven by unchanged source,
authenticated cache bytes and ordered identities/labels/counts, not by counts
alone or truth-level particle matching. No independent sorting/trimming.

Export one shared TRAIN-only, ragged FP32 attention+T2 probability bank from
the selected D033. Use a separately declared fresh scratch bank directory;
admit personal quota against 4*sum(n^2), probability/population overhead and
reserve. Bounded shards/mmap reuse prevent all attention from entering RAM.
Keep both selected checkpoints, all histories and validation probabilities
durably. No automatic deletion, cleanup, test evaluation or campaign followup.

One new Oscar L40S GPU preflight checks installed native FP32/BF16 parity,
lambda-zero training-kernel parity, both production fits on common representative
miniatures, teacher/bank and native checkpoint readback, separate maximum TRAIN
backward/teacher-export and validation inference support. Scan all required
prepared payloads and measure full-bank storage support. Project 20-epoch runtime
with 2x margin, include original-source replay and bank I/O, reserve full-cache
host memory, require usage below 85% host/90% GPU and 48h job ceilings. Measurements
are estimates, not full 3M training timings. Failed evidence blocks science.

Science DAG: export_D033 -> train_LOGIT_D000 and train_ATTN_D000 -> aggregate.
Gate and science each require exact clean pushed source, saved complete dry
review, explicit hash authorization and durable submission journal/claim.
Science additionally requires the real successful exact gate job/evidence.
No automatic retry after ambiguous sbatch. All outputs have a new namespace.
The primary contrast is ATTN_D000 minus matched LOGIT_D000; historical M0HLT,
OFFLINE, DIRECT_D000, COARSE_D000 and D033 are authenticated references only.
Local tests do not substitute for real Oscar acceptance.

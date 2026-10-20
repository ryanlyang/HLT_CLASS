# FullStat final-rung fixed-OFFLINE embedding prediction

Authorized 2026-10-20: adapt the embedding-ladder objective to ONLY the final
existing COARSE_D033 -> fresh EMBED_D000 rung on Oscar. This plan, not the
trial ALLJETS/SPORC/fixed30 plan, governs the new namespace.

Freeze the completed FullStat recovery source 9bc3cae0, selected D033 epoch 17,
pure OFFLINE selected checkpoint, native D000 inputs, coordinate seeds and
original compact D033 TRAIN T=2 bank. Parent completion and artifacts must
authenticate in their original worktree. Do not depend on the running
attention experiment or regenerate its bank.

Exactly 3M TRAIN / 500k validation, true batch 128, 20 complete epochs, warmup 2/cosine,
AdamW/native parameter groups, BF16 model/FP32 losses, original full-validation
AUC-first checkpoint selection. The 1.5M test and 105325 unused rows remain
sealed. Event independence and old-trial overlap remain unverified.

Loss is 0.25 CE + 0.75 T^2 KL(q_D033||p_D000) + 0.3 SmoothL1(g(h_D000),z_OFFLINE).
T=2; SmoothL1 beta 1 and mean over jets/dimensions. Reuse the existing native
pre-classifier 128-vector tap, 128-GELU-128 predictor and isolated initialization.
Targets come from the frozen PURE OFFLINE classifier, NOT D033 or U000.
Export TRAIN ONLY in FP32/eval; per-dimension TRAIN population moments are
accumulated in FP64, scale floored at 1e-3, standardized targets stored FP32.
No validation embeddings, attention loss, fusion, particle reconstruction,
warm start, alternate head input, loss tuning or inference-time predictor.

Primary contrast: new EMBED_D000 minus original COARSE_D000. The completed
logit baseline is a historical reference with matched seeds/recipe; a native
zero-aux full-epoch miniature equality check is mandatory. Do not claim a new
matched logit fit or multi-seed significance. Include M0HLT/OFFLINE/DIRECT/D033.

New source-pinned Oscar L40S, 6 CPU / 180000 MiB gate (8h ceiling) authenticates
all required TRAIN OFFLINE and ordinary-role D000 cache bytes, exports one
compact bank, validates exact ordered jet joins, installed native FP32 parity,
FP32/BF16 head and gradient witnesses, zero-aux native KD equality, longest
TRAIN backward/validation forward and OFFLINE FP32 export stress. Run one
complete 3M/500k technical epoch and selected native checkpoint/prediction
readback. This measures full-epoch time, not a short-batch extrapolation.
Admit a 2x conservative 20-epoch request (12h floor, 48h ceiling), below 85% host
and 90% GPU memory. Bound each epoch's checkpoint overhead by the measured
probe's checkpoint/report/readback phase. Scientific quality never gates admission.

Science is two jobs: train_EMBED_D000 -> CPU aggregate. Reuse the gate's
authenticated compact embedding bank and original D033 probabilities in RAM.
No repeated teacher export, prior-rung fit, logit fit or automatic follow-up.
Keep progress, every epoch history, immutable epoch weights/optimizer/RNG
checkpoints and selected native outputs. Reserve 12 GiB personally available
durable space. No automatic deletion or resumption; failed executions remain.
An explicit restart tool is not part of this registration: retained epochs
allow a separately reviewed recovery without promising an automatic resume.

Implementation donor 5ed50964085e9bf364030764bbbc755179c445ae. Existing runtime
is byte-frozen; new package/entrypoints only. Exact pushed source, dry-plan
hash authorization and successful journal-bound real Oscar gate precede
science. Local installed-Weaver miniatures are not remote acceptance.

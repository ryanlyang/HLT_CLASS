# FullStat Multi3 with pure-OFFLINE instance identification

Active scientific plan, version 1. This is one fresh Oscar student and a
historical comparison to the completed Multi3; neither teacher fitting nor a
fresh control is registered. The existing Multi3 teachers are **U100, CONCAT,
FUSION**, not OFFLINE, CONCAT, FUSION. Pure OFFLINE is a fourth, training-only
source for the new matching task.

Use the authenticated 3,000,000 TRAIN and 500,000 development-validation rows,
native D000 HLT inputs, original Multi3 initialization/order seeds, batch 128,
20 epochs, and the existing FullStat optimizer and two-epoch warmup/cosine
schedule. Select by the existing full-validation selection key. Final test
stays sealed. Poor finite metrics never reject a fit.

## Objective and graph

`CE(native logits, labels) + (10/3) * sum(SmoothL1(predictor_i(h), target_i))
+ lambda(update) * CE(cosine(matching_head(h), OFFLINE_candidates)/0.1, 0)`.

The three existing predictors target TRAIN-standardized native 128-dimensional
U100, CONCAT and FUSION representations. Their architecture, initialization,
normalization and loss reduction remain unchanged. The independent matching
head is Linear(128,128), GELU, Linear(128,128), seeded 2026103101 inside an isolated
CPU RNG context. Its output and keys are L2-normalized with epsilon 1e-12 in
FP32. Lambda is `0.3 * min(1, update/(3*ceil(N_train/128)))` for one-based updates.
The main classification head consumes only native h. Auxiliary losses update
the shared encoder, but do not directly update the native classifier. No logit
KD, retained predictor, relational loss, or additional student view is used.
Remove every auxiliary head for deployment; reload into native HLT ParT.

## Matching population and sampler

Reuse the original frozen pure-OFFLINE teacher's authenticated TRAIN bank:
FP32 export, FP64 TRAIN population standardization, standard-deviation floor
0.001, then FP32 L2 normalization. No new teacher inference inside training.
Each query identifies its own canonical jet among one paired positive and
1,024 distinct TRAIN negatives. Use the versioned `luka_u100_contrastive`
sampler's `mix` law exactly: H=256, S=256, G=512; H proposal 4096 balanced
different-class rows, ranks 257..1024, canonical-identity tie breaks. Assemble
H then S then G, excluding the positive and prior components. Exclude known
authenticated event/alias groups when available; absent grouping is reported,
not represented as proven event independence. Root seed 2026103102, replicate0,
identity/epoch/component/stream keyed PCG64. Never mine validation or test keys.

## Performance and recovery

Retain the reference eight-query FP32 reduction kernels for mining. Prepare all
proposals before mining, upload their indices once, and implement lexicographic
ranking with stable GPU sorts (canonical order, then descending score). Transfer
only selected hard indices to CPU once per batch. Use bounded one-batch candidate
lookahead on a dedicated CUDA stream that reads only frozen keys and waits for
their publication; no asynchronous access to mutable training state. Require exact
candidate/evidence parity against the reference, including ties and exclusions.
Use three persistent spawned CPU workers for independent per-jet proposal and
post-mining construction, with ordered joins and the same identity-keyed RNG.
Only immutable CPU indices/identities cross process boundaries; CUDA keys and
all trainable state stay in the parent. Include child peak RSS in host admission.
Keep frozen keys resident on GPU, batch diagnostic scalar readback, and avoid
repeated inference, decompression, full-bank scans, or exhaustive per-update
metadata hashing. Do not alter candidate counts, precision, token support,
batch size, optimizer, or sampler law for speed. Benchmark the complete update
including sampling, loss, backward and optimizer; do not claim Oscar speed from
a local microbenchmark.

One 48-hour science allocation performs integrated installed-Weaver parity, sampler
parity, full-candidate longest-support stress, timing and memory admission;
these probes are discarded before fresh scientific initialization. There is no
separate preflight job. Persist failed admission evidence. Use exact pushed
source, original-source replay of completed parents, immutable references,
atomic outputs, exact Slurm IDs, dry review, and bounded detached submission.
Save full model/optimizer/RNG and validated history each completed epoch. An
explicit recovery must continue the same registered schedule; a completed fit
can be finalized without repeating optimization. Never silently retry or reuse
partial artifacts merely because paths exist.

This user-authorized Oscar plan supersedes legacy Tigris-only execution text
for this campaign. Local parity is not Oscar acceptance; the actual allocation
must supply its own measured admission evidence before the fit.

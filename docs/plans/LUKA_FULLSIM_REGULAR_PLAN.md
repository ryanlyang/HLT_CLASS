# Luka regular-teacher DIRECT / COARSE, 100k / 50k

Authorized for implementation on 2026-10-13. This is a new campaign, not a
recovery or relabelling of the deleted fusion experiment. For this campaign
only, this plan supersedes the fusion-only graph, RAM-only preprocessing and
historical-worktree admission in `LUKA_FULLSIM_SCIENCE_PLAN.md`. The original
foundation, physical conventions, view kernels, feature kernels and training
recipe remain authoritative. No old artifact or old contract is rewritten.

Use the restored 890-file Luka snapshot, strict **offline** pT > 200 GeV,
matched nonempty pairs, QCD only from QCD files, natural eleven-class ratios,
and the original deterministic file-disjoint 100000 / 50000 selected rows.
Require the original inventory and split hashes; do not resplit after seeing
results. Both tracking sides are provisionally mm by operator decision, not
producer confirmation. Zero error masks only the error, not its finite signed
value. Neutral tracking is unavailable. Retain tails, support 512 and no
truncation. Identity/mapping metadata are never model features.

## Fixed graph

Nine fresh single-encoder fits: M0HLT (D000 CE), OFFLINE (native offline CE),
U000 (CE), DIRECT_D000 from U000, and
U000 -> COARSE_U050 -> COARSE_U100 -> COARSE_D066 -> COARSE_D033 -> COARSE_D000.
Five TRAIN-only T=2 probability reducers, aggregate and complete make 16 science
jobs. No fusion, dense branch, warm starts, imported weights or technical-probe
teachers. OFFLINE and intermediate offline-containing views are oracle controls;
M0HLT, DIRECT_D000 and COARSE_D000 deploy on identical native HLT inputs.

Reuse unchanged `jetclass2_delphes.campaign.recipe()` / `runner.train_kernel`:
batch 256, BF16 forward/FP32 loss, AdamW, 60--100 passes, patience 15 from first
validation, warmup 3/hold 45/decay 60/floor, KD 25% CE + 75% forward KL at T=2
with one T-squared factor. Preserve coordinate-paired initialization/sampler
seeds. All 50k validation jets select checkpoints and supply development metrics.
Poor metrics never fail a task or skip a row. Single seed; ladder uses more
training compute than direct. Recovery references M0HLT and pure OFFLINE.

## Faster preparation and new admission

Rebuild foundation from restored raw bytes; no deleted worktree, preflight,
assignment or checkpoint is required. One CPU stage audits and prepares each
ordinary selected row once. Initially eight spawn workers process separate
source files, one outstanding task per worker; publish and combine in canonical
membership order. Only selected contiguous entries are decoded. Store ragged
float32 17-feature/4-vector blocks, int64 offsets/labels and uint8 identities for
all seven coordinates, plus original assignment arrays. No disk padding to 512.
Batch padding/masks still use the unchanged RAM cache implementation.

Persist completion-last preparation and cache manifests. Verify exact row joins,
array shapes/dtypes, finite values, capacity, file checksums and original-style
per-coordinate input fingerprints on reuse. GPU workers load only their one
coordinate; reducers load TRAIN only. Never fall back to rebuilding ROOT views
inside a scientific GPU job. Cache changes affect storage/execution, not physics.
Compare against the original serial preparation in local fixtures and a real
TRAIN-file replay inside the CPU gate. Initial audit merges file-local moments
in fixed order (roundoff may differ); input bytes must agree exactly.

Three gate jobs: foundation, CPU durable preparation, genuine SPORC A100 GPU
preflight. Gate GPU envelope initially 6 CPUs/90000 MiB/8h; preparation 8 CPUs/
64000 MiB/8h. These are gate bounds, not measured speed claims. The fresh
preflight authenticates installed-Weaver parity, full-support batch 256 stress,
all seven cached coordinates and full-population CE/KD passes/checkpoint/bank
round trips. Science requests come from those measurements and must fit debug's
24h ceiling and host/GPU headroom. Probe metrics never select a method.
SPORC-specific real acceptance is required here; no Tigris claim is made.

Clean exact pushed source, immutable plans, full dry review, explicit plan-hash
authorization, exclusive live claims and per-job journals are required for both
gate and science. Failed/ambiguous directories are preserved, never retried
silently. No live submission is part of local implementation. No automatic
science follow-up unless separately authorized.

## Validation-length preflight correction and recovery

The original a54c60e5 gate completed foundation (21880174) and preparation
(21880175), then rejected a validation maximum longer than TRAIN (21880176).
The user authorized fixing this operational assumption and reusing the completed
inputs. This amendment supersedes the v1 cross-role length comparison, not any
scientific data/model/training rule.

Preflight v2 stress-tests each coordinate's longest TRAIN input with the existing
batch-256, three-step backward test. Separately, it repeats that coordinate's
longest validation input at batch 256 for three inference-only forward passes.
No validation labels, losses, gradients, optimizer or scientific metrics enter
the validation stress. Record maxima separately by role, and fail on resource
headroom/nonfinite failures. Keep full-population technical passes and installed
parity. Do not truncate particles, pad with fake active particles, increase
memory silently, or require validation lengths to be bounded by TRAIN lengths.

Recovery creates a fresh sibling gate root and queues only one replacement
preflight. Authenticate the original clean a54c60e5 source, exact gate journal,
successful foundation/preparation accounting, FAILED/1:0 original preflight,
failure log, and original prepared/cache parents. Preserve all original files.
Every existing src module must remain byte-identical except the regular campaign,
preflight and submission orchestration; only regular_recovery.py may be added.
Also compare the campaign's scientific declarations (population, conventions,
nodes, tasks, registration and gate policy) as syntax trees. No changed physical
reader/cache, scientific worker, model, matching, seeds, recipe or split is
admitted. Preparation and execution source parents remain distinct.

New science under the recovery root must bind the two original successful CPU
jobs and the new successful preflight job, with the old prepared-root path.
The old failed preflight never becomes success evidence. Full dry review and
explicit execution remain mandatory separately for recovery and science. A
second failure remains inspectable and is not silently retried.

## Test status

The old 149553-jet final holdout was already evaluated in the previous fusion
campaign. It is **not a fresh blind test**. This new campaign has no final-test
particle/evaluation capability; `final_test_accessed=false` refers to this run,
not to the entire research history. New conclusions use development validation
until a separately registered evaluation is explicitly authorized.

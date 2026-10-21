# FullStat standalone CE + OFFLINE embedding sweep, Oscar

Authorized 2026-10-20: seven fresh HLT-only models, lambda **0, 2, 10, 50,
100, 500, 1000**, on the existing FullStat **3M TRAIN / 500k validation**.
Lambda 0 is the matched CE-only control. Exactly 20 complete epochs, batch 128,
original FullStat two-epoch warmup/cosine AdamW and full-validation AUC-first selection.
No logit KD, D033 supervision, attention loss, ladder, fusion or warm start.
This separate plan supersedes neither the older KD+embedding final rung nor
the trial 299k/fixed-30 embedding sweep. Existing experiments stay immutable.

All seven arms share M0HLT native initialization and sampler seeds, auxiliary
head initialization, HLT inputs, optimizer and schedule. Loss is exactly
`CE + lambda * mean SmoothL1_beta1(g(h_HLT), z_OFFLINE)` in FP32, model BF16.
The unchanged 128-GELU-128 training-only predictor receives the native 128-vector
before classification. Its output never enters classification; deployment
contains only native HLT ParT. At lambda 0 the predictor has no backward path.
No gradient clipping, coefficient normalization or adaptive loss rescaling.
The embedded original base recipe records optimizer/schedule provenance only;
its legacy KD fields are unused. This independent recipe declares CE=1, KD=0.

Reuse the original selected pure OFFLINE teacher and FullStat cache, authenticated
in the original pinned worktree. Export TRAIN embeddings once in FP32/eval,
standardize using FP64 TRAIN population moments with scale floor 1e-3, store
FP32 targets. No validation embeddings or test particles. All consumers verify
exact ordered TRAIN identities/labels and target checksums, then load targets
into RAM. No per-batch random network reads or copied particle caches.

1.5M final-test and 105,325 unused rows stay sealed; event independence and old
trial overlap remain unverified. All results are adaptive, single-seed
development validation, not a test-selected winner. Recovery uses the NEW
lambda 0 control and frozen OFFLINE reference; the old M0HLT is labeled historical.
Poor metrics never fail admission or skip an arm, including extreme weights.
Invalid data, nonfinite losses/gradients, source drift and corruption fail closed.

One real Oscar L40S preflight (6 CPUs / 180,000 MiB / 8h) checks installed/native
FP32/BF16 auxiliary parity, exact zero-aux native CE miniature equality, OFFLINE
FP32 export maximum support, both roles' maximum support for every weight,
and one full 3M/500k technical epoch with selected native checkpoint/prediction
readback for EACH arm. Technical weights never initialize science. Measure
full-epoch time, input/target loading, authentication, peak RAM/GPU and storage;
use 2x conservative time requests, a 12h floor and 48h ceiling per science fit.

Science has seven independent GPU fits plus CPU aggregation. Retain selected
native checkpoints, selected validation predictions, complete epoch histories,
target bank, source and receipts; no per-epoch optimizer copies, automatic
resume or deletion. Declare 12 GiB personally available durable storage, also
check measured payload budget. Print batch progress/ETA every 128 updates and
per-epoch AUC/best AUC, CE, unweighted and weighted embedding losses.

Explicit opt-in `auto` queueing authorizes the exact one-gate/eight-science DAG
with bounded walltimes before launching anything. A detached submit-host
controller binds the exact gate journal, waits for matching job/owner/account
COMPLETED/0:0, reauthenticates source/receipts/resources, saves the measured dry
plan and submits only that graph. No science on failed/cancelled/missing or
corrupt gate evidence. Missing accounting waits, never passes. Permanent claims
and journals prevent duplicate or ambiguous retries; do not adopt manual
science roots. Poll every 60s, maximum 14 days. Nohup survives ordinary SSH disconnects,
not host reboot/controller death. No compute allocation held while waiting.

Additive source integration: `007885f7360f4cd28e9f91a690e04f0f4ea0afff`.
Local focused/installed-Weaver CUDA miniatures and failure injection precede
queue readiness; actual Oscar acceptance remains mandatory before science.

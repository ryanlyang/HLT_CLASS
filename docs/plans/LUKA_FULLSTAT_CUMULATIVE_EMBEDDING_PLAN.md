# FullStat cumulative representation ladder, Oscar, v1

User-authorized 2026-10-23: four fresh scientific fits, no standalone preflight
or setup job. This is not the proposed relational loss, old logit ladder, or
SPORC trial immediate-predecessor ladder. Those records remain unchanged.

## Science

Reuse the authenticated FullStat 3,000,000 TRAIN / 500,000 validation jet-row
split and prepared views. Final-test 1,500,000 and unused 105,325 rows remain
sealed. Event independence remains unverified. No rebalancing or new cuts.

| New node | Input | Frozen native embedding targets | Coefficient each |
| --- | --- | --- | --- |
| UE100 | U100 | original OFFLINE | 5 |
| DE066 | D066 | OFFLINE, new UE100 | 5/2 |
| DE033 | D033 | OFFLINE, new UE100, new DE066 | 5/3 |
| DE000 | D000 | OFFLINE, new UE100, new DE066, new DE033 | 5/4 |

Each node starts from scratch with the original coordinate-paired model and
sampling seeds. Exactly 20 complete epochs, physical batch128, existing
FullStat20 AdamW groups and two-epoch warmup/cosine schedule, BF16 native CUDA
forward, FP32 heads/losses, last partial batch retained. CE coefficient1;
embedding objective is (5/N) times the sum of N independently mean-reduced
SmoothL1(beta1) losses. No logit KD, warm start, attention, fusion, relational
loss, early stopping or adaptive loss weights. The coefficient sum is5; this
does not assert an unchanged gradient norm across rungs.

Each target is the selected frozen teacher's native 128-D pre-classifier
representation, NOT its auxiliary prediction. Export FP32/eval on TRAIN only;
independently standardize with FP64 TRAIN population moments, scale floor1e-3,
stored FP32 targets. Separate 128-GELU-128 prediction heads have isolated CPU
initialization streams; native initialization/dropout/sampling RNG is unchanged.
OFFLINE and corresponding U100/D066/D033 head seeds match the existing four-
teacher experiment. Never average unaligned teacher axes. Actual and predicted
targets never enter the classifier. All selected checkpoints contain native
weights only; DE000 alone is HLT-deployable. Intermediate validation rows are
explicit richer-input oracle diagnostics.

Reuse original OFFLINE bank bytes and original contract after validation.
Export exactly three new banks, one after each selected UE100/DE066/DE033 fit.
Cache each view once per job and reuse its TRAIN cache for outgoing export.
Never reconstruct ROOT views or store new particle caches/attention banks.
Keep selected native checkpoint, full validation predictions and all20 epoch
metrics/losses per node, plus immutable lineage/receipts. No per-epoch weights.

Selection: existing full-validation AUC-first selector and CE/logR50/earliest
update tie-breaks. No auxiliary/class/operating-point checkpoint selection.
Compare final DE000 against verified completed four-teacher and U100-w5,
historical M0HLT and OFFLINE; no new control fit. Report AUC/accuracy/recovery,
R30/R50 from the same selected probabilities, per-epoch progress. Adaptive,
single-seed development evidence, not independent test or matched total compute.
Poor scientific results always publish and release technically valid descendants.

## No separate preflight: explicit bounded reuse

Authenticate the completed b21c2d4e four-teacher campaign in its original clean
pushed source, including setup acceptance, completed fit, exact jobs, selected
probabilities, OFFLINE bank and its upstream U100/original FullStat lineage.
Reuse original Oscar seven-coordinate native acceptance and completed four-head
acceptance; do not relabel those as measurements of this new objective.
Require original L40S identity, installed environment/backend, 6CPU/180000MiB,
batch128, native model, prepared population and all coordinate support summaries.
Walltime inherits the larger accepted original/four-teacher training request
plus120 minutes for authentication/export, at most48h; reject if outside cap.
New fit runtime is unmeasured. Reserve12GiB personal durable headroom.

One reviewed submission queues exactly four jobs, train_UE100 -> train_DE066 ->
train_DE033 -> train_DE000. Each job also lists every earlier fit as afterok
dependency because all their banks are consumed. No preflight, setup, benchmark,
technical epoch, detached controller, automatic retry or additional submission.
Each job revalidates all prior stage receipts and successful exact-job accounting.

Inside the scientific job, authenticate joined targets/views and perform three
disposable maximum-TRAIN-support forward/backward updates, separate maximum-
validation-support forward, and native tap/inference parity with its actual
head set. Record finite gradients/memory/support and discard probes. These are
short integrated safety checks, not zero validation or a hidden full preflight.
Reset scientific initialization/RNG by constructing the scientific model afresh.
Do not change batches, particle capacity, inputs, loss or epochs if a check fails.
Technical failure prevents dependent jobs; low AUC does not.

## Acceptance and provenance

Additive package luka_fullstat_cumulative_embedding, new v1 namespace and thin
CLI/Slurm/helper. Original artifacts and runtimes are read-only. Donor execution
b21c2d4e01211a31e57d879c5a86ca90aab2ca0b; integration baseline
7e8dd105af1ead66b7ec120d7a391ca8310b5401. Record donors in LEGACY_SOURCE_MAP.
Clean pushed source, complete exact dry review and explicit four-job submission
authorization required. Exclusive claims/journals prevent ambiguous repeats.

Require focused baseline/regression, loss/seed/gradient and native parity,
TRAIN-only normalization/joins/corruption/ancestry tests, all-four-stage local
installed-Weaver GPU miniature, immutable export and probability readback,
failed-parent and poor-score continuation tests, exact shell/DAG validation.
Local tests are not new Oscar/Tigris resource certification. Existing genuine
Oscar evidence is reused, with new combined-view/head checks inside each job.
No full campaign is submitted by implementation alone.

# FullStat fine-tail contracts v1

Scientific authority: [active plan](../plans/LUKA_FULLSTAT_FINE_TAIL_PLAN.md).
All new artifacts have `LUKA_FULLSTAT_FINE_TAIL_<KIND>/v1`, schema 1,
canonical content hash, explicit parents and unchanged sealed FullStat flags.
Imported artifacts retain their original contracts; no relabeling, mutation or
pretending that an old view is D022/D011.

## Recipe and views

`RECIPE` registers `k1_e40_plain`: main CE0.5, total auxiliary KD1 at T2,
total embedding40, no relation. Each of N teachers receives KD1/N and E40/N.
KD uses independent auxiliary Linear128->11 heads, forward KL batchmean,
T-squared exactly once. Embedding predictors are independent
Linear128->GELU->Linear128, SmoothL1 beta1 mean. Teacher targets are detached;
neither true nor predicted teacher outputs enter the main classifier.
Teacher order is part of the recipe, including loss-report accumulation order
after canonical JSON roundtrip. FullStat optimizer/schedule remain 20 epochs,
warmup2/cosine20, physical128. Only the native selected backbone is saved.

`VIEW_RECIPE` binds the original physical D-kernel hash and exact fractions
D022=2/9, D011=1/9, D000=0. Saved original full-cardinality assignments,
source row IDs, identities/labels/order, support and capacity512 are unchanged.
Momentum interpolation and atomic identity/validity-aware tracking switches
operate on physical particles before ParT feature construction. New coordinates
have namespaced initialization/sampler seeds; D000 retains the original seeds.
Global historical coordinate APIs are untouched.

`VIEW_BUILD` records source-bound assignment and original D000 file references,
per-chunk new array hashes, original prepared hash, view hash, role summaries,
parallelism and elapsed time. Each chunk replays D000 exactly before acceptance.
Only TRAIN/validation particle entries are requested; original role-aware spans
may include excluded rows, never sealed-test/unused entries. ROOT basket byte
decompression and whole-file checksum reads are not test inference. Bound spawn
workers to six with six in-flight blocks, retain new view arrays in RAM only,
and preserve raw data/cache unchanged. D000 loads original cache directly.

## Frozen lineage

`REUSE` is emitted after replaying the completed non-relational split-head
finalization/retry source and authenticating selected native models, successful
task receipts/accounting and imported banks. It binds source bytes, parent spec,
exact environment, prepared population and original resource evidence.
Accept FINALIZE or FINALIZE_RETRY v1 only, with successful DE000 completion.
Existing teachers are OFFLINE, hybrid UE100, split-head DE066/DE033 and M0HLT.
Never select a teacher from the ongoing relational ladder or smaller trial.

New `BANK` and `KD_BANK` bind selected native checkpoint/report hashes and
prepared population. Native FP32 128D embeddings use per-teacher TRAIN-only
FP64 moments and scale floor1e-3; stored targets are FP32. Native KD probabilities
use the original BF16-forward/FP32-softmax T2 path. Both require exact ordered
TRAIN identity/label joins; validation cannot consume TRAIN targets. Native
outputs, not auxiliary predictions, are exported by DE022 and DE011 only.

`BUNDLE` binds all imported/new banks and completed new ancestor TASK hashes.
DE022 has five teachers; DE011 adds new DE022; DE000 adds new DE011 while
retaining DE022 and M0HLT. Total objective weights remain constant as N grows.

## Execution/publication

`SPEC` binds clean exact pushed source, REUSE, recipes/views, 3M TRAIN/500k
validation and three jobs. Final test1.5M stays sealed; unused105325 is unchanged.
Reserve at least8GiB personal persistent headroom; filesystem free space alone
does not certify quota. Reuse L40S/6CPU/180000MiB acceptance and verified
runtime_repair_v1. Conservative request is at least24h, limited to48h.

`STARTUP` is disposable in-worker evidence, not a separate preflight job:
native FP32/BF16 equality, CE-only main-head gradient, connected finite heads,
physical128 max-support updates, memory and bounded full-step timing. Include
actual preparation/authentication/loading elapsed time in the 20-epoch estimate;
fail before science if the allocation is insufficient. This is not a measured
full epoch or a guarantee. Discard technical model/optimizer and reset science.

`TRAINING_REPORT` validates all20 epoch histories, fixed schedule, exact loss
weights and native validation-AUC checkpoint selection. `FIT` publishes native
checkpoint, report and selected validation probabilities before outgoing banks.
`TASK` publishes last after bank readback, startup/view validation and environment/
source recheck. Descendants require exact successful scheduler IDs and valid
ancestor receipts. Poor finite performance is never a failure criterion.
Interrupted/partial fits are preserved, never silently overwritten or retrained.

`PLAN` has exactly3fits with afterok dependencies, no separate setup/preflight/
aggregate job, no requeue or automatic retry. `SUBMISSION_CLAIM` and immutable
submission journals prevent duplicates and bind exact accepted IDs.

Detached launch uses `LAUNCH_INTENT` -> full dry `PLAN` -> `LAUNCH_APPROVAL` ->
live submission -> `LAUNCH_COMPLETE`, or `LAUNCH_FAILURE`. Explicit token
authorizes only these three fits with the saved recipes/parent/source/quota.
An exclusive adjacent launch directory is created before nohup/setsid; the
foreground immediately tails the log. SSH disconnect does not cancel the
launcher; node reboot or site process policy is not protected. No old job is
signaled, no claim deleted, no implicit retry/r2, no final-test access.

Read-only reporting recomputes R30/R50 from saved validation probabilities at
the same AUC-selected checkpoint and compares original M0HLT/OFFLINE and prior
split-head endpoints. Only DE000 is HLT-only; D022/D011 are privileged teachers.
No new inference or test evaluation is authorized by result inspection.

# FullStat Multi3 contrastive contracts v1

Authority: [active plan](../plans/LUKA_FULLSTAT_MULTI3_CONTRASTIVE_PLAN.md).
All new artifacts use `LUKA_FULLSTAT_MULTI3_CONTRASTIVE_<KIND>/v1`, canonical
content hashes, explicit parent hashes and the existing sealed development
validation flags. Existing Multi3 and contrastive artifacts keep their own
unchanged versions.

`RECIPE` binds the old FullStat Multi3 recipe and old instance sampler. It fixes
20 epochs, batch128, CE1, U100/CONCAT/FUSION embedding weights10/3 each, pure
OFFLINE matching0.3 ramped over3 epochs, temperature0.1 and1024 negatives. Keys
use the existing OFFLINE TRAIN-standardized bank and the donor's NumPy FP32
L2 norm. The matching MLP has an independent isolated CPU seed. No privileged
input reaches the student forward path or native deployment.

`REUSE` binds the completed ceb21461 export-recovery campaign, exact saved job,
receipt, selected native checkpoint/report/predictions, four bank manifests and
payload hashes, owner specs, original-source inventory and U100 acceptance.
Authentication executes historical validators in the historical clean worktrees.
Only then may current code load those frozen banks using unchanged bank
validators. Unique identity and label arrays must equal TRAIN row-for-row.
Uncertified event/alias independence remains false.

`SPEC` binds clean pushed source and REUSE, a dedicated nonoverlapping result
root, exact original node seeds, recipe, references,6CPU/192000MiB/one L40S,
48h allocation,6GiB output budget plus2GiB headroom, and exactly one fit with
zero teacher or separate preflight jobs. `PLAN`, dry ledger, exclusive live
claim and append-only exact-DAG journal bind the submitted job. No automatic
resubmission after ambiguous scheduler responses.

`STARTUP` records actual installed-Weaver parity, exact reference sampler
comparison, full1025-candidate longest-support gradient stress, CE-only native
classifier gradient, native deployment parity, measured full-update timing,
validation timing and peak host/GPU memory, including summed CPU worker peaks.
Exactly three persistent CPU candidate workers are measured. Measurements are written before
admission. The conservative estimate is preparation +120min +1.5 times20
training epochs and22 validation passes; it must fit the allocation. Memory
must stay under90% GPU and85% host minus8GiB. Finite poor scientific results
never fail admission or completion.

Optimization preserves candidates, ordering, tie rules and FP32 dot reductions:
batched index uploads, eight-query mining kernels, stable canonical-then-score
GPU ranking, one hard-index readback, vectorized exclusions, and one-batch
lookahead reading only frozen keys on its own synchronized CUDA stream.
Three spawned CPU workers construct independent jet proposals and post-mining
components; ordered joins preserve every per-jet RNG call and candidate order.
The diagnostic stream never reuses a training-stream CPU pool.
Training keeps FP32 auxiliary heads/losses and the original BF16 backbone.
Teacher inference, target decompression and whole-bank scans are absent from
updates. Candidate evidence is collected for checks, not rebuilt solely for
discard on every training step.

`EPOCH` binds spec/prepared/four bank hashes/recipe, validation/loss/diagnostic
row, and an atomic full model+AdamW+CPU/CUDA-RNG state. Every completed epoch is
retained. `TRAINING_REPORT` requires20 contiguous epochs and selection by the
unchanged validation selection key. `TASK` binds exact scheduler job/source,
selected full/native states, all epoch states/receipts, report, startup evidence
and ordered validation predictions. Deployment reload must reproduce the saved
selected validation metrics. `RESULTS` compares the same row ordering and
AUC-selected checkpoints for AUC, accuracy, R50 and R30; no metric reselection.

`RESUME` is an explicit same-source, same-recipe continuation into a fresh root.
It requires terminal unsuccessful accounting, a contiguous authenticated epoch
prefix, and an exclusive claim on that failed attempt. Previous bytes stay
untouched; copies receive new parent bindings. Restore last model/optimizer/RNG,
original epoch permutations and ramp/LR position. If all20 epochs exist, perform
publication only. Changed-code recovery or an ambiguous submission requires
separate inspection; never infer permission to rerun from missing output files.

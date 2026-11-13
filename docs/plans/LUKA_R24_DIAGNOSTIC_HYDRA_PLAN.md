# R24 reproducibility diagnostic campaign

Active scientific plan, 2026-11-13. Hydra is the authorized execution site.
This additive campaign investigates the difference between the original R24
A6000 result and D02 on four GTX1080 Ti cards. These were both seed zero, but
the original native forward used BF16 and the curriculum used FP32, a new
training loop, synchronized BatchNorm and rank-specific dropout. They were
not an isolated seed comparison.

## Fourteen fits

Every row below is a matched pair: suffix `_C` is CE + U100 embedding prediction
weight 5; suffix `_R` adds the unchanged R24 reconstruction objective. R24 has
zero weight through continuous epoch position 3, a linear ramp over positions
3 to 6, and weight .3 thereafter. Global batch 128, 30 epochs, original AdamW
and learning-rate schedule, 299234 TRAIN and 74895 development-validation jets.
No optimizer reset, additional teacher or changed architecture.

| Pair | GPUs | Native precision | Worker | Only intended adjustment |
|---|---|---|---|---|
| P0 | 4 GTX1080 Ti | FP32 | curriculum | Repeat D01/D02 seed zero |
| PD | 4 GTX1080 Ti | FP32 | curriculum | Dropout RNG offset 1000003 |
| PO | 4 GTX1080 Ti | FP32 | curriculum | Training-order seed offset 1000003 |
| PI | 4 GTX1080 Ti | FP32 | curriculum | Native ParT initialization offset 1000003 |
| OB | 1 A6000 | BF16 | original immutable worker | Faithful original C0/R24 rerun |
| NB | 1 A6000 | BF16 | curriculum | New worker on the original precision/hardware |
| NF | 1 A6000 | FP32 | curriculum | New worker with FP32 native forward |

PI preserves the U100 predictor and reconstruction decoder initialization, and
the baseline training-order/dropout streams. PD changes both CPU and per-rank
CUDA training RNG seeds after initialization. PO changes the epoch permutation
seed only; a dropout mask therefore need not follow the same jet across orders.
Original worker RNG semantics are preserved for OB, including any difference
from the curriculum's explicit post-initialization reset. OB versus NB tests
the whole worker change; it does not isolate only the loop's arithmetic.

Predictor and reconstruction decoder stay FP32. Four-card fits use global
padding, SyncBatchNorm and sample-weighted gradient SUM; single-card fits use
ordinary BatchNorm. All deployment checkpoints are the native HLT-only ParT.
Complete OFFLINE particles are TRAIN targets only. Final-test access is sealed.
Recovery uses historical M0HLT/OFFLINE anchors; validation is exploratory and
original50k is not independent. No finite quality threshold gates submission.

## Numerical diagnostics and release

Two independent setup/release lanes prevent A6000 availability from delaying
the eight Pascal fits. Each setup runs genuine production miniatures with
reload, full-validation replay, longest HLT/OFFLINE support, full and uneven
batches, timing and aggregate host/per-GPU memory measurements for its cells.
Fresh fit startup miniatures are discarded before scientific initialization.

Pascal setup also compares an identical 12-update sequence on one and four
GPUs, including unequal shards and the zero/ramp/full-weight transition.
Dropout is disabled only for this numerical diagnostic. Loss, parameters,
gradients, BatchNorm buffers and AdamW moments are compared at every update.
This is a bounded numerical check, not proof of identical 30-epoch stochastic
trajectories. Tolerances are fixed in the versioned contract before execution.

Initial component hashes and actual RNG seeds are saved for every new-worker
fit. Epoch records retain CE, embedding and reconstruction losses, applied
weights, gradient diagnostics, validation AUC and selected checkpoint.
Original-worker artifacts retain their original schema and receive a separate
diagnostic-campaign binding receipt; the donor source and spec are unchanged.

Bootstrap submits Pascal setup/release first, then A6000 setup/release. Each
CPU release waits `afterok` on its own setup, validates measured resources and
queues independent fits in pair order, reconstruction then control. Reports
wait `afterany` on their lane's fits. No science fit depends on another fit or
the other lane. Queue order encourages starts; Slurm chooses actual starts.

Pascal requests 4 typed GPUs, 12 CPUs and 64000 MiB; A6000 requests 1 typed GPU,
6 CPUs and 32000 MiB. Setup limit 6h per lane; measured fit requests have 2x
margin plus 30 minutes, bounded at 48h. Reserve 9 GiB outputs plus 1 GiB headroom.
Exact pushed source, authenticated immutable parents, scheduler dry review and
exact-ID submission journals are required. There is no automatic retry.

Primary analysis is each R-minus-C selected AUC and same-epoch contrast.
Compare the three A6000 pairs to distinguish worker and precision effects;
compare P0 with NF to examine distributed/hardware effects together. Compare
PD/PO/PI with P0 to locate sensitivity to individual randomness sources. One
pair per factor does not estimate a seed distribution or prove a causal
explanation for the historical winner. Preserve all results, including losses.

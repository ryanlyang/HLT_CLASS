# FullStat soft-subclass contracts v1

Authority: [active scientific plan](../plans/LUKA_FULLSTAT_SUBCLASS_PLAN.md).
Artifacts use `LUKA_FULLSTAT_SUBCLASS_<KIND>/v1`, canonical content hashes,
explicit parents and sealed development-validation flags. Historical contracts
are unchanged.

`REUSE` requires the completed a7249f49 Multi3 contrastive campaign, job8361672,
authenticated in its original clean source. It freezes reference predictions,
selected checkpoint/report/receipt, all four teacher banks, historical runtime
inventory and original Oscar acceptance. Existing U100/CONCAT/FUSION targets
and pure-OFFLINE matching keys keep their original transformations and sampler.

`SUBCLASS_RECIPE` fixes OFFLINE TRAIN-only construction, QCD10/signal5 prototype
counts, cached native classifier readout,75/25 dimension-normalized embedding
and confusion-margin distance, deterministic single-thread MiniBatchKMeans,
and TRAIN-calibrated soft assignments. Full details and numerical settings are
in the recipe and plan. Finite degenerate clusters are reported, not rejected.

`SUBCLASS_BANK` binds prepared TRAIN identity, OFFLINE manifest/checkpoint and
target recipe. Immutable `targets.npz` stores canonical identities, labels and
FP32 `(N,10)` probabilities. Signal padding is zero. `prototypes.npz` stores
FP64 centers `(11,10,138)`, within-class margin statistics, source embedding
statistics and11 temperatures. Targets are nonnegative, finite and normalized
within true class. Wrong role, row order, teacher, payload or recipe fails.
The manifest records per-class hard/soft occupancy, entropy, achieved median
membership, distance and duplicate-center diagnostics. These are not physics
subclass labels or validation-selected definitions.

`RECIPE` preserves the complete Multi3 contrastive recipe and adds an isolated
Linear128→60 head, conditional KL coefficient0.5 ramped over3 epochs. Native CE
is1, three embedding losses10/3 each, OFFLINE contrastive0.3 ramped over3 epochs.
True labels choose auxiliary loss groups only. Every auxiliary head is removed
for native HLT-only inference.

`SPEC` binds exactly one fresh20-epoch student, existing3M/500k population,
source, reuse,8GiB minimum quota (6GiB output plus2GiB headroom), one L40S,
6CPU/192000MiB/48h. `PLAN` and append-only submission journals bind one exact
job; dry review precedes live submission. No teacher/setup/preflight jobs.
Detached launch is bounded authorization for this single plan, not retry.

`STARTUP` measures the new full objective on the actual allocation, including
longest-support gradients, installed-Weaver parity, exact optimized/reference
candidate parity, native classifier CE-only gradient, all auxiliary connections,
native deployment equivalence,64 measured updates and validation timing. Its
resource estimate includes target fitting time and process memory. Probe weights
are discarded before scientific initialization.

`EPOCH` and `TRAINING_REPORT` additionally bind the subclass manifest hash, full
model/optimizer/RNG, contiguous history, both ramp coefficients and every loss.
Native selected predictions must reproduce saved metrics. `TASK` includes both
target payloads, manifest, complete epoch states and publication inventory.
`RESULTS` uses the same AUC-selected checkpoint for accuracy/AUC/R50/R30 and
gap recoveries against M0HLT/OFFLINE; all comparisons use identical row order.

`RESUME` allows explicit same-source/same-recipe continuation into a fresh root
only after terminal unsuccessful accounting. It authenticates/copies the exact
frozen target bank and epoch prefix; it never refits subclasses. Completed20
epochs permit publication without optimization. Changed-source recovery and
ambiguous submissions require inspection, not automatic retry.

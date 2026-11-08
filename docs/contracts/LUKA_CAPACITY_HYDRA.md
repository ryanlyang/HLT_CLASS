# HLT capacity and complementarity contract v1

Authority: [active scientific plan](../plans/LUKA_CAPACITY_HYDRA_PLAN.md).
Runtime namespace: `hlt_classification.luka_capacity_hydra`. All new JSON
artifacts use `LUKA_CAPACITY_HYDRA_<KIND>/v1`, schema1, content hashes, exact
parents and the existing fullsim evaluation-disclosure flags.

## Registered inputs and models

SPEC binds the exact clean pushed source, immutable recipe, completed selective
parent spec, environment, prepared population, original U100 bank, selected IT
checkpoint/bank/receipt and completed S2IT/S4/S6 predictions/receipts/reports.
Original artifacts are authenticated by the original code in its pinned
worktree subprocess. Parent file hashes are checked again before reuse.
Runtime changes relative to donor `3fb9e7f11f628ad9d0d8c4c0d3694cc918e6e8a6`
must be additive and confined to this campaign's namespace, CLI and wrappers.

Twelve fresh fits: two widths (128/256), three objectives (CE, CE+U100-w5,
CE+U100-w5+IT-w5), two matched seeds. Width256 changes native embed_dims only
to [256,1024,256]; depth, attention heads and input support stay fixed. Pooled
prediction MLPs map the student width to128 to128, with isolated initialization
RNGs. Target normalization, SmoothL1 loss and seeds are versioned in RECIPE.
Native same-width/same-seed initialization and shuffles are objective-matched.
TRAIN target banks have exact ordered HLT identities and labels. CE reads none.
No target bank has a validation training role. HLT and privileged inputs are
never concatenated into a student input. Final-test access is forbidden.

All fits use the original thirty-epoch batch128 optimizer/LR and validation
selection rule. EPOCH binds the task, target parents, update count, loss terms,
metrics, schedule, timing and sampled weighted encoder gradients. Finite poor
performance is valid; it never cancels a fit or blocks another fit.

FIT inventories selected_full.pt, selected.pt, validation_predictions.npz,
training_report.json and deployment.json. TRAINING_REPORT authenticates all
epoch records and recomputes the selected epoch. DEPLOYMENT binds the selected
checkpoint and registered width, parameter count, state bytes and warmed
forward latency/memory. Deployed state contains mod.* only and is loaded
strictly through the width-aware Native loader. Prediction replay and metrics
must agree with the selected epoch. No auxiliary heads survive deployment.

## Complementarity and comparisons

COMPLEMENTARITY uses ONLY S2IT/S4/S6 HLT predictions, all three fixed equal
probability pairs and their fixed triple. Historical M0HLT/OFFLINE are recovery
anchors, never ensemble members. Unique identities are joined exactly with
matching labels; missing/extra/duplicate rows fail. Original50k uses the
authenticated overlapping subset. No learned weights, oracle selectors or
validation-driven ensemble membership.

Metrics include macro/per-class OVR AUC, CE, accuracy, QCD rejection and passing
counts/censoring, confusion matrices, and pairwise correctness/disagreement.
Full-validation AUC intervals use500 paired class-stratified jet bootstraps,
seed2026110709, against every member. Intervals describe jet-IID sensitivity
conditional on the already selected checkpoints. Event independence is
unverified; training-seed and checkpoint-selection uncertainty are excluded.
Recovery-point intervals hold historical anchors fixed.

RESULTS compares each new fit with same-width/same-seed CE and U100 controls,
and reports wide-minus-native and differences in supervision gains. Incomplete
controls are null, never substituted with historical runs. Both seeds receive
a descriptive mean/min/max, not a significance claim. PROGRESS validates only
published epoch records and is explicitly provisional. Historical references
and original50k do not become independent test evidence.

## Technical qualification and release

Three bootstrap jobs: independent CPU analysis, A6000 setup, and CPU release
afterok setup. Setup runs the actual production fit path for two technical
epochs in every cell, with real selected targets, FP32 installed-factory
initial parity, strict native reload, full/partial backward and largest-support
stress. Representative warmed timing is separated from stress. A historical
native checkpoint replay must meet the existing AUC absolute tolerance5e-5.

RESOURCES records all twelve profiles, GPU/environment, host/GPU peaks, twelve
timed steps, full validation time, diagnostics and preparation time. ACCEPTANCE
is recomputed from those measurements:2x safety,900s fixed overhead plus15min
startup allowance,48h cap; host peak below85% of32000MiB and GPU peak below88%
of VRAM. Wrong GPU, nonfinite/missing measurement, invalid lineage or resource
overrun fails closed with diagnostics preserved. Accuracy is no quality gate.

The authorized bounded dry plan covers analysis/setup/release, twelve fits and
report (16jobs). CPU allocation accepts one requested CPU with Slurm's observed
one/two allocated CPU rounding; no GPU is requested by CPU jobs. Release checks
exact setup COMPLETED/0:0, authenticated receipt, miniatures and ACCEPTANCE.
Only fit walltimes may shrink from the authorized envelope. Twelve fits have
no mutual dependencies; report is afterany. CPU analysis is not a quality gate.

Exact-ID journals and exclusive live/worker claims prevent duplicate retries.
Completed ledger reuse is idempotent; ambiguous partial journals require
inspection. Output publication is immutable and atomic. Existing campaigns
and their source/artifacts must remain unchanged.

See [runbook](../LUKA_CAPACITY_HYDRA_RUNBOOK.md) and
[validation evidence](../LUKA_CAPACITY_HYDRA_VALIDATION.md).

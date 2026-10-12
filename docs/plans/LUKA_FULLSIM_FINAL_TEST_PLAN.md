# Luka FullSim locked all-model final-test comparison

User authorization, 2026-10-12: evaluate all twelve completed stage-3 models
on the already sealed final-test population. This additive stage is the only
exception to the older Luka plans' test-access prohibition. It does not modify
ordinary loaders, source verifiers, splits, training, or historical receipts.

## Frozen scientific protocol

Bind science commit `01a738328b267da1ecc13bbf4263c2e4a2d9e4e2` and the completed
`luka_science_01a73832_r1` campaign. Require all 21 committed tasks, including
aggregate/completion, and twelve scientific validation-selected checkpoints.
No last-epoch substitute, checkpoint search, missing-model skip, refit,
calibration, or test-driven hyperparameter choice is permitted.

Use every eligible row in the frozen final-test files, exactly 149553 jets
across 337 files, split hash
`79ac6e577b92a5d2ed00d848c996812b388cab57f9d97d6b519f2bc10060c3b2`.
Unused ordinary-role rows do not become test. Preserve original selection,
natural proportions, provisional mm, zero-error masking, full cardinality,
matching, seven coarse views, 17-feature transforms and 512 support cap.
Do not tune or truncate if a test input violates support; fail technically.

All models predict the same ordered rows at T=1, batch256, CUDA BF16 forward
and FP32 softmax using the unchanged inference/model kernels. Offline-containing
models are explicitly oracle controls. D000/D000 is two HLT encoders, not a
single-HLT model; all single D000 models receive identical native HLT bytes.
All seven views are built once in RAM, with explicit growing-residency bounds.
No test particle/view cache is published. Selected weights are loaded one model
at a time; no optimizer, teacher bank, backward pass, or training routine runs.

Publish all validation/test accuracy, macro OVR AUC, CE, classwise QCD rejection
at 50%, and recovery relative to the test M0HLT and pure OFFLINE controls.
R50 is the unchanged ROC summary (threshold determined at 50% test signal
efficiency), not a validation-fixed operating-point efficiency. Zero passing
QCD is censored, not infinite measured rejection. The primary contrast is
FINAL_BRIDGE_D000 minus DIRECT_D000; also report bridge minus M0HLT.
Report paired source-file bootstrap 95% intervals for accuracy/AUC differences,
200 fixed draws, seed 20261012. These describe test-sampling uncertainty for
fixed fits, not training-seed variability or proven cross-file event independence.
No test-metric ranking changes the frozen inventory; any subsequent tuning or
winner selection is exploratory and cannot reuse this as an untouched holdout.

## Locks and execution

Create separate immutable finalist and execution locks before any test particle
read. Locks bind all checkpoints/reports, parent completion, full split,
preprocessing, metrics, exact new source and the measured SPORC environment.
Parent authentication runs in its original clean science worktree; new code
must leave every old tracked src byte identical and add only four named final
test modules. Do not weaken historical five-module source admission.

A new GPU worker first performs TRAIN-only saved-assignment/view replay and
all-checkpoint inference smoke checks, plus installed/native-mask parity. This
is an integrated miniature before unsealing, not another fit or scientific
selection. It then atomically consumes a campaign-wide execution claim in the
science root, before test access. The process-bound capability is required by
every test reader/cache entry point. Duplicate/interrupted executions stop;
successful committed outputs can be revalidated without decoding test again.
This creates only a new access marker in the old root, never edits old artifacts.

Two separately reviewable DAG jobs: one GPU evaluation, then one CPU report.
Retain measured debug A100 / 6 CPUs / 256000 MiB. Request the existing conservative
1386-minute training envelope for the single all-model evaluation, not an ETA.
CPU report: 1 CPU, 32 GiB, 4 hours. No new GPU/resource equivalence is claimed
until the integrated worker runs; runtime checks enforce the accepted identity,
environment, headroom, and workspace. A full dry plan, pushed clean source and
explicit exact-plan execution phrase remain required. No automatic retries,
cancellation, scheduler migration, or cluster submission by local implementation.

Predictions, ordered identities/labels/file groups and per-model results are
hash-bound, atomic evaluation artifacts (not reusable training banks). CPU report
uses only those committed predictions. All rows must complete regardless of
their metrics. The complete test receipt is published last; no partial results
are represented as a complete final comparison.

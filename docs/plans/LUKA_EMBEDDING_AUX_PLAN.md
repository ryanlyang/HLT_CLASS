# Luka HLT auxiliary offline-embedding prediction, v1

## Authorized experiment

One new, from-scratch **single HLT ParT** fit, not a ladder. Compare with the
authenticated completed fixed30 M0HLT CE-only fit. Use the completed OFFLINE
classifier as a frozen training-target generator and an explicitly labeled
oracle reference. No logit KD, attention loss, direct backbone-feature matching,
warm start, teacher labels, or offline inference input is authorized.

This tests whether predicting an informative offline summary through a small
nonlinear head improves the shared HLT encoder. It does not claim that offline
information lost in HLT can be recovered, or that gains are guaranteed.

## Population and matched comparison

Use the trial ALLJETS population: 299,234 TRAIN and 74,895 validation, the same
authenticated D000 inputs, provisional mm convention and frozen preprocessing.
The old holdout has already been consumed and explicitly retired into TRAIN;
there is **no independent test set**. Original50k validation is report-only.

Parent: fixed30 campaign at commit
`74aa58c582ff6f497a2c6d91c503f5489fac3e9a`. Authenticate M0HLT and OFFLINE
receipts, selected checkpoints, predictions and reports in their original
source. Match M0HLT's initialization and sampler seeds, batch 128, AdamW,
learning-rate schedule, native architecture, BF16 policy, 30 full epochs and
validation selection. Reuse M0HLT only after these checks; a missing or
incompatible parent is a blocker, not permission to silently substitute one.

## Fixed objective and model

Tap the final normalized 128-dimensional jet/class embedding immediately before
the installed Weaver ParT classification head (`mod.fc`). The teacher runs in
eval/inference mode. Its embedding targets are exported once for TRAIN only.
Compute per-dimension TRAIN means and population standard deviations in FP64;
use `max(std, 1e-3)` and store FP32 standardized targets. No target normalization
uses validation, and no extra compression model is trained.

Attach `Linear(128,128) -> GELU -> Linear(128,128)` to the student's same tap.
Initialize the predictor in an isolated RNG context so the backbone and dropout
sequence still match CE. Train every backbone parameter from scratch with:

`loss = CE(logits, labels) + 0.3 * mean(SmoothL1(prediction, target, beta=1))`.

The auxiliary loss is averaged across jets and 128 dimensions, evaluated in
FP32, and backpropagates through the head into the HLT encoder. The head is not
fed to the classifier. Standardization avoids a few large-variance dimensions
dominating; SmoothL1 limits outlier influence. These are fixed design choices,
not validation-tuned settings. Select the checkpoint by the existing full-val
AUC / CE / rejection / earlier-update ordering, never by auxiliary loss.

At deployment discard the predictor and load only the native ParT state into
the unmodified installed-Weaver wrapper. Verify probability parity after
serialization. This adds no inference parameters or operations.

## Storage and execution

Do not require the deleted durable OFFLINE caches. Rebuild only selected TRAIN
OFFLINE inputs in bounded memory from checksummed raw files; replay the original
input fingerprint exactly. Do not redo matching, rebuild other views, or write
particle caches. D000 TRAIN/validation caches remain authenticated read-only.

A TRAIN target is 128 FP32 numbers (~146 MiB for this population), plus compact
row identities/labels and normalization. No particle-attention banks, epoch
checkpoints, or copied datasets. Keep the compact target bank, reports,
predictions and one selected deployable checkpoint for reproducibility. The
temporary predictor is not persisted in that checkpoint. A 2 GiB dedicated
persistent allowance is required; never delete shared/old artifacts.

Default site is SPORC tier3 A100; explicit debug opt-in is supported. One
preflight job produces the authenticated target bank, performs a real full
population technical epoch, checks native parity and longest TRAIN/validation
stress, and measures RAM, GPU memory and runtime. Science is one new 30-epoch
fit plus compact reporting in one job, admitted only after exact preflight
COMPLETED/0:0 accounting and passing technical/resource checks. Never gate on
accuracy, AUC or auxiliary prediction quality. No automatic submission or
cancellation is authorized by implementation alone.

## Acceptance

Focused tests before/after changes; target joins, corrupt and partial artifacts,
TRAIN-only normalization, missing-cache rebuild/fingerprint parity, gradients,
head removal, RNG isolation, lambda-zero native-training parity, native
installed-Weaver tests, full fixed30 tiny fixture, exact dry-run/live journals,
resource rejection, source/recipe drift and result reauthentication. Remote
preflight remains required: local tests do not constitute SPORC acceptance.

# Luka single-HLT embedding-weight sweep

User authorization (2026-10-20): three SPORC tier3 fits with auxiliary weights
0.75, 1.5 and 10. The latest request moves both preflight and science from debug
to tier3. This supersedes the single-weight restriction of the original
auxiliary plan only for this new namespace. It does not change any old run.

Each fresh HLT-only model minimizes `CE + weight * SmoothL1(predicted, OFFLINE)`.
There is no logit KD, attention loss, ladder, fusion or warm start. Match the
original M0HLT initialization/sampler seeds, native ParT, batch 128, AdamW,
fixed 30-epoch learning-rate schedule and full-validation AUC-first selection.
Use the same trial ALLJETS 299,234 TRAIN / 74,895 validation; original50k is
report-only. The former test was retired; no independent test is available.

Reuse the completed 0.3 experiment's frozen TRAIN-only standardized 128-vector
bank read-only, authenticated in its original source. Predictor and target
normalization are unchanged. The prediction head receives gradients but is not
fed to the classifier and is removed from the native deployment checkpoint.
Report CE, unweighted and weighted auxiliary loss each epoch. Weight 10 is intentionally
aggressive: low accuracy or weak prediction must never cancel it.

One shared real A100/tier3 preflight runs native/zero-weight parity, longest
TRAIN/validation-support stress and a full technical epoch for each weight.
Per-weight measured resource admission retains the registered 24h ceiling; numerical
failure or corruption fails closed. Science is three independent jobs, not
three sequential fits sharing optimizer/model state. Exact pushed source,
dry review and successful accounting remain required; no automatic follow-up.

Reserve 2 GiB for the entire new campaign. No new particle caches or target-bank
copies; no epoch checkpoints. Retain one native selected checkpoint, predictions,
history and receipts per model, plus compact comparison snapshots. Never delete
shared data, old models/banks or Fresh_check. Interrupted outputs require review,
not automatic overwrite/retry.

Compare all three against authenticated CE, OFFLINE oracle and existing0.3
predictions on identical validation rows. This is a one-seed exploratory weight
sweep, not an independent test or a guaranteed improvement.

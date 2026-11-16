# Balanced DE000 with KD on the deployed head

Status: implementation and local validation, 2026-11-16. This is the active
scientific plan for the one-fit Oscar follow-up requested by the user.

## Question and scope

Does direct DE050 logit distillation improve the balanced HLT-only DE000,
compared with the completed balanced strategy whose deployed head receives CE
only? Train one fresh DE000 for 20 epochs. Reuse the five completed balanced
references from `luka_fullstat_balanced_ff22b20f_r1`; do not retrain teachers or
warm-start from the old DE000. The old DE000 may still be running when this
campaign is registered. No jobs are cancelled or submitted by implementation.

## Objective

Let `q_t = softmax(frozen_native_teacher_logits / 2)` and
`K(q,z) = 4 * KL(q || softmax(z / 2))`, with class sum and batch mean.
The saved teacher probabilities already use temperature 2.

```
L = 0.25 * CE(main_logits, truth)
  + 0.75 * K(q_DE050, main_logits)
  + 2.5  * sum_t SmoothL1(predicted_embedding_t, standardized_embedding_t)
  + 0.25 * sum_t K(q_t, auxiliary_logits_t)
  + ramp(update, three_epochs, maximum=0.3) * OFFLINE_contrastive
t = OFFLINE, CONCAT, UE100, DE050
```

SmoothL1 uses beta 1 and mean over rows and dimensions. The four auxiliary
logit heads remain separate; their total coefficient stays 1. The new main
KD term uses DE050's native probabilities, not its auxiliary prediction head.
Only CE and the new direct KD term differentiate into the deployed classifier;
all objectives can train the shared encoder. Teacher quantities are detached.
No subclass objective is present in this balanced experiment.

This replaces CE1 with CE0.25 + KD0.75 on the main head. It therefore tests
that mixture, not an isolated addition of KD at unchanged CE strength.
No guarantee of better recovery is implied.

## Frozen comparison conditions

Use the original authenticated 3M TRAIN membership and its 2,425,536 eligible
rows with offline reconstructed jet pT in `[200, 2000)` GeV. All 11 classes
and all 18 bins of width 100 GeV have equal expected draw probability. Draw
2,425,536 times per epoch, with replacement: uniform class/bin cell, then
uniform row within that cell. Batch 128, final batch 64, 18,950 updates/epoch.
Reuse the original native/auxiliary initialization, occurrence draw and negative
sampling seeds, AdamW, 2-epoch warmup and cosine schedule to epoch 20. No
extra class or pT loss weighting, early stopping, or input truncation.

Use the same OFFLINE matching head, 1,024 negatives, temperature 0.1,
256/256/512 negative mixture, 4,096 hard proposals and rank interval 257–1024.
Repeated training draws retain occurrence-specific negatives. TRAIN targets
and normalization are frozen from unique eligible rows, not replacement draws.
There is no additional teacher forward pass during student updates.

All 500,000 natural validation jets remain unchanged, including out-of-range
pT. Select by maximum macro AUC, then minimum CE, maximum macro log R50,
earliest update. Publish natural/in-range/outside/per-bin validation metrics
and R50/R30 for that one selected checkpoint. Recovery uses new balanced
M0HLT/OFFLINE anchors within each reported population; nonpositive anchor gaps
give unresolved recovery, never job failure. Final test remains sealed.

Deployment exports only the native HLT ParT. Teacher inputs, predictions,
auxiliary heads and matching head are absent from inference.

## Provenance and execution

Introduce `LUKA_FULLSTAT_BALANCED_MAIN_KD_*/v1`. This explicitly supersedes the
original CE-only assertion for this namespace only. Original balanced artifacts
retain their original contracts. Authenticate the five references through their
clean pinned original source commit `ff22b20f27fdc67d0fbd1a1df7005eb4626d19b7`,
including scheduler completion, source/runtime evidence, result inventories,
epoch/report/checkpoint lineage and frozen banks. Check the conservative local
import closure of the reused production helpers against that source before
using them in the new worker. Reuse the original population/view/bank contracts
without relabelling those artifacts as a new dataset.

The user's requested no-separate-preflight workflow uses integrated acceptance
inside the single science allocation: installed-Weaver/native parity, all-loss
finite gradients, explicit main CE/KD gradient equality, serial/optimized
negative equivalence, full-support stress, measured memory and timing. Discard
the probe and reset all model/optimizer/RNG state before the scientific fit.
These checks replace a standalone preflight job; they do not claim prior real
Oscar or Tigris execution. Failed checks publish diagnostics and stop.

Request one L40S, six CPUs, 192000 MiB host memory and 48 hours. Require at least
8 GiB available result quota; budget 6 GiB for 20 complete epoch states, selected
native checkpoint, validation predictions and reports. No outgoing teacher bank
is generated. The detached launcher authenticates, writes/reviews a dry plan,
then submits exactly one authorized job; duplicate/ambiguous submission fails
closed and keeps journals. No automatic requeue.

Commit each complete epoch with optimizer, full student, RNG and draw cursor.
Explicit fresh-root recovery can resume this new namespace only. If 20 epochs
are committed, recovery repeats publication without training. Never reinterpret
old CE-only optimizer states as this new objective. See the
[runbook](../LUKA_FULLSTAT_BALANCED_MAIN_KD_RUNBOOK.md),
[contract](../contracts/LUKA_FULLSTAT_BALANCED_MAIN_KD.md) and
[validation evidence](../LUKA_FULLSTAT_BALANCED_MAIN_KD_VALIDATION.md).

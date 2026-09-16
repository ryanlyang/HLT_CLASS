# JetClass2 native offline + HLT concatenation oracle

Implementation authority, 2026-09-16. This is one independent CE fit, not a
new ladder, distillation run, or deployable HLT classifier.

## Scientific definition

Read the same exact TRAIN_500K and shared validation membership as the selected
JetClass2 learned-handoff campaign. For each jet concatenate **native offline,
then native HLT** particles. Do not concatenate persistent-HLT U000 with HLT:
that would duplicate its unmatched-HLT residuals. No matching, interpolation,
deduplication, or truncation enters this new input construction.

Compute the canonical 17 numerical features separately on each reconstruction,
using its own constituent-sum axis and relative normalization. Preserve native
stored four-vectors. A single canonical 11-class ParT attends jointly over both
collections, including cross-reconstruction pair interactions. Its shared
17-feature embedding receives an additional learned 128-dimensional source
embedding (offline=0, HLT=1). This is an explicit oracle-only exception to the
ordinary input contract's prohibition of source categories. Row IDs, labels,
matching maps, file/particle indices and salience never enter the model.

Native capacity stays 240 per side; combined capacity is 480. Fail on overflow;
trim=False. Pad only to the batch maximum (minimum 16). Source codes travel as
an 18th transport channel and are removed before the numerical embedding;
padding uses -1 and contributes no source embedding. This does not redefine
the ordinary 17-feature model or any existing campaign.

Use the learned-handoff CE_SINGLE_D000 initialization and sampler seeds for
the shared ParT; initialize source embeddings in a separate RNG domain. Train
fresh with CE only, batch 256, no accumulation, AdamW, BF16 forward / FP32 loss.
Keep the existing schedule: warmup 1-3, hold 3e-4 through 45, cosine to 1.5e-5
by 60, floor through 100; minimum 60, patience 15 starting after 60, delta
5e-5, restore the best AUC/CE/log-R50/update-ranked checkpoint.

## Comparison and publication

Reuse and authenticate the existing V_checkpoint/V_diagnostic/V_report split.
Select solely on V_checkpoint, evaluate selected weights on V_report. Compare
against completed M0HLT, persistent-HLT U000, and CE_SINGLE_D000 reports from
that exact campaign. Label U000 accurately: it is NOT a pure-offline reference.
The read-only results command may add STATIC_U000_D000 once its authenticated
report exists; its completion is not a submission dependency.

Report accuracy, AUC, macro R50, classwise QCD rejection and linear recovery.
Keep zero-background censoring as null with its one-event resolution, not an
invented finite R50. No guarantee of an improvement and no performance gate.
Final test remains sealed. A pure-offline comparison needs a separate matched
reference in future; this request does not authorize a second fit.

## One-job execution

Create an immutable standalone spec and dry run in a fresh root, with exact
pushed source and read-only reference parents. One SPORC tier3/A100 job performs
source/data/receipt checks, native RAM-cache construction, installed-Weaver
zero-tag parity, actual-data miniature training and full-batch longest-view
GPU/optimizer memory checks, then resets all model/RNG/optimizer state and
runs the full fit automatically only after its own acceptance passes. This
study-specific in-job miniature replaces a separate gate job, not the gate.
Existing fusion acceptance does not certify concatenation memory use.

Default envelope: 8 CPUs/workers, 128 GiB host RAM, one A100, 48 hours. This is
a requested ceiling, not a measured throughput claim. Conservative total
cache/worker bounds must fit 75% of RAM; the probe must fit 85% of GPU memory.
The GPU stress length is the largest actual concatenated length across the
entire authenticated train/validation population, with batch 256, never a
percentile or a truncated sequence. Reject a slow projected run before science
if its conservative estimate exceeds the requested walltime. No silent batch,
precision, sequence or schedule reduction. CPU count/worker count may be
frozen differently when creating a fresh spec; never mutate existing jobs.

Persist selected weights, training history, acceptance, comparison and output
hash manifest only. Particle caches/probabilities/optimizer stay RAM-only.
No automatic retry or overwrite; use a fresh root after a terminal failure.
Submitters require explicit authorization, canonical dry run, a submission
claim and durable intent/receipt. Workers never submit/cancel any other jobs.

## Acceptance tests

Test native token preservation including HLT>offline, per-side normalization,
source code/mask integrity, role sealing, ordered parallel preparation, memory
bounds, unchanged canonical numerical embedding, source gradients, ordinary
zero-tag Weaver logits/gradient parity, matched seed and RNG reset, validation
identity joins, source/output tamper refusal, one-command dry/live safety and
synthetic train/selected-checkpoint/report publication. Installed-Weaver/A100
evidence is produced on SPORC; local synthetic tests do not imply that evidence.

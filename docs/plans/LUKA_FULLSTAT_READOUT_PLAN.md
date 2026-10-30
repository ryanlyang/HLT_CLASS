# FullStat U100 frozen-readout CE on Oscar

Status: locally implemented/tested; exact pushed release and Oscar execution pending.
This is the explicitly requested FullStat adaptation
of `LUKA_U100_FROZEN_READOUT_CE_HANDOFF.md`, not a change to the smaller multiview
campaign. One fresh HLT-only student; reuse completed U100-w5 as a historical
reference. No fresh control, new ladder, warm start, or standalone preflight.

## Registered science

- Existing authenticated 3,000,000 TRAIN / 500,000 validation population;
  1,500,000 final-test and 105,325 unused rows remain sealed. Paired jet-row
  stratification is retained; event independence is unverified.
- Use the original regular FullStat COARSE_U100 selected checkpoint and TRAIN
  bank authenticated through U100-w5 source
  `7ea8b0c693734a26ced3820354e047f4af26c979`. No small-data or hybrid UE100 artifacts.
- Fresh native HLT ParT with the historical U100-w5 predictor, initialization,
  sampling, AdamW grouping, batch 128 and 20-epoch warmup/cosine schedule.
- `L = CE(native,y) + 5 SmoothL1(prediction,z) + a CE(W(mu+sigma*prediction)+b,y)`.
  `a=min(completed_updates/ceil(Ntrain/128)/3,1)`. No temperature or logit KD.
  Mean/scale come exclusively from the authenticated TRAIN bank; FP64 stored
  moments are converted to FP32 for the frozen readout. W,b are the selected
  teacher's exact native Linear(128,11) classifier, frozen buffers. All readout
  math is FP32. Its input remains differentiable; shared encoder and predictor
  train, while the native classifier receives only its own CE gradients.
  Record unweighted loss gradient norms/cosines on the first TRAIN batch of
  epochs 1, 3, 10 and 20 using the existing graph, without extra updates or RNG.
- Native validation AUC selects the checkpoint using the existing tie-breaks.
  All 20 epochs run even if finite performance is poor. Predictors/readout are
  removed at inference. No true or predicted teacher values enter that classifier.

## Execution and evidence

One Oscar L40S GPU job, 6 CPUs, 180000 MiB, at most 24 hours, 4 GiB durable
result allowance. Authenticate old source/results/bank and exact pushed additive
source before submission. The job performs bounded teacher/readout parity,
installed-Weaver parity, largest-support gradient/memory probes and measured
timing admission before its fresh fit. Probe updates are discarded. Existing
Oscar evidence is reused; new workload checks are recorded separately, never
called a new full-epoch preflight. No extra scientific fit is hidden in setup.

Persist complete student/predictor/optimizer/RNG state and validation history
after every epoch. Persist selected native and training state before final
readback. Failures retain all evidence; no implicit resume or duplicate submit.
Atomic hashed publication binds bank, classifier, normalization, source, spec,
selected epoch and exact scheduler job. Historical roots are read-only.

Report native AUC, accuracy, R50/R30 and gap recovery against historical M0HLT
and OFFLINE, with historical U100-w5 comparison. These are adaptive single-seed
validation comparisons, not a matched new-control or independent-test result.

The detached one-command launcher records bounded authorization, full dry plan
and exact hash, then submits one job. SSH disconnect does not stop it. No
campaign is submitted as part of local implementation.

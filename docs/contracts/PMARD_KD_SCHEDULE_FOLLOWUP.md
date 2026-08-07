# PMARD Paired KD Schedule Follow-up

Contract family: `v1`

This validation-only study resolves the confounding seen when the original
10-, 20-, and 40-pass rows shared a fixed peak learning rate and validated
only eight times. It compares CE-only training, T0 self-distillation, and the
best T100 dual-teacher recipes under matched optimization and observation,
including a registered 60-pass extrapolation.
It does not open test jets or promote a deployable model.

## Frozen recipe selection

The follow-up may be created only after the 36-row T100 sweep has published a
valid aggregate report. At each parent-grid exposure (`10`, `20`, and `40`
complete train passes), it takes both the lowest-validation-CE T100 recipe and
the best registered PMARD utility recipe. The 60-pass extrapolation freezes
the two recipes selected from the longest available parent exposure, 40
passes; it does not select a recipe using any 60-pass result. If one row wins
both rules it is run once and records both selection roles. This selection is
source- and hash-bound before any follow-up model is trained.

For every registered schedule/exposure group, all models start from scratch
with the same initialization and sampler seeds and consume the same 300,000
natural-population HLT train rows. The controls are:

- `K0`: class-weighted CE only;
- `K1`: 25% CE plus 75% T0 self-KD;
- selected T100 rows: the exact CE/T0/T100 weights and independent
  temperatures inherited from the parent sweep.

Every model is evaluated on the same 100,000 validation rows exactly after
each complete train-role pass. Selection remains minimum validation CE, then
maximum accuracy, then earliest update. Thus the 20-, 40-, and 60-pass results no
longer hide instability between sparse validation boundaries.

## Optimizer schedules

The parent peak learning rate is `L` (currently `3e-4`). Ten-pass rows use
`L`. Twenty-, forty-, and sixty-pass rows each run two diagnostics:

- `scaled_lr`: `L * sqrt(10 / passes)` (`2.1213203436e-4`, `1.5e-4`, and
  `1.2247448714e-4`);
- `fixed_lr`: the original `L`, retained to isolate learning-rate effects.

All other optimizer semantics, including the five-percent warmup, cosine
decay, minimum-LR fraction, weight decay, batch size, and class weighting,
remain unchanged. At batch size 256, validation occurs every 1,172 updates
and total budgets are 11,720, 23,440, 46,880, and 70,320 updates.

When the CE and utility winners differ at every exposure, the registry has 28
rows: four at 10 passes and eight each at 20, 40, and 60 passes. The Slurm array is
uncapped. The aggregate reports direct deltas from each T100 row to its
schedule-matched K0 and K1 controls.

## Reuse and access boundaries

The follow-up reuses the parent sweep's authenticated compact float32 T0/T100
logit cache. It never reruns matching, rebuilds repaired particle views, or
publishes particle datasets. K0 does not load teacher logits; K1 uses only T0;
dual-KD rows use both T0 and T100. Every training input remains HLT-only.

The specification binds the clean source snapshot, separate execution
worktree, parent sweep specification and aggregate, target manifest and target
file hash, complete derived registry, and cluster site. Consumers revalidate
the parent campaign lineage and cache before use. Missing rows, non-epoch
validation histories, altered losses, optimizer drift, or lineage drift fail
aggregation closed. Scientific underperformance remains a valid result.

Durable contracts:

- `hlt_classification_pmard_kd_followup_spec_v1`;
- `hlt_classification_pmard_kd_followup_report_v1`;
- `hlt_classification_pmard_kd_followup_ledger_v1`;
- existing `hlt_classification_pmard_training_report_v5` and resume v5.

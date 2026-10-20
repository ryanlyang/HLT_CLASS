# Alternating privileged and self-contained gate ladder - v1

Active scientific authority for the new `luka_alternating_gate` package only.
Existing campaigns, splits, checkpoints and scientific code are read-only.

## Question and registered comparison

Does removing dependence on the richer view **at each rung**, rather than only
at D000, improve a deployable HLT classifier? A second question is whether
matching the teacher's additional gate helps more than logit KD alone.
This is a hypothesis, not an expectation or admission criterion.

All arms start from the same authenticated, selected fixed30 COARSE_U100.
Here U100 is the existing aligned starting view (sometimes called D100 in
discussion), not a new dataset coordinate. Use the existing interpolated
D066/D033/D000 views, NOT the particle-swap or FullStat campaigns.

At each coordinate x, the sequence is TEACH_x -> SELF_x. Only SELF_x becomes
the teacher for the next coordinate. Thus, for each arm:

```text
COARSE_U100
  -> TEACH_D066 -> SELF_D066
  -> TEACH_D033 -> SELF_D033
  -> TEACH_D000 -> SELF_D000
```

| Arm | TEACH gate inputs | SELF gate inputs | SELF supervision |
| --- | --- | --- | --- |
| CONTROL | current/current | current/current | labels + immediate TEACH logits |
| LOGIT | current/richer | current/current | labels + immediate TEACH logits |
| GATE | current/richer | current/current | same, plus 0.1 centered log-gate MSE |

For LOGIT/GATE, richer is U100 at D066, D066 at D033, and D033 at D000.
Every TEACH receives labels and logit KD from the preceding SELF (or anchor).
Every SELF receives targets exported from its own TEACH **with the TEACH's
registered richer context**, not from an accidentally context-free teacher.
The CONTROL arm has the same six-fit budget and final architecture, but never
receives richer gate context. All 18 fits are independently cold initialized;
same per-coordinate initialization/sampler seeds as the parent controls, fresh
optimizer, no warm start, frozen phase, early stop or performance-based pruning.

## Fixed data, model and optimization

- ALLJETS development split: 299,234 TRAIN / 74,895 validation. Former test
  explicitly repurposed into training; there is **no independent test**.
- Original 50,000 validation rows are report-only using the same checkpoints.
  All rows use identical ordered authenticated populations and aligned slots.
- Batch 128, fixed 30 complete passes, existing AdamW/warmup3/cosine recipe,
  0.25 label CE + 0.75 temperature-2 logit KD (existing T-squared convention).
- Full-validation macro OVR AUC selects the checkpoint, then existing tie rules.
- Single encoder/native Weaver ParT. Native Q/K/V and pairwise physics bias
  remain intact. Reuse the bounded rank-4 nine-factor gate, shared across heads
  in particle blocks 6/7: add log(4)*tanh(u_i.v_j/2 + b_j) to the native bias.
- Gate MSE matches valid-key-centered log corrections, averaged equally per
  jet; teacher factors are detached. It does not match full native attention.
  Auxiliary targets are never supplied to the student's forward method.
- SELF has a gate predicted from current/current inputs and rejects every
  external context argument. Only SELF_D000 endpoints (and CONTROL TEACH_D000)
  are HLT-only. Earlier SELF views still contain offline information; privileged
  TEACH_D000 is an oracle. No claim of full removal of unavailable information.

## Execution and retention

One new SPORC A100 **debug** preflight. It rehearses all 18 fits with one full
population pass, separate TRAIN/backward and validation/forward maximum-support
batch-128 stresses, installed-Weaver FP32/BF16 parity, fresh initialization,
checkpoint/probability readback, and exact teacher/slot lineage. Science has 18
per-fit jobs plus one aggregate, afterok dependencies, no retries or automatic
follow-up. Each science fit receives its own measured walltime with 1.75 safety
factor and 24h maximum. Resource/lineage failures stop admission; scores never do.

Targets are exported by their consumer into private verified node-local scratch.
Only GATE TEACH nodes export nine FP32 factors per particle; no NxN attention
bank. After durable selected outputs and readback verification, retire only
that consumer's enumerated owned target files, with an immutable deletion receipt.
Keep small selected checkpoints, predictions, all epoch metrics and lineage;
these are needed for descendant authentication/re-export. No epoch checkpoints,
optimizer snapshots, persistent target banks or copied particle caches.
Declare at least 2 personally available persistent GiB; enforce a 2 GiB campaign
payload ceiling, and independently check scratch/headroom. Never clean old runs.

## Acceptance and interpretation

Local tests must cover the full 18-fit graph, all three context-removal steps,
factor-only supervision, same-coordinate versus downward teacher routing,
source/metadata tampering, post-retirement results, safe cleanup and exact dry
submission. Genuine cluster preflight is required before science, regardless of
local CUDA success. No job is submitted by implementation or tests.

Compare LOGIT_SELF_D000 and GATE_SELF_D000 to CONTROL_SELF_D000 and the retained
M0HLT/OFFLINE/DIRECT_D000 references. Report intermediate transfer gaps, complete
validation histories, input scope and extra training budget. One seed and reused
development validation cannot establish generalization or statistical significance.

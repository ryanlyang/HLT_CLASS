# HLT-learnable shared messages on Hydra

Scientific registration v1; twelve fresh HLT fits. This plan supersedes neither
the existing cross-message campaign nor its immutable artifacts. Its hypothesis
is that a shared downstream classifier can encourage useful messages that the
HLT path can also produce. Greater oracle recovery alone does not establish
transferability. No recovery improvement is promised.

## Architecture and objective

Every fit retains CE and the original frozen COARSE_U100 jet embedding target
with coefficient **5**. All deployable inputs are the same authenticated D000
HLT features, four-vectors and masks. Initialization is fresh, not continuation.

Run native Particle Transformer blocks 1–4 once. An HLT predictor reads these
128-dimensional contextual tokens, their own 17 features, and the native
eight-head pairwise bias. It uses a 128-wide four-head attention block with a
learned eight-to-four bias map, a residual feed-forward layer and a 16-dimensional
output. A privileged cross-attention producer reads the same HLT context and
frozen COARSE_U100 final particle tokens. U100 tokens are detached; neither the
encoder nor the original embedding targets are updated.

Both messages undergo the **same non-affine layer normalization before both
comparison and injection**. They use one shared, zero-initialized 16-to-128
expansion. Each route then uses the very same blocks 5–8, class aggregation and
classifier. There is no separate privileged classifier. The HLT path alone is
used for deployment, validation selection and reporting.

At training position t (epochs, including fractional progress), minimize

```
(CE_HLT + alpha(t) * CE_privileged) / (1 + alpha(t))
  + 5 * SmoothL1(predicted_U100_embedding, original_U100_embedding)
  + beta(t) * mean_jet(mean_valid_particle(mean_channel(SmoothL1(m_H, m_U))))
```

The embedding term uses only the HLT route. The agreement normally differentiates
both producers, including their shared HLT context. There is no logit KD. For
the first three epochs beta is zero; it rises linearly to its registered value
at epoch six. Alpha is active from the start. The fade variant multiplies alpha
and beta by `clip((20-t)/10, 0, 1)` after epoch ten. Neither loss uses validation
examples. Layer normalization removes scale shortcuts, but cannot guarantee
noncollapsed or useful targets: message variance and agreement are diagnostics,
never accuracy-based admission gates.

## Registered fits, preferred submission order

| Task | alpha | beta | Difference |
|---|---:|---:|---|
| joint_b0p1_s0 | .25 | .1 | Primary proposal |
| control_s0 | 0 | 0 | Identical deployed architecture, CE + U100-w5 |
| joint_b0p3_s0 | .25 | .3 | Stronger agreement |
| joint_b0p03_s0 | .25 | .03 | Gentler agreement |
| frozen_b0p1_s0 | 0 | .1 | Same HLT predictor, existing frozen teacher_m16 targets |
| joint_b0p1_s1 | .25 | .1 | Primary proposal, second seed |
| control_s1 | 0 | 0 | Matched second seed |
| joint_a0p1_s0 | .1 | .1 | Less privileged CE |
| joint_b1_s0 | .25 | 1 | Strong agreement |
| joint_no_agreement_s0 | .25 | 0 | Shared training alone |
| joint_student_only_s0 | .25 | .1 | Detach privileged message in agreement only |
| joint_fade_s0 | .25 | .1 | Fade both privileged terms over epochs 10–20 |

The frozen comparison uses the completed original m16 teacher and its own HLT
context, not the student's context. Its compact targets receive the same
non-affine normalization. Its weights, targets and privileged encoder are frozen.
Controls contain no privileged producer and consume no U100 particle tokens;
they still have U100 **jet embedding supervision**. They are not CE-only controls.

All twelve use batch 128, 30 epochs, the existing AdamW/cosine schedule and
matched native initialization/sampler seeds. Save the earliest best checkpoint
under the donor selection key (AUC, then its registered tie-breaks). Selection
always uses HLT-only full validation. Compare each arm with its same-seed new
architecture control; historical U100-w5 is a secondary reference. Report
`100*(AUC-M0HLT)/(OFFLINE-M0HLT)` with historical anchors explicitly labelled.

## Data, execution and implementation

Reuse the authenticated A6000 v3 parent at commit
`199928da991f8a4194dcf42df009130cfcf84118`, its completed setup and completed
teacher_m16, original input bundle, adapter delta, U100 encoder delta and venv.
No new SPORC transfer. Authenticate the old worktree in its own Python process.
New runtime is additive relative to `546077aa7fce428bf3f9f136ab2277ac5eed5f85`.

TRAIN has 299,234 jets; validation 74,895. Original50k overlaps validation and is
report-only; no independent test is available. Only TRAIN U100 tokens are built
for this campaign, in allocation-local memory. No privileged validation forward
is needed. New checkpoints retain the HLT predictor/expansion and omit both the
privileged producer and the auxiliary embedding head.

Hydra uses one RTX A6000 per job, `gpus/default/normal`, 6 CPUs and 32,000 MiB RAM.
Do not pin the node. Setup runs warmed production miniatures, including full and
partial batches, both routes, backwards/AdamW, checkpoint publication/reload and
HLT validation. Technical miniatures run at schedule position seven so agreement
is exercised. They retain the largest128 jets in a full batch and at least two
jets in the partial batch, avoiding native BatchNorm singleton support. Resource
projection excludes disposable warmup from per-step
timing, includes preparation once, uses 2x measured time plus 900s, and admits
only finite estimates within 48h and 85% of RAM/GPU limits. No quality threshold.

Launch queues GPU setup and an `afterok` CPU release job in `compute`. The latter
authenticates acceptance and submits all twelve independent fits with measured
walltimes and a report depending on all fits via `afterany`. It survives logout
without a login-host watcher. CPU release failure permits explicit, journal-aware
`finish`; ambiguous partial submissions fail closed rather than duplicate jobs.
There are 12 scientific fits and 3 orchestration jobs (setup, release, report).

Immutable specs, exact-job journals, source/environment/input hashes and atomic
outputs are mandatory. A fresh campaign root preserves existing experiments.
Four GiB output allowance plus one GiB filesystem headroom is required; no
automatic deletions. Scientific underperformance must never fail a job or skip
an arm. Technical failures preserve diagnostics. Local tests cannot replace
the real Hydra setup, which gates automatic science release.

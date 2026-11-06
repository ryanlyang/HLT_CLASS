# Direct measured OFFLINE reconstruction: scientific v1, execution R2

Authority: [implementation plan](../plans/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_IMPLEMENTATION_PLAN.md).
Execution: [Hydra runbook](../LUKA_DIRECT_OFFLINE_RECONSTRUCTION_RUNBOOK.md).
Evidence: [local validation](../LUKA_DIRECT_OFFLINE_RECONSTRUCTION_VALIDATION.md).

## Scientific registration

The separate scope is `direct_reconstruction_only`, authorization token
`AUTHORIZE_DIRECT_RECONSTRUCTION_AUTO_V2`, and job prefix `lukanr_`. The existing
forty-row A/B/C registry and recipes retain their scientific identities.
All twelve R fits preserve the authenticated D000 HLT view, fresh native
initialization, batch128,30epochs, schedule, CE+U100 embedding weight5, and
native validation selection/export/replay. Reuse authenticated same-seed C0/C1;
do not substitute a different seed or retrain references.

| Row | Decoder memory | Full measured target | Extra consistency | Auxiliary weight | Seed |
|---|---|---|---|---|---|
| R1 | Global h | OFFLINE | No |0.1|0|
| R2 | h + block8 tokens | OFFLINE | No |0.1|0|
| R3 | Global h | OFFLINE | No |0.1|1|
| R4 | h + block8 tokens | OFFLINE | No |0.1|1|
| R5 | Global h | OFFLINE | No |0.03|0|
| R6 | h + block8 tokens | OFFLINE | No |0.03|0|
| R7 | Global h | OFFLINE | No |0.3|0|
| R8 | h + block8 tokens | OFFLINE | No |0.3|0|
| R9 | Global h | OFFLINE | Yes |0.1|0|
| R10 | h + block8 tokens | OFFLINE | Yes |0.1|0|
| R11 | Global h | HLT | No |0.1|0|
| R12 | h + block8 tokens | HLT | No |0.1|0|

Submission order is `R1 R2 R3 R4 R11 R12 R5 R6 R7 R8 R9 R10`. R11/R12 retain
U100-w5; they control the reconstruction target source, not all offline use.
New weights ramp linearly over three epochs. No reconstructed values are fed
back to classifier inputs, attention, particle tokens or pooled embedding.

## Complete targets and decoder

Reuse the existing raw TRAIN transfer; no OFFLINE validation or final-test
targets are admitted. Ordered identities/labels must join the same TRAIN view.
Every measured particle is retained. No HLT-to-OFFLINE particle matching is
assumed. The source raw contract currently rejects empty jets; the mathematical
loss also handles N=0 without dropping or changing any admitted source row.

The22 fields and masked source-specific FP64 TRAIN moments reuse the audited
S transforms: log pT and energy relative to the paired observable HLT vector pT,
relative eta, sin/cos relative phi, six-category and three-charge one-hots,
asinh d0/dz, log positive uncertainties, and four effective validity bits.
Continuous standard deviations have floor0.001; categorical/validity fields
are not standardized. Invalid regression fields are masked before arithmetic.
Frames and full raw p4/charged-energy aggregates remain FP64. Ragged normalized
FP32 values and packed validity live in memory-mapped files. Separate R banks
are prepared from existing raw files; C density fitting/targets are not needed.

K is the maximum TRAIN multiplicity across OFFLINE and HLT, with minimum1,
and is shared across every row. Every forward predicts K slots and0..K count
logits. Actual target counts, padding, fields, labels, identities, normalization
and frames are loss metadata only. None enters the decoder.

The decoder has K learned width256 queries, four pre-LayerNorm blocks,
eight heads, FF1024/GELU, zero dropout and no BatchNorm. Each block has self
attention, cross attention and identity-initialized global-h FiLM. Global memory
is one projected h token; token mode also uses projected live block8 tokens
with the effective native mask. No rank positions are added. Heads predict
three standardized kinematic fields, angle, category6, charge3, tracking4,
validity4, presence and global-only count. Angle predicts normalized sin/cos
through a scalar angle. Isolated CPU seed2026110601+1000003*seed initializes
all auxiliary parameters before optimizer construction, including K-dependent
ones. Query std0.02 and regression/validity output std0.001; category, charge
and count weights/biases are zero for exact uniform initial logits. Presence
weights are zero with bias log(0.1/0.9). Native and original U100 predictor
initialization/RNG are preserved.

Native inference stays on its original BF16 path. Decoder/loss are FP32;
physical inverse and aggregate sums are FP64. Independent predicted pT/E can
yield unphysical finite p4: record this, without hiding it behind a mass clamp.

## Exact loss and backward

Let `field=(kinematic+composition+track)/3`. Kinematic is mean SmoothL1(beta1)
over the five standardized coordinates; composition is the mean of category
and charge CE divided by log6/log3; track is half mean validity BCE plus half
mean valid-field SmoothL1, with zero regression when no tracking field is valid.

For N targets, K slots, `D=N+0.1*(K-N)`, positive cost `softplus(-presence)` and
negative cost `softplus(presence)`, solve the rectangular assignment with cost:

```text
field/N + (positive - 0.1*negative)/D
```

Canonical lexicographic target ordering makes ties reproducible without cost
perturbations. SciPy solves a detached CPU FP64 copy of the finite FP32 cost.
All N targets receive distinct slots. The matched field loss is recomputed
with gradients. No Chamfer, nearest-neighbor relaxation or target crop is used.

```text
Lreco = mean_matched_field + balanced_presence + 0.1*CE_count/log(K+1)
Ltotal = CE_HLT + 5*embedding_SmoothL1 + ramped_weight*mean_jets(Lreco)
```

R9/R10 additionally add0.1 times the mean of three consistency losses:
SmoothL1 on asinh total p4/q, asinh signed mass-squared/q², and charged-energy
fraction. All K slots contribute through soft presence, with charge probability
used for charged energy. Known HLT frames are applied only in the loss.
Consistency does not change assignment.

The native batch128 runs exactly once. Decoder-only chunks of eight jets
backpropagate into temporary h/T leaves and decoder parameters, scaled by
chunk/B and lambda. Their accumulated h/T VJPs are injected into the original
native graph together with CE+5*embedding. One optimizer update follows.
This preserves native BatchNorm and avoids retaining all decoder graphs.

## Artifacts and admission

Scientific v1 envelopes are
`LUKA_NEXT_SUPERVISION_DIRECT_RECONSTRUCTION_{RECIPE,INVENTORY,BANK,TARGETS}/v1`.
The execution envelope is `LUKA_NEXT_SUPERVISION_DIRECT_RECONSTRUCTION_SPEC/v2`,
with execution_version R2. It binds CPU compute/default/normal, one requested
CPU,4000MiB,60minutes, and `slurm_afterok_measured_independent_v1` release policy.
The former SPEC/v1 manual flow is not silently converted to automatic release;
use a fresh source-pinned R2 root. A/B/C remain on their existing manual workflow.
Each uses the established content hash, scope flags, parent hashes and immutable
publication. Banks include payload hashes, normalization, complete counts and
ordered TRAIN identity/label arrays. TARGETS binds common K and both banks.
Shared FIT/TRAINING_REPORT/EPOCH/TRAIN_PROBE/TASK artifacts bind the new spec;
fit parents also bind TARGETS. Shared RESOURCE_CHECK/ACCEPTANCE remain v2 and
include R target review, six-path coverage and R execution timing.

Registration computes storage from authenticated actual raw token counts,
two index/frame/aggregate banks, bounded native checkpoint/prediction/log
allowances for12science+12technical fits, plus2GiB contingency. User-declared
available quota and filesystem space must cover this budget plus1GiB headroom.
No full decoder checkpoint or full per-epoch reconstruction dump is saved.

Genuine setup must authenticate installed software/source, demonstrate exact
native/U100 initialization and FP32 forward parity, prepare all TRAIN targets,
and warm/measure six paths with full128, real partial batches, native/target
multiplicity tails and valid-track tails. K never changes between paths.
CPU assignment, decoder forward/backward, native forward/backward, probes,
data preparation and validation are included. Timing projection retains2x
full training/validation/probe cost plus actual per-fit loading and900seconds;
initial limits48hours and85% of one A6000/32000MiB host memory.

The user authorizes one bootstrap plan covering setup, durable CPU release,
the complete scientific recipe and all twelve resource-bounded fits. Bootstrap
submits setup plus release with `afterok:<exact setup ID>` and
`--kill-on-invalid-dep=yes`. Release authenticates its job, original live claim,
successful exact setup accounting, receipt hashes, source and measured limits.
It persists and scheduler-dry-reviews every measured science command before
submitting; no second user action is needed. A bounded accounting wait handles
Slurm publication lag. A failed setup cannot release science. Completed ledgers
are idempotent; ambiguous partial submissions fail without automatic retries.
CPU checks accept the observed one-to-two CPU rounding while authenticating
the one-CPU request, memory, task count, account, partition and GPU absence.

All twelve fits are independent: no scientific dependencies, holds, lanes,
arrays or priority changes. Only setup-to-release has a dependency. Poor finite
metrics never reject admission or skip a row. Invalid inputs, broken lineage,
nonfinite required quantities and measured resource excess fail with evidence.

## Reporting

Native validation AUC/recovery and matched same-seed deltas are primary. Running
epochs additionally show same-epoch versus selected C0/C1 deltas. Original50k
overlaps validation and is report-only; no independent-test claim is available.
First-TRAIN-batch probes record field groups, validity, presence precision/recall
at0.5, count/slot usage, h/T condition VJPs, aggregate errors and unphysical fractions.
The token norm is the direct decoder-memory path before native chain rule;
the h path also updates the encoder in both modes.
Zero and within-class/coarse-kinematics shuffles move h/T/mask together and
report actual swapped fractions. Shuffle comparisons exclude physical aggregate
loss and are explicitly labeled. Two deterministic first-batch aggregate
examples are saved per epoch. No validation OFFLINE inference is introduced.

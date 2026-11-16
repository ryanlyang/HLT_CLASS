# R24 reconstruction stability: 42-run Hydra campaign design

2026-11-16 operational addendum: the separate
[SOFT recovery and report audit](../LUKA_STABILITY_SOFT_RECOVERY.md) governs
numerical-v2 replacements for SOFT_s7/s8/s9 only. The original42 fits retain
the v1 design below. CE-only with EMA is a subsequent comparison, not a new
interpretation of the C128 controls.

Campaign specification, 2026-11-15. The user approved and requested implementation
of the seven candidates and42-fit allocation:14 training configurations at
three seeds each, on4 GTX1080 Ti GPUs per fit. The separate stability package
implements this matrix. See the [runbook](../LUKA_RECONSTRUCTION_STABILITY_HYDRA_RUNBOOK.md),
[v1 contracts](../contracts/LUKA_RECONSTRUCTION_STABILITY_HYDRA.md) and
[validation evidence](../LUKA_RECONSTRUCTION_STABILITY_HYDRA_VALIDATION.md).
These settings are experimental; they are not claimed empirically optimal.

The 42-run design below takes precedence over the earlier brainstorming
options later in this document for this proposed campaign. The existing
[repeatability plan](LUKA_RECONSTRUCTION_REPEAT_HYDRA_PLAN.md) and its
[v1 contracts](../contracts/LUKA_RECONSTRUCTION_REPEAT_HYDRA.md) continue to
govern their original executions. No running campaign is redefined.

The objective is a repeatable improvement over CE + U100 embedding weight5,
with native HLT-only ParT inference. Consistent 50-55%+ recovery is a research
target, not an established capability or an execution acceptance criterion.

## Campaign objective and exact allocation

Test whether specific changes to R24 training produce more repeatable HLT
classification gains over an appropriately matched CE + U100-w5 control.
Preserve the complete OFFLINE TRAIN particle target, native ParT architecture
and U100 embedding target. This tests seven mechanisms without combining all
of them into an uninterpretable new recipe.

Use three fresh seed indices **7, 8 and 9**, fixed before their outcomes are
observed. Every configuration uses all three. Task IDs are `<configuration>_s7`,
`_s8` and `_s9`. Reuse the original imported seed bases and index stride1000003;
record the resulting integer seeds and same-seed component hashes. Do not
share a trained warm-start between different seed indices.

| ID | Configuration | Effective batch | Primary same-seed control | Fits |
| --- | --- | ---: | --- | ---: |
| C128 | CE + U100-w5, original schedule | 128 | Self/reference | 3 |
| R24 | Original delay3/ramp3/maximum0.3 reconstruction | 128 | C128 | 3 |
| G30 | R24 with reconstruction encoder-gradient budget0.3 | 128 | C128 | 3 |
| C_L50 | U100-w5 with encoder LR transition to0.5x | 128 | Self/reference | 3 |
| R_L50 | R24 with the same encoder LR transition to0.5x | 128 | C_L50 | 3 |
| C_L25 | U100-w5 with encoder LR transition to0.25x | 128 | Self/reference | 3 |
| R_L25 | R24 with the same encoder LR transition to0.25x | 128 | C_L25 | 3 |
| PROJ | R24 with asymmetric reconstruction-gradient projection | 128 | C128 | 3 |
| DEC2 | R24 with two extra decoder-only steps during positions3-9 | 128 | C128 | 3 |
| C256 | U100-w5, two global128 microbatches per update | 256 | Self/reference | 3 |
| R256 | R24, two global128 microbatches per update | 256 | C256 | 3 |
| C512 | U100-w5, four global128 microbatches per update | 512 | Self/reference | 3 |
| R512 | R24, four global128 microbatches per update | 512 | C512 | 3 |
| SOFT | R24 with annealed soft particle assignment, then hard assignment | 128 | C128 | 3 |

Total: **5 control configurations + 9 reconstruction configurations =14;
14 x3 =42 scientific training fits**. There are15 control fits and27
reconstruction fits. Historical C0/R24 checkpoints are context, not substitutes
for these fresh controls. All42 fits include CE1 + U100 embedding weight5.

Candidate6, EMA, adds two predetermined averaged parameter streams to every
fit: half-lives0.5 and1 TRAIN epoch. It needs no additional training trajectory.
Those are correlated readouts, not extra seeds or42 additional experiments.
Candidate2 receives two LR settings and candidate5 receives two batch sizes;
candidates1,3,4,7 receive one setting each. Candidate6 gets two averaging
horizons. Combined gradient-budget/LR/projection arms and physical512 batches
are outside these42 and can be a later study.

## Recommended common setup

- Data: the existing authenticated299234 TRAIN /74895 development-validation
  jets, with identical HLT D000 views and per-jet identities in every arm.
- Model: the unchanged native HLT ParT and U100 predictor; reconstruction arms
  attach the existing R24 decoder conditioned on pooled HLT plus block8 tokens.
  Keep the full TRAIN-determined output support; never crop OFFLINE targets.
- Training:30 epochs from scratch, natural TRAIN permutation, FP32, TF32 off,
  SyncBatchNorm, existing AdamW betas/epsilon/weight-decay exclusions. Export
  only native HLT classification weights. No new teacher fits or OFFLINE
  validation targets are needed.
- Native schedule: original three-epoch warmup to3e-4 and cosine decay to1.5e-5
  at epoch30, except the explicitly specified encoder multipliers. No automatic
  LR scaling for larger batches, optimizer resets or extra training epochs.
- Reconstruction: zero encoder/decoder coefficient through position3;
  `w(t) = 0.3 * clip((t-3)/3, 0, 1)` thereafter. Preserve equal mixed-field
  weighting, presence/count terms, target normalization and original hard
  assignment except in SOFT. G30/PROJ alter only the native auxiliary gradient;
  DEC2 adds declared decoder steps. No hidden whole-jet consistency term.
- Hardware: one node,4 typed `gpu:geforce_gtx_1080_ti:4`, initially12 CPUs and
  64000MiB per scientific fit. CPU release/report use compute. Exact limits
  must be established by production measurements, including new readouts.

Use a separate fresh source/spec/output root. Within a seed, match native,
predictor and applicable decoder initialization, TRAIN permutation and native
dropout streams. Isolate extra decoder steps, probes and evaluation from
native-training RNG. Record separate RNG domains rather than allowing an
extra diagnostic forward to change later training dropout. This gives fair
comparisons within the new campaign; it does not promise bitwise replay of
the historical R24 trajectory.

Define the shared encoder parameter set once, by saved names: all trainable
native `mod.*` parameters except the final `mod.fc.*` classifier. It includes
the particle/class transformer and embedding/pair modules. Predictor and
decoder parameters are separate. LR groups cross this ownership partition
with the existing decay/no-decay partition. Refuse missing or duplicate
parameters. Preserve ordinary updates for classifier/predictor/decoder where
no intervention is declared.

## Concrete settings for this matrix

### G30 and PROJ: global parameter gradients

Use the candidate1/3 formulas below on globally sample-weighted gradients in
the same shared parameter coordinates. G30 has `rho=0.3`, `eps=1e-12`, and
norm-statistic EMA half-life0.1 TRAIN epoch. Initialize these two statistics at
the first active reconstruction update; advance them using examples processed.
Use the instantaneous cap as well as the smoothed ratio, never exceeding
`w(t)`. A zero base gradient gives zero auxiliary injection. The decoder still
uses `w(t)`; G30 does not quietly reduce decoder learning with the encoder cap.

PROJ protects the full CE+U100-w5 base gradient, leaves that base unchanged,
and performs no norm cap. Treat squared base norm at or below1e-20 as zero;
otherwise use the exact projection formula when the dot product is negative.
Reduce dot products/norm sums in FP64. Log norms, cap/projection frequency,
removed fractions and AdamW update norms. Neither arm claims to guarantee
non-increasing validation loss or a bounded AdamW parameter step.

### C_L50/R_L50 and C_L25/R_L25: lower encoder LR

For `m` equal to0.5 or0.25, multiply only the shared encoder learning rate by
`1 - (1-m) * clip(t-3, 0, 1)`. It is unchanged through position3, transitions
linearly during positions3-4 and remains at multiplier`m` afterward. This
multiplies the original time-dependent cosine LR, not a constant peak LR.
Apply the identical rule to each corresponding no-reconstruction control.
Keep AdamW moments and the other parameter groups' LR schedules intact.

### DEC2: two decoder catch-up steps

During `3 < t <= 9`, perform two decoder-only updates before the ordinary joint
update, using the same batch's detached HLT jet/tokens and complete OFFLINE
target. Their reconstruction coefficient is the current `w(t)` and their LR
is the original scheduled decoder LR. Before/after that interval, use ordinary
R24 updates. This arm changes the decoder update count, not the encoder delay.

Cache the one native training forward, then use fresh decoder graphs for each
extra update and for the final joint update. Recompute assignments each time.
Keep optimizer state owned separately by native/predictor versus decoder so
extra steps never advance native Adam moments or weight decay. Extra decoder
RNG is separate from native RNG. Prove that disabling the extra steps recovers
ordinary R24 updates, including optimizer moments and buffers. DEC2 is compared
with R24, not claimed to isolate every difference from historical D20.

### C256/R256 and C512/R512: accumulation

Keep each global microbatch128, split32 per rank, and SyncBatchNorm over that
microbatch. Accumulate respectively2 or4 such microbatches before a single
optimizer step. Both base and auxiliary gradients are weighted by their actual
jet counts over the full accumulation group. Zero gradients once per group,
hold all parameters fixed through the group, and flush the final partial group
at each epoch boundary. The30-epoch totals remain299234 examples per epoch.

For all arms define schedule position at an optimizer update as
`t = (epoch-1) + examples_consumed_this_epoch /299234`, using the group's ending
position. All its microbatches use that same scheduled coefficient/LR. The
epoch-end LR is exactly the registered epoch boundary, including the final
partial group. Apply this convention also to C128/R24 for internal consistency;
document its small difference from the old equal-step-fraction clock.

Per-epoch optimizer counts are2338 at128,1169 at256 and585 at512. Over30 epochs
these are70140,35070 and17550. Do not describe accumulation as an otherwise
identical AdamW trajectory: step frequency and cumulative decay change. The
matched controls measure the practical consequence. Physical256/512 forwards
are excluded from this matrix; there is no silent memory-dependent fallback.

### EMA05/EMA10: two averaging horizons on every fit

Maintain two training-only copies of native floating-point parameters. Start
each from the live model at the first optimizer boundary at/after position3.
After each later update use decay `2**(-delta_examples/(h*299234))`, with
`h=0.5` or`1.0`. This specifies the same averaging duration across batch sizes.
Before initialization the averaged stream is marked unavailable. Do not
average predictor/decoder weights or feed EMA predictions into training.

At every epoch, evaluate the live native model and both EMA models on the same
full validation population. EMA BatchNorm state is recalculated on the same
fixed8192 TRAIN HLT jets, chosen by an identity-hash ordering and stored in a
manifest before science. Disable dropout, use global128 calibration batches
and reset/update only BatchNorm running state on evaluation copies. Use the
same calibration procedure and ordering at every epoch and across seeds;
never read validation to estimate statistics or mutate the live student.

To distinguish averaging from BatchNorm recalibration, also evaluate a copy of
the current native weights with the identical TRAIN-only recalibration. Thus
there are **three parameter streams and four readouts per fit**: `RAW`,
`RAW_BN`, `EMA05`, `EMA10`. RAW remains the primary training-intervention
comparison; compare EMA against RAW_BN when attributing a benefit to averaging
itself. For privilege comparisons always match readout to readout in the
appropriate control. These extra evaluations are included in qualification,
walltime and storage accounting; they are not extra fits or independent seeds.

Select/store the best native checkpoint and probabilities separately for each
readout using the original selection tie rule. Do not silently choose the
largest of four AUCs as the reported primary result. Up to168 selected readout
records come from42 fits. Each deploys a single native HLT ParT, with no extra
inference heads or ensemble.

### SOFT: a fully specified initial loss proposal

Use the same `K` predicted slots and `N` real particles. Let `F_ij` be the
original equal-field pair loss, `p_i=softplus(-presence_i)`,
`n_i=softplus(presence_i)` and `D=N+0.1*(K-N)`. Define real-target costs
`C_ij=F_ij/N+p_i/D` and one unmatched-column cost `C_i0=0.1*n_i/D`.
The transport plan has row sums1, real-target column sums1 and unmatched-column
sum`K-N`. Omit the unmatched column when its mass is zero. For `N=0`, use the
original empty-target presence/count loss directly; do not divide by zero.
The original count term remains unchanged and outside assignment.

Use entropy-regularized minimum transport with reference masses
`Q_ij = column_mass_j/K`. Its scalar objective is the original linear cost plus
`tau * sum(P * log(P/Q))`, with zero-mass columns removed. At zero temperature
its optimum cost agrees with the original hard set loss, including presence;
ties need not produce the identical assignment. All particles retain unit
target mass. There is no independent row-wise softmax or target cropping.

Freeze one positive cost scale before science: choose512 fixed TRAIN jets by
identity-hash ordering with domain`stability-soft-calibration-v1` and use one
discarded technical initialization at seed2026111501, in evaluation mode.
Compute each nonempty jet's
mean absolute deviation of costs after subtracting each row's mean. Take the
median across jets, floored at1e-6. Store calibration membership, initialization,
scale and source hashes. No validation performance or trained scientific
checkpoint selects this scale.

For `3 < t < 9`, anneal temperature geometrically from`0.1*scale` to`0.01*scale`.
At/after position9 use the original exact hard loss. The R24 reconstruction
coefficient still ramps from0 to0.3 over positions3-6. Through position3 the
weighted reconstruction gradient is zero; retain the reference optimizer's
zero-gradient/weight-decay behavior. Temperature and coefficient have separate roles.

Solve in log space under no-grad and inject the converged envelope gradient
using the detached transport plan and differentiable costs; include the entropy
term in the scalar loss record. This is an approximation to the regularized
optimum gradient, not arbitrary stop-gradient through a different objective.
Initial numerical requirements: at most500 Sinkhorn iterations, maximum
absolute marginal residual1e-4, and absolute primal-dual gap divided by
`max(1, abs(primal_objective))` at most1e-4. Record both residuals and finite
objectives. Qualification must verify finite differences and tiny exact
examples before the recipe can be registered. If
these settings do not converge within resources, revise the numerical protocol
explicitly before submission; do not silently skip SOFT or fall back to hard
matching inside its soft phase. No validation-quality gate is allowed.

The numeric settings above are a concrete recommended starting proposal.
Solver accuracy/resource qualification is still required, so this document
alone is not evidence that SOFT or the full campaign is executable.

## Results, steadiness and decision rules

For every fit/readout save full epoch metrics and a selected checkpoint. Keep
the same validation AUC selection rule, with CE/rejection/earliest-update ties.
Historical recovery uses the unchanged M0HLT/OFFLINE anchors. Raw AUC and
matched-control delta are primary numerical evidence; percentages are a
rescaling of those metrics, not a separate source of statistical evidence.

Report by configuration and readout: all three seeds, selected AUC/recovery,
mean/minimum/sample standard deviation, three paired deltas, their mean/minimum
and number positive, and selected-checkpoint counts at50% and52%. Incomplete
groups retain explicit denominators; a missing control yields no paired delta.
For each alternative also report its paired change relative to ordinary R24.

Keep the existing fixed epochs21-30 mean, minimum, last-epoch recovery, threshold
counts and longest streak separately from checkpoint selection. Do not pick a
different late window per recipe or compare a late average to a control peak.
Track TRAIN CE/embedding/reconstruction losses, cap/projection/transport
statistics, gradient/update norms, optimizer counts, timing and memory so a
steady but worse classifier is not mistakenly called an improvement.

Recommend follow-up using mean matched gain first, then worst-seed gain and
the number of positive pairs, considering absolute recovery and late stability.
Publish the full table rather than choosing a favorable seed/readout. Three
seeds screen repeatability; any claim of consistent50-55%+ performance still
needs further independent evidence. The repeatedly used development validation
and overlapping original50k remain exploratory; final test stays sealed.

## Hydra setup, automatic release and queue order

Plan one4-GPU `setup` job, one durable CPU `release` job with `afterok:setup`,
42 independent4-GPU fits, and one CPU `report` with `afterany` on those42.
That is **45 Slurm jobs**, including42 scientific fits. Once authorized
bootstrap submission succeeds, setup completion releases the science without
an interactive session. Controls are comparisons, not fit-to-fit dependencies.
The report includes failures and missing results; finite poor AUC never skips
a registered fit or blocks another recipe.

Submit six reference fits first: C128_s7,R24_s7,C128_s8,R24_s8,C128_s9,R24_s9.
Then cycle through remaining seed groups7,8,9 in each of these passes:

```text
Pass A: G30, C_L25, R_L25, C256, R256, SOFT
Pass B: PROJ, C_L50, R_L50, C512, R512, DEC2
```

For example, all six Pass A configurations at seed7 are followed by the same
six at seed8 and seed9, then Pass B. This is6+18+18=42 submissions, with early
coverage of several mechanisms and adjacent relevant controls. It encourages
coverage but cannot force Slurm start order or concurrency. Every fit remains
independent after shared setup; no waiting for a control's accuracy or completion.

Setup must qualify the actual worker on four cards: native/U100 reference
updates and exports, globally reduced cap/projection versus a serial reference,
accumulation with partial groups and known normalization semantics, DEC2 state
isolation, transport masses/gradients/hard limit, EMA updates/BN isolation,
all three seeds' initial identities, and native-only reloads. Miniatures use
real TRAIN targets but their states are discarded before each fresh science
fit. Measure every distinct expensive path, all four validation readouts and
BN calibration, plus longest HLT/largest OFFLINE support and final partial
batches. Existing R24 throughput is context, not an accepted new resource bound.

Initial planning ceilings are6h for setup,48h per fit and1h per CPU job, with
12 CPUs/64000MiB for GPU tasks. These are proposed limits to qualify, not runtime
predictions. Request measured walltimes with the established2x safety factor
and explicit startup/reporting overhead. A legitimate resource failure retains
diagnostics for a revised spec rather than silently changing precision, batch,
number of seeds or target support.

Before implementation registration, inventory retained inputs and fresh quota.
Measure selected native/readout checkpoints, predictions, full training states
and setup artifacts; calculate a new42-fit output budget and reserve. Do not
reuse the old18-fit12GiB allowance or assume the previous quota declaration is
still available. Results/progress readers must not require space reserved for
future training outputs. Source, data, epoch and receipt authentication still
apply to reads.

Use exact pushed clean source, fresh versioned recipe/artifact identities,
immutable parent lineage, atomic publication, scheduler dry review and exact-ID
submission journals. Shared preparation must not consume scientific RNG or
retain a warm model. Any failed or partially accepted submission requires
inspection of the accepted IDs; there is no broad retry/cancellation command.

## Implementation and qualification boundary

The implementation uses a separate package and v1 identities. Original
scientific/checkpoint sources remain immutable. Full training is not submitted
by implementing the code. Commit/push, new-root registration and scheduler
review precede the two-job bootstrap in the runbook. Its genuine four-card
setup must pass before the durable CPU release submits any scientific fit.

Technical implementation choices within this design:

- DEC2 uses disjoint named ownership groups in AdamW. During decoder-only
  steps, all native/predictor gradients are None, so their parameters, decay,
  counters and moments do not advance. No new optimizer reset is introduced.
- SOFT solves in FP64 log space while trainable tensors/native execution remain
  FP32. The fixed technical native initialization uses2026111501; the existing
  decoder's isolated initialization derives its seed from that technical seed
  index. Both seed integers and the full initialization digest are recorded.
  The solver starts with50 ordinary log-scaling iterations, then uses
  log-potential relaxation1.8 if needed, within the same500-iteration cap.
  This addresses slow null-mass convergence without changing the objective,
  temperature, marginals or either acceptance tolerance.
- Five common-start transition comparisons per configuration give70 numerical
  checks. Complete reference states are broadcast in memory. Saved hashes and
  named comparisons preserve qualification evidence without70 large redundant
  tensor dumps. Scientific checkpoints/predictions remain retained.
- Resources measure active positions4.5/8.99 and hard position20 as applicable.
  DEC2/SOFT projections charge six passes at the measured expensive phase and
  24 at the ordinary hard phase, then apply2x safety. Four validation readouts
  and TRAIN BN recalibration are included for every epoch.
- Registration serializes model states to inventory output needs. The current
  298-slot local estimate is29GiB plus1GiB reserve; installed serialization and
  actual setup measurements are authoritative. No old quota declaration is
  reused. Read-only results/progress do not demand future-write space.

No scientific target, matrix row, seed, batch normalization population or
selection rule is changed by these technical choices. Real Hydra qualification
remains distinct from local unit/distributed/synthetic-CUDA validation.

## Candidate rationale and earlier alternatives

The sections below preserve the reasoning behind the seven candidates. Where
an earlier option differs from the concrete matrix above, it is future work
rather than an extra uncounted fit in this campaign.

## Evidence motivating this direction

These are user-pasted completed-fit summaries, not a new local authentication
of Brown checkpoints. The repeat campaign is
`reconstruction_repeat_15c35c42_4x1080ti_r1`. Its five reconstruction fits at
seed6 had no completion receipts in the latest pasted snapshot.

| Recipe | Seed4 selected recovery | Seed5 selected recovery | Mean matched AUC gain, these two seeds |
| --- | ---: | ---: | ---: |
| CE + U100-w5 control | 36.8% | 42.3% | 0 |
| R24: delay3, ramp3, weight0.3 | 47.3% | 38.9% | +0.00014072 |
| R20: weight2, no delay | 33.1% | 34.1% | -0.00023737 |
| R22: weight10, no delay | 50.0% | 39.9% | +0.00021247 |
| R28: block4+8 conditioning | 37.4% | 39.4% | -0.00004772 |
| D20: decoder warmup, longer cosine ramp | 51.1% | 39.0% | +0.00021637 |

Every reconstruction recipe lost to its seed5 control. All ten completed
reconstruction fits had zero epochs at or above52% within epochs21-30. Several
seed4 selected checkpoints improved substantially, but late performance often
deteriorated. The controls deteriorated too: late mean recovery was -82.9%,
-116.3% and -108.2% for seeds4,5,6. Negative recovery means AUC below the
historical HLT anchor; it does not mean negative AUC.

The completed diagnostic snapshots also showed:

- P0_R41.4% versus P0_C39.4%; changing only the sampler seed gave PO_R54.0%
  versus PO_C39.1%. Four GTX1080 Ti GPUs can therefore reach that observed
  level; hardware alone does not explain the low repeats.
- Changing the dropout seed gave PD_R45.4% versus PD_C37.5%.
- Changing native initialization gave PI_R35.2%, but PI_C failed. That pair
  cannot establish whether reconstruction helped the changed initialization.
- All six available diagnostic pairs had positive selected-AUC deltas, some
  very small. The new repeats show that benefit is not universal across seeds.

These patterns motivate optimization hypotheses; they do not identify a cause.
Overfitting, optimizer dynamics and normalization state remain possible
contributors. A historical best55.4% checkpoint is not the expected outcome
for an arbitrary new seed. Original50k overlaps validation and is not an
independent replication.

## Current mechanism and common comparison

The reference is R24 with the complete OFFLINE particle-set target and its
existing exact assignment and mixed-field reconstruction loss. The base loss
is `L_base = CE + 5 * SmoothL1(predicted_U100, target_U100)`.

The current worker separately computes decoder gradients and reconstruction
gradients entering the pooled jet representation and block8 particle tokens.
It adds their effects to the native model's base gradient before AdamW. There
is no adaptive gradient cap or conflict projection in this reconstruction
worker. Its shared optimizer applies the same scheduled learning rate to all
parameter groups.

The native learning rate warms to3e-4 at epoch-position3 and then decays by a
cosine schedule to1.5e-5 at position30. R24's reconstruction coefficient is zero
through position3 and rises linearly to0.3 at position6. Reconstruction thus
starts near the learning-rate maximum. This is a plausible sensitive
transition, not proof of instability.

Unless a candidate explicitly changes it, keep the authenticated299234 TRAIN /
74895 validation population, all OFFLINE TRAIN particles, global batch128,
30 epochs, U100-w5, fixed field normalization, native architecture and original
checkpoint selection rule. Preferred execution remains4 GTX1080 Ti, FP32 and
SyncBatchNorm. All new extra state is training-only; export one native HLT ParT.
The matrix above fixes recommended starting values; no empirical optimality is
claimed. Other values mentioned in the rationale are alternatives outside it.

**Current batch size:**128 jets total per optimizer update, split across the
four GPUs as32 jets per GPU for a full batch. There is currently no gradient
accumulation. The last batch can be smaller. Four GPUs do not make the current
batch512: they divide the registered global128 batch.

## Candidate 1: Bound reconstruction's encoder gradient

**Question:** does a fixed loss coefficient produce unpredictable relative
influence across batches, epochs and seeds?

Compute `g_base` and the unweighted `g_rec` over the same shared native
parameter coordinates, excluding the final classifier and training-only heads.
Use separate decoder gradients as today. Start with an encoder budget
`rho = 0.3`: the reconstruction gradient norm may be at most30% of the base
gradient norm. This is a proposed budget, not a guarantee about AdamW's actual
parameter displacement.

For a scheduled maximum `w(t)` matching R24, one concrete rule is:

```text
alpha = min(w(t), rho * EMA(norm(g_base)) / (EMA(norm(g_rec)) + eps),
                  rho * norm(g_base) / (norm(g_rec) + eps))
g_shared = g_base + alpha * g_rec
```

The smoothed ratio reduces coefficient jitter; the instantaneous ratio
enforces the cap. Do not amplify above `w(t)` to compensate for a weak decoder.
If the base gradient is zero, inject zero auxiliary gradient and record it.
The decoder retains its declared R24 coefficient independently.

Apply the calculation to sample-weighted, globally reduced gradients across
all four ranks. Independent rank-local caps define a different algorithm.
Normalize neither gradients across unrelated coordinate systems nor padded
particle positions. A cheaper cap on representation gradients would be a
separate approximation, not an encoder-parameter norm guarantee.

Log raw/capped ratios, cap frequency, per-block norms and actual native update
norms. AdamW moments and weight decay mean the gradient cap is not a strict
trust region. Extra gradient extraction must be included in memory/runtime
qualification. Compare with unmodified R24 and the same-seed U100-w5 control.

This is the first priority. It is inspired by gradient-magnitude balancing,
not an implementation of the complete [GradNorm algorithm](https://proceedings.mlr.press/v80/chen18a.html).

## Candidate 2: A gentler encoder learning-rate transition

**Question:** is the encoder changing too quickly when reconstruction begins?

Keep the R24 delay/ramp and decoder learning rate. Reduce only the shared
encoder learning-rate multiplier from1 to0.5 or0.25 over positions3-4, then
retain that multiplier on the remaining original schedule. Preserve AdamW
moments and weight-decay groups; do not introduce an optimizer reset in the
same first comparison. Classifier, U100 predictor and decoder retain their
explicitly recorded original schedules.

This allows the decoder to adapt while making smaller changes to the native
representation. It might also undertrain the encoder, so a smoother trajectory
alone is not success. Measure selected AUC and the matched-control gain.

Required controls are U100-w5 at both the original and revised encoder learning
rates, plus ordinary R24. A lower-LR reconstruction fit beating the old control
does not isolate a privilege benefit. Candidate1 plus2 is a sensible combined
arm after the individual effects are measurable.

Decoder warmup already exists in D20 and showed mixed repeats. This proposal
adds an encoder learning-rate change; simply repeating warmup is not new.

## Candidate 3: Remove direct gradient opposition

**Question:** do reconstruction updates repeatedly oppose the existing
CE + U100-w5 objective?

On the same shared parameter coordinates as candidate1, if
`dot(g_rec, g_base) < 0` and the base norm is nonzero, use:

```text
g_rec_projected = g_rec - dot(g_rec, g_base) / norm(g_base)**2 * g_base
g_shared = g_base + w(t) * g_rec_projected
```

Leave `g_base` and decoder learning intact. Compute global dot products after
rank reduction, handle zero gradients explicitly, and log projection frequency,
pre/post dot products and the removed gradient fraction. If combined with
candidate1, project first and cap the remaining auxiliary gradient second.

This only removes opposition in a local, raw-gradient sense. It does not
guarantee improvement after AdamW or on validation. CE and U100 can also
disagree; protecting their sum does not separately protect CE. Protecting CE
alone would be a distinct arm, not a silent change.

The earlier [embedding-objective campaign](LUKA_EMBEDDING_OBJECTIVE_HYDRA_PLAN.md)
already implements CE protection for embedding gradients in its R7/R8. The
different question here is whether protecting CE+U100 from the reconstruction
gradient helps. Do not claim the technique itself is new, or infer the earlier
campaign's outcome without its results. Method context:
[Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

## Candidate 4: Let the decoder catch up before moving the encoder

**New question:** does an under-adapted decoder send a poor learning signal
into an otherwise useful HLT representation?

Give the decoder two extra updates on detached, cached HLT representations
before the usual joint update, initially during positions3-9 only. The main
HLT forward and its CE/U100 gradient still happen once per native update.
Recompute the reconstruction gradient using the updated decoder before
injecting it into the encoder; do not reuse a stale gradient.

Use explicit optimizer ownership: decoder-only updates advance only decoder
state, and the shared encoder/classifier/predictor advance once. Track native
and decoder update counts separately and schedule by native data exposure.
The detached updates must not change backbone BatchNorm buffers, consume
backbone dropout draws or add CE/U100 optimizer steps.

This goes beyond D20's one decoder update per batch during its warm period:
it changes relative optimization rates while encoder training continues.
Compare zero versus two extra decoder updates, with every other schedule held
fixed. Include a D20 reference if the purpose is specifically to improve on
decoder warmup. Cost can be substantial because matching and decoding repeat.

A decoder that fits faster can also overfit or produce stronger harmful
gradients. Better reconstruction error alone is not evidence of better
classification. Candidate1 is a plausible later partner, not a reason to
combine both in the first isolating comparison.

## Candidate 5: Test a global optimizer batch of512

**New question:** does the strong sampler sensitivity partly reflect noisy
updates that depend too heavily on the current small set of jets?

The campaign comparison is128 versus256 versus512 jets per optimizer update, for both
the base and reconstruction objectives. Distinguish two execution choices:

| Execution | Jets per GPU per forward, four GPUs | Forwards per update | Jets per optimizer update | SyncBatchNorm population per forward |
| --- | ---: | ---: | ---: | ---: |
| Current reference | 32 | 1 | 128 | 128 |
| Accumulated256 | 32 | 2 | 256 | 128 |
| Accumulated512 | 32 | 4 | 512 | 128 |
| Physical512, if qualified | 128 | 1 | 512 | 512 |

The preferred first implementation is four global128 microbatches accumulated
into one effective512 update, keeping the existing per-forward memory and
normalization population. Physical512 is a distinct future option requiring measured
memory qualification on four GTX1080 Ti cards; it is not interchangeable with
accumulated512 because the normalization population differs. Any physical512
comparison needs its own U100-w5 control with the same normalization behavior.

For accumulated512, preserve each
microbatch's global SyncBatchNorm computation and original per-jet loss
normalization. Normalize partial groups by their actual jet count. Hold model
parameters fixed through accumulation and take one AdamW step afterward.

This is gradient accumulation, not four additional native steps and not
BatchNorm over512 simultaneous examples. Keep the same30 passes through TRAIN
and express the learning-rate schedule in examples seen. Do not automatically
multiply the learning rate by four.

With299234 TRAIN jets and an update at each epoch's final partial group, this
changes2338 optimizer updates per epoch to585. The30-epoch comparison keeps
the number of TRAIN examples presented fixed, not the number of AdamW steps.
All four GPU contributions and all accumulation microbatches must be included
in any candidate1/3 gradient cap or projection applied at the optimizer step.

Mandatory comparisons are accumulated U100-w5 and accumulated R24, against
their global128 versions. Accumulation changes optimizer-step frequency and
AdamW/weight-decay dynamics as well as stochastic noise. Record this and the
number of updates; do not describe it as a pure batch-order intervention or
as preserving the original optimizer trajectory.

This can reduce update variance but does not promise better generalization.
It also does not solve long-range ordering effects or a systematically bad
reconstruction objective. It is a medium-priority candidate after1/2.

## Candidate 6: Evaluate a moving average of the native model

**New question:** are useful representations transient, so that smoothing
nearby parameter states gives more dependable deployable checkpoints?

Maintain an exponential moving average (EMA) of native ParT parameters after
each optimizer step, with a predeclared start and averaging horizon. A first
proposal is to start at position3 with a half-life of0.5 TRAIN epochs, converted
to an update decay from the examples processed. Initialize from the current
native model at that point. It does not feed back into the training targets.

Evaluate ordinary and EMA native models as distinct, predeclared readouts.
Keep one selected checkpoint per readout; do not choose whichever looks best
and present it as a single originally specified method. Apply the same EMA
readout to U100-w5 controls. A primary EMA analysis uses EMA controls too.

Specify BatchNorm handling separately from weight averaging: recalculate its
statistics on a fixed TRAIN-only HLT subset with dropout disabled before EMA
evaluation, without mutating the live student. Never fit those statistics on
validation. Account for this extra forward work and store the resulting native
buffers. Export one native model; inference has no ensemble or extra heads.

EMA can smooth short-lived variation but cannot create an unvisited good
solution or reliably undo sustained degradation. The extreme late declines
make this a supporting intervention, not the main explanation. It can be
collected alongside new fits relatively cheaply, but must be prospectively
declared. Context: [EMA dynamics and benefits](https://arxiv.org/abs/2411.18704).

## Candidate 7: Smooth the particle assignment during early reconstruction

**New, more experimental question:** do abrupt changes in which predicted slot
matches each OFFLINE particle make the encoder supervision unstable?

The current loss uses an exact rectangular linear assignment. Close competing
assignments can switch after a small prediction change. This is a hypothesis
about gradient variance, not evidence that the existing solver is wrong.

A possible alternative uses entropy-regularized soft assignment early, then
anneals toward the original hard matching. Preserve all OFFLINE particles,
their unit target mass, the fixed predicted-slot capacity and an explicit
unmatched-slot mass of `K-N`. Presence/no-object loss must use the same soft
occupancy. Independent row-wise softmax matching is insufficient: it can send
many slots to the same particle and leave other targets unrepresented.

This needs a separately defined loss, including whether gradients pass through
the transport plan or use the envelope gradient of the regularized objective.
Do not silently detach a soft plan while describing a different differentiable
objective. Handle empty jets, full occupancy, categorical fields, masks and
normalization, and establish the small-temperature relationship to the hard
loss. Temperature values and annealing require TRAIN-only calibration of the
cost scale before registration.

Measure assignment turnover on a fixed TRAIN probe with stochastic layers
disabled, as well as loss/gradient variance and student AUC. Soft assignments
can blur physically different particles and harm the task. This changes
scientific loss semantics and needs its own version; this campaign gives it
one exploratory configuration alongside the simpler interventions. Method context:
[entropy-regularized transport](https://proceedings.neurips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html).

## Proposed priorities and fair evaluation

| Priority | Comparison | Main question |
| --- | --- | --- |
| First | R24, candidate1 and candidate2 | Is the encoder receiving too large a reconstruction-driven change? |
| Next | Candidate3; candidate4 | Does gradient opposition or a lagging decoder explain weak seeds? |
| Supporting | Candidate5; candidate6 | Do batch averaging or parameter averaging improve repeatability? |
| Exploratory | Candidate7 | Is hard-assignment switching an important source of variance? |

The42-fit matrix above implements this allocation; combinations such as1+2
remain future work. Finish reviewing seed6 and the existing per-epoch TRAIN
losses, base/auxiliary gradients and normalization behavior to interpret the
earlier study. Do not use those results to choose favorable new seed identities.
Do not conclude that learning-rate decay will fix the late decline: the
existing schedule already decays. Early stopping could save runtime but would
not explain or repair differences in selected recovery across seeds.

For a new screen, use the same predeclared seed set for every compared arm and
at least three paired seeds. Choose new seed identities before seeing their
outcomes. Seeds4/5/6 may be reused for an explicitly labeled development
comparison, but cannot serve as unseen confirmation after motivating this
design. Use no validation-based choice of lucky seeds, activation times,
averaging windows or per-seed hyperparameters.

Always include ordinary R24 and CE+U100-w5, plus the matched altered-LR,
accumulation or EMA controls where applicable. Report:

1. Selected AUC/recovery and delta against the appropriate same-seed control.
2. Mean, minimum, spread and individual seed values; number of positive pairs.
3. Complete seed denominators and selected-checkpoint threshold counts.
4. Fixed late-window recovery/threshold counts and full epoch curves, separately
   from selected results. Do not compare a late average with a control's peak.
5. Gradient/update statistics appropriate to each hypothesis and extra compute.

Three seeds are a screening experiment, not a reliable population estimate of
tail failure risk. The repeatedly consulted validation remains exploratory.
An improvement must ultimately survive fresh replication/evaluation before
being described as consistently exceeding50-55%.

## Implementation references and boundaries

- [Repeat kernel](../../src/hlt_classification/luka_reconstruction_repeat_hydra/kernel.py):
  native/base gradients, decoder VJP injection, reduction and AdamW update.
- [Current schedules](../../src/hlt_classification/luka_reconstruction_repeat_hydra/contracts.py)
  and [native learning rate](../../src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py).
- [Exact reconstruction assignment/loss](../../src/hlt_classification/luka_next_supervision_hydra/reconstruction_batched.py).
- [Training/export](../../src/hlt_classification/luka_reconstruction_repeat_hydra/training.py)
  and [repeat reporting](../../src/hlt_classification/luka_reconstruction_repeat_hydra/reporting.py).
- [Earlier R24 curriculum](LUKA_R24_CURRICULUM_HYDRA_PLAN.md) for delays, ramps,
  fades and decoder preparation already considered.

No source file is migrated by this document. Scientific reference commits are
58d0c0ed (original reconstruction), a549fef1 (curriculum), 61f84e02 (corrected
diagnostics) and15c35c42 (three-seed repeats). A later implementation needs new
recipe identities/contracts for changed gradients, optimizer schedules or
matching, immutable fresh output roots, same-seed initialization checks and
numerical qualification of distributed semantics. Old artifacts are not
reinterpreted. Shared pretraining, if introduced later, must be separately
declared and must not accidentally reduce the number of independent seeds.

Maintain HLT-only deployment, OFFLINE TRAIN supervision only, sealed final-test
access, measured four-card resources and no scientific-quality admission gate.
This edit documents the approved42-fit design without implementing or submitting
it. The next task is a separate runtime/contract implementation and qualification
of these settings, with setup plus durable automatic release as specified above.

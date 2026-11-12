# Embedding assistants, teacher responses, and regional targets on Hydra

Status: scientific recipe v1; execution repair v5, 2026-11-11.

Setup164517 passed the v4 numerical qualification, C0, LS_D050 and TR
miniatures, then exhausted both GTX1080Ti cards in TP's crop encoder during
epoch2. Execution v5 checkpoints each TP full-jet/crop native encoder and its
stateless transformer blocks using non-reentrant recomputation. The nested
block checkpoints also bound the peak of a single encoder backward. Block
implementations remain installed Weaver calls; mutable block buffers are rejected.
Preserve dropout RNG and the entire batch; during
backward recomputation replace registered buffers with disposable clones and
restore their original references, so BatchNorm/SyncBatchNorm running moments
and counters update only on the original forward. Evaluation remains native.
No input, crop, objective, optimizer, precision, batch or teacher selection
changes. SPEC/v5 binds TP_EXECUTION_POLICY/v1; RESOURCES/ACCEPTANCE advance to v2.

Immediately after authenticated preparation, GPU setup compares ordinary and
checkpointed TP updates on both ranks with dropout enabled, across all six
source/scale conditions. It checks loss, gradients, parameters, optimizer
moments, normalization buffers and RNG. Next it exercises the real longest
TRAIN supports at global batch128 and uneven batches11/98, all six conditions,
and checks the existing host/GPU memory bounds. These checks precede U100
numerical replay, other miniature fits, exports and full validation. Their
TP_CHECKPOINT/v1 and TP_STRESS/v1 records are required by resource admission.
The existing full qualification remains required; local tests cannot certify
two-card execution. Reuse the original completed92397736 preparation again;
no new export or particle reconstruction is needed.

Diagnostic164226 reproduced the historical-bank difference on both CPU and
GTX1080Ti: standardized RMS6.25e-5, while CPU/GPU RMS was5.16e-7 and GPU default
versus math attention was identical on610 rows. Setup163911's4096-row maximum
was0.00145745 and RMS5.68152e-5. Scales range0.14857 to1.14471; none is floored.
This supports bounded historical-export variation, without identifying its
exact original hardware/library setting or measuring its effect on recovery.

Execution v4 explicitly supersedes v3's elementwise1e-4 admission rule with
U100_NUMERICAL_POLICY/v1: historical raw AND standardized maximum <=0.002 and
RMS <=0.0001; current CPU/GPU and GPU default/math standardized maximum <=1e-5
and RMS <=2e-6. Both the original4096-row witness and all three paths on four
evenly spaced original batches plus the tail must satisfy the historical bounds.
These are fixed qualification limits, not automatically tuned to each run.
The standardized maximum is0.2% of one TRAIN standard deviation; RMS is0.01%.
The raw maximum also covers the known full-witness standardized maximum times
the largest TRAIN scale (<0.00167), rather than assuming the610-row raw maximum
bounds all4096 rows. Independent local parity stays substantially tighter.
The former elementwise decision remains recorded as a diagnostic. Source,
teacher, population and input/target byte hashes remain exact. The original
TRAIN bank, model, loss, optimizer, scientific recipe and GPU profile are fixed.
SPEC/v4 binds the policy; U100_REPLAY/v3 records all comparisons and decisions.

Setup162131 completed authenticated preparation in8816.77s, then failed the
frozen U100 reproduction witness (maximum standardized difference0.00145674).
The historical exporter uses consecutive batches128 on one card. The v2
witness regrouped random panel rows and split each batch across two ranks.
Execution v3 preserves each original batch and padding context, assigning
whole inference batches to ranks; training still splits global batch128.
Setup163911 showed that preserving batches did not resolve the mismatch.

Fresh setup checks the frozen teacher before constructing quarter views.
For the completed92397736 preparation, --resume-spec authenticates its exact
terminal setup, source, manifests and array hashes, then hard-links immutable
payloads into a fresh root with new manifests. No scientific fit or old GPU
qualification is reused. Source logs/data remain intact; no new SPORC export
or full particle reconstruction is needed for that recovery.
PREPARATION_REUSE/v1 records lineage and remains unchanged.
The unchanged TRAIN targets and scientific RECIPE/v1 remain authoritative.
Real two-GPU qualification is still pending; a remaining reproduction failure
must stop with diagnostics; bounds never increase automatically.

**Repaired implementation awaits real Hydra qualification.** Setup161306 failed
while replaying historical inputs, before training. D000's first differing
value was one float32 ULP in relative eta; all five historical TRAIN views
had matching population/size summaries but different input hashes. This does
not establish the size of every discrepancy or its exact platform cause.

The repair preserves exact historical tensors instead of weakening comparison:
D000 and TRAIN U100 come from the original Brown bundle; OFFLINE comes from
the existing teacher-sweep transfer; a minimal native delta supplies TRAIN
D033/D066 and validation D033/D066/U100. Exporting that delta may reconstruct
missing SPORC caches from retained raw data and original assignments, but must
reproduce the original file bytes and whole-population fingerprints. Only
the new D025/D050/D075 views are encoded from physical rows on Brown.
Physical TRANSFER/v1 and RECIPE/v1 stay unchanged; SPEC/v2 binds the additional
NATIVE_TRANSFER/v1 hash. A fresh execution root is required. No failed partial
preparation is accepted as a completed artifact. Existing crop construction,
losses, models, teacher targets and all 30 recipes remain unchanged.

The user requested three ladder
sizes, ten response configurations, ten regional configurations, and the
existing two-older-GPU Hydra execution style. The user selected the
299,234 TRAIN / 74,895 development-validation screen first, followed by
separate FullStat confirmation of promising methods.

This document fixes the recipes and implementation requirements. The
[runbook](../LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md) provides export and queue
commands; the [validation record](../LUKA_EMBEDDING_BRIDGE_HYDRA_VALIDATION.md)
separates local evidence from remote checks. The
[versioned contract](../contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md) lists the
implemented artifacts.
Defaults express research judgment, not measured optima or a promise of
60%/70% recovery. It authorizes no submission by itself and changes no existing
campaign, source snapshot, dataset, or queued job.

## 1. Research question and intended contribution

Can the successful HLT U100 embedding-prediction objective benefit from:

1. **Embedding assistants:** models on intermediate views that predict the
   original U100 embedding, passing their predictions to the next rung.
2. **Teacher responses:** predicting how one shared teacher's representation
   changes when selected information is improved.
3. **Regional embeddings:** predicting representations of corresponding
   coarse and fine regions of the OFFLINE jet.

All final students remain native HLT-only ParT classifiers. The additions
are training-only prediction heads. No teacher, predicted embedding, region
descriptor, or privileged field enters the deployed classifier.

The motivation is the user's observation that embedding prediction has held
up better than logit KD at higher statistics, and that intermediate-view
ladders can help transfer. This screen tests mechanisms consistent with those
observations. It cannot establish their cause, an information-theoretic ceiling,
or their FullStat benefit.

Existing multi-teacher, relational, reconstruction, and particle-target methods
are not relabelled as new experiments. In particular:

- [The earlier embedding ladder](LUKA_EMBEDDING_LADDER_PLAN.md) propagated logits
  and retained a fixed OFFLINE embedding objective.
- [The FullStat relational ladder](LUKA_FULLSTAT_RELATIONAL_LADDER_PLAN.md)
  exported native intermediate embeddings and explicitly prohibited exporting
  auxiliary predictions. This NEW campaign intentionally exports assistant
  predictions, under new contracts; old contracts remain valid for old runs.
- [Selective teachers](LUKA_SELECTIVE_TEACHER_HYDRA_PLAN.md) were independently
  trained. The response teacher below is one shared model across conditions.
- [Particle prediction](LUKA_PARTICLE_TEACHER_HYDRA_PLAN.md) targeted individual
  aligned particle representations. Regional targets below come from separately
  encoded crops containing potentially different HLT/OFFLINE particles.

A useful result must improve native classification against a fresh matched
control, not just reduce an auxiliary loss.

## 2. Exact scientific budget and task registry

There are **30 scientific fits**, including **22 final HLT classifiers**.
U100 is reused. R0 and P0 are display aliases for one fresh C0, not two fits.

| Category | Task IDs | Fits |
| --- | --- | ---: |
| Short ladder assistant and HLT endpoint | LS_D050, LS_D000 | 2 |
| Medium ladder assistants and HLT endpoint | LM_D066, LM_D033, LM_D000 | 3 |
| Long ladder assistants and HLT endpoint | LL_D075, LL_D050, LL_D025, LL_D000 | 4 |
| Shared response teacher | TR | 1 |
| Shared regional teacher | TP | 1 |
| Response students | R1-R9 | 9 |
| Regional students | P1-P9 | 9 |
| Fresh common HLT control | C0, displayed also as R0/P0 | 1 |
| Total | | 30 |

Counts by role: six privileged assistants, two target teachers, three ladder
HLT endpoints, eighteen R/P HLT students, and one HLT control.

The short and long D050 fits are distinct: their predecessor targets differ.
One scientific fit means 30 full epochs of one registered model. Target exports,
qualification, launch/release, and reporting are separately accounted operational
work; they must not be hidden as extra scientific training.

All final HLT fits use seed0 with the inherited paired initialization and
sampler seeds. This is a breadth-first single-seed screen. Seed replication and
FullStat confirmation are subsequent experiments, not silently added to the
30 fits. The literal inherited seeds are copied into the immutable spec;
family names must not change native initialization or sample order.

## 3. Common student objective and optimization

Let h(x) be the native 128-dimensional HLT pre-classifier embedding.
The native 11-class head receives only h. Let z_U be the authenticated,
TRAIN-standardized, frozen U100 target, and g_U the original embedding predictor.

Define S(a,b) as SmoothL1(beta=1), averaged over the 128 coordinates and then
over jets. The baseline is:

    L_base = CE(native_logits, true_class) + 5 * S(g_U(h), z_U)

Every final HLT row retains this exact baseline. New terms are additive:

    L_student = L_base + lambda * r(u) * L_new
    r(u) = min(1, u / (3 * ceil(299234 / 128)))

Here u is the one-based optimizer-update index. The ramp reaches its maximum
at the end of epoch 3. Original U100 weight5 is active from the first update.
Ladder endpoints use lambda=5; R/P coefficients are in their tables below.

| Setting | Planned value |
| --- | --- |
| TRAIN / development validation | 299,234 / 74,895 authenticated original rows |
| Epochs / global batch | 30 complete epochs / 128, retaining the partial batch |
| Native model | Original native17-input, width128, eight particle blocks, two class-attention blocks, 11 classes |
| Optimizer | Existing U100 AdamW groups, betas, epsilon and weight decay |
| LR | Existing warmup3/cosine30: peak 3e-4, final 1.5e-5 |
| Precision | FP32 on two GTX 1080 Ti GPUs, TF32/autocast disabled |
| Distributed normalization | DDP with SyncBatchNorm |
| Student checkpoint selection | Existing full-validation macro-OVR AUC selection and deterministic tie rules |
| Main inference | Ordinary native HLT ParT, with auxiliary heads removed |

Import the registered optimizer and selection functions, rather than
reconstructing an approximately equivalent recipe. Changes to those functions
require versioning and a new matched control.

The inspected native selection key maximizes validation macro-OVR AUC, then
minimizes CE, then maximizes mean log QCD rejection at 50% signal efficiency
(missing rejection ranks last), then prefers the earlier update. Preserve
that ordering for C0 and all final HLT students.

The new screen does not also add logit KD, CONCAT/FUSION regression, subclasses,
contrastive matching, gradient surgery, or retained predictors. Those are
separate objectives. Successful additions can later be tested on the strongest
verified FullStat baseline.

Auxiliary initialization uses isolated RNG streams. Adding an unused head must
not change native weights, g_U initialization, sample ordering, or training
dropout streams. Zero-weight tests must recover the fresh FP32 control update.
All final models share the same deployed parameter count.

Do not interpret equal numeric coefficients as equal encoder-gradient strength.
Record raw loss, weighted loss, and occasional gradient diagnostics.

## 4. Data, views, and information boundaries

### 4.1 Authentication and reuse

Authenticate the original small-data preparation, native D000 inputs, selected
regular COARSE_U100 checkpoint/bank, normalization, identity order and labels.
Do not substitute a FullStat, fusion, newer U100, or pure-OFFLINE teacher.

The reference implementation inspected for this design is repository commit
45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08. This is a donor reference, not the
future execution commit; execution must pin its own clean pushed source.

Input availability on Brown is not established by local documentation.
Registration must inventory:

- D000 and U100 TRAIN/validation feature caches;
- original physical HLT/OFFLINE particles and original assignment manifests;
- existing D066/D033 caches and provenance;
- material needed to reconstruct D025/D050/D075;
- physical OFFLINE TRAIN and validation particles for regional teachers;
- all required original preprocessing conventions and normalization artifacts.

Missing material receives a minimal, role-restricted authenticated export from
the original source environment. Never attempt to reconstruct physical tracking
fields from standardized features or invent a new particle matching.

### 4.2 Exact ladder coordinates

Use the original FullSim view semantics from
[stage2](../../src/hlt_classification/luka_fullsim/stage2.py),
[the original view kernel](../../src/hlt_classification/cms_proxy_ladder/views.py),
and [the native frontend](../../src/hlt_classification/literature_context/transform.py).
Do not substitute the different JetClass2 view implementation.

| Coordinate | Retained privileged measurement fraction |
| --- | ---: |
| U100 | 1, on completed HLT particle support |
| D075 | 3/4 |
| D066 | 2/3 exactly |
| D050 | 1/2 |
| D033 | 1/3 exactly |
| D025 | 1/4 |
| D000 | 0; exact original HLT |

The labels D066/D033 are conventional names for exact thirds, not decimal
0.66/0.33. Preserve original identity-based field choices, kinematic transforms,
group coupling, unmatched fallback, ordering and support. Intermediate views
are not linear interpolation between already standardized feature tensors.

Existing prepared wrappers register only a subset of these coordinates.
Add a campaign-local adapter and new manifests for the extra coordinates;
do not modify legacy preparation manifests or make quarter views pretend to
have an existing third-view hash. Replay D000/U100/D066/D033 against originals.

Assignment/source indices, coordinate IDs, construction switches and offline
quantities are prohibited inputs to deployed students. Ladder teachers may
consume their declared richer view. They do not receive construction metadata.

### 4.3 Evaluation-only targets

TRAIN targets and statistics are the only ones used for optimization.
Validation may be evaluated with richer views for explicitly labelled oracle
metrics, assistant checkpoint selection, teacher selection, and target
predictability diagnostics. Those arrays must be exposed through an evaluation
reader that the training iterator cannot access.

Keep the existing U100 TRAIN bank unchanged. If evaluation U100 targets are
missing, export the same frozen checkpoint in declared FP32 evaluation mode
and use the original TRAIN mean/scale. Record numerical differences from the
original bank on a TRAIN witness; do not silently regenerate that bank.

Use a fixed 4,096-row validation diagnostic panel selected by identity hash,
or all validation rows for assistant selection as specified below. Panel
selection is independent of labels, target values, errors and model results.
No target normalization or region choice is fitted on validation.

Original50k overlaps validation. Historical test reuse and other inherited
disclosures remain explicit. No independent test is introduced; final-test
operations are absent from this campaign.

## 5. Family L: assistants in a fixed U100 coordinate system

### 5.1 What is passed between models

An assistant is a native ParT on its registered intermediate input with:

- its own native classifier trained by true-class CE;
- a 128 -> 128 -> 128 GELU predictor p_j of the original standardized U100 vector.

Export **p_j**, not the assistant's native h_j, native logits, or softmax.
All p_j live in the same U100 coordinate system. Do not independently
re-standardize, rotate, whiten, or rescale an outgoing assistant bank.

The first assistant in each chain learns:

    L_first = CE + 5 * S(p_1, z_U)

For later assistants, keep the total embedding coefficient at 5 and introduce
the predecessor gradually:

    b(u) = 0.5 * r(u)
    L_assistant = CE
                + 5 * (1-b(u)) * S(p_j, z_U)
                + 5 * b(u) * S(p_j, stop_gradient(p_previous))

Both terms train the SAME predictor p_j. At full strength, half the embedding
weight anchors the original target and half follows the predecessor. This
bounds accumulated drift while giving the chain a real effect. The first
assistant has no duplicated predecessor loss.

Each D000 endpoint uses two independent heads from native HLT h:

    L_endpoint = CE + 5*S(g_U(h), z_U)
                    + 5*r(u)*S(g_bridge(h), stop_gradient(p_last))

The final auxiliary head does not receive the actual intermediate view or the
teacher's prediction as an input. All new models start from scratch; the
procedure transfers targets, not weights.

### 5.2 Registered paths

| Task | Input | Predicted predecessor | Outgoing role |
| --- | --- | --- | --- |
| LS_D050 | D050 | Original z_U only | Assistant bank |
| LS_D000 | D000 | LS_D050 | HLT endpoint |
| LM_D066 | D066 | Original z_U only | Assistant bank |
| LM_D033 | D033 | LM_D066, anchored to z_U | Assistant bank |
| LM_D000 | D000 | LM_D033 | HLT endpoint |
| LL_D075 | D075 | Original z_U only | Assistant bank |
| LL_D050 | D050 | LL_D075, anchored to z_U | Assistant bank |
| LL_D025 | D025 | LL_D050, anchored to z_U | Assistant bank |
| LL_D000 | D000 | LL_D025 | HLT endpoint |

An assistant's selected checkpoint minimizes full-validation S(p_j,z_U).
Exact ties use higher native validation AUC, then earlier epoch. The target
criterion is deliberately the original U100 vector at every rung, so a model
cannot win selection simply by imitating a deteriorating predecessor.
Native AUC still selects the final HLT endpoints.

Export selected TRAIN p_j banks after checkpoint reload; optionally export
the evaluation-only diagnostic panel. Record native AUC, original-target error,
predecessor error, per-coordinate variance and mean-predictor error. Finite
constant or poor predictions are scientific results and still produce banks.

### 5.3 Interpretation

The hypothesis is that intermediate inputs produce a more accessible estimate
of a useful rich representation. This is not guaranteed denoising: an assistant
can remove useful variation or add errors. With nested ideal conditional means,
a chain need not add information or change the optimal endpoint predictor.
Any gains here concern finite-data/finite-model learning and optimization.

Compare endpoints with C0 and with each other. Report total chain GPU-hours,
not only final-student cost. The design has no matched same-view repetition
or compute-matched assistant control, so chain-length effects cannot alone
establish that view spacing caused the gain.

## 6. Family R: responses of one shared teacher

### 6.1 Train TR on supported information conditions

Create one from-scratch native ParT teacher with shared parameters and one
classifier across six views:

    B = exact D000
    K = privileged kinematics
    I = privileged identity/charge
    T = privileged tracking
    IT = privileged identity/charge and tracking
    KIT = complete U100 fields on the same HLT support

Reuse the exact field groups and splice semantics from
[selective views](../../src/hlt_classification/luka_selective_teacher_hydra/views.py)
and [their contract definitions](../../src/hlt_classification/luka_selective_teacher_hydra/contracts.py).
K replaces both the designated kinematic features and p4. I/T retain HLT
kinematics. Unmatched slots retain their original fallback. The original
controlled hybrid consistency conventions are preserved, not repaired ad hoc.

One update uses one condition for the entire global batch, identical on both
ranks. Deterministically cycle a shuffled six-condition order; rotate between
epochs. Every jet appears once per epoch, and conditions receive approximately
equal updates. Thirty epochs do not mean thirty epochs per condition.

TR's native CE coefficient is 1. On KIT updates only, add weight1 SmoothL1
between its native h, expressed using the original U100 mean/scale, and z_U.
This anchors the common representation without a freely adapting projection
that could hide relevant variation. No loss forces different conditions to
produce identical embeddings, and no view ID enters the teacher network.

Select TR by the mean full-validation native macro-OVR AUC across the six
conditions, then lower mean CE, then earlier epoch. This averages declared
conditions; it is not a deployed HLT score. Record every condition separately.
Use evaluation mode for target extraction: frozen weights, normalization buffers,
and no dropout. All differences use this ONE checkpoint.

### 6.2 Exact response targets

Estimate a common coordinate mean m and standard deviation s from TRAIN
embeddings, with each of the six conditions weighted equally and s >= 1e-3.
FP64 statistics are stored with their source hashes. Define:

    F(v) = (h_TR(v) - m) / s
    d_K  = F(K)  - F(B)
    d_I  = F(I)  - F(B)
    d_T  = F(T)  - F(B)
    d_IT = F(IT) - F(B)
    d_J  = F(IT) - F(I) - F(T) + F(B)

d_J measures a non-additive model response to the controlled substitutions.
It is not a causal detector effect or independent information beyond the four
evaluations defining it.

For each k in K/I/T/IT, freeze a scalar
a_k=max(0.1, sqrt(mean_TRAIN,coordinate(d_k^2))). Predict d_k/a_k.
This preserves relative axes and exact zero response while bounding amplification
of tiny responses. Do not center differences separately or normalize per jet.
The interaction uses a_IT, not an independently amplified interaction scale.

No subtraction between separately trained teacher embeddings is allowed.
Common weights give coordinates a common meaning but do not prove that every
difference is useful or HLT-predictable.

### 6.3 Student heads and ten-row grid

One removable query-conditioned predictor maps [h_HLT, one_hot(query)] through
Linear -> GELU -> Linear to 128 outputs. Query IDs are K/I/T/IT/J. Width128;
identical parameter shape in all R rows. Queries contain no target-dependent
metadata. The native classifier and original U100 head are unchanged.

| Row | New targets | Maximum lambda |
| --- | --- | ---: |
| R0 = C0 | None | 0 |
| R1 | d_I/a_I | 1 |
| R2 | d_T/a_T | 1 |
| R3 | d_K/a_K | 1 |
| R4 | d_I/a_I, d_T/a_T, d_IT/a_IT | 0.3 |
| R5 | Same bundle; preferred main recipe | 1 |
| R6 | Same bundle | 3 |
| R7 | Same bundle | 10 |
| R8 | d_I/a_I, d_T/a_T, d_J/a_IT | 1 |
| R9 | F(I)/a_I, F(T)/a_T, F(IT)/a_IT | 1 |

L_new averages SmoothL1 over coordinates, then queries, then jets. R9 uses
the same query IDs, scales, predictor shape and total multiplier as R5.
It changes targets to absolute embeddings. Their loss/gradient magnitudes
may still differ; equal lambda is not a gradient-matched control.

R5 versus R9 asks whether responses help beyond adding this teacher's absolute
representations. R8 versus R5 asks whether emphasizing the interaction is
better than predicting the combined response. R1-R3 locate useful information
families; R4-R7 test strength over a broad range.

Response prediction can isolate the least predictable information. Record
validation error against TRAIN-mean response prediction, per-query variance,
and gradient agreement with CE. Keep all finite results, including negative
transfer at weight10.

## 7. Family P: learned regional representations

### 7.1 A fixed, nested spatial question catalog

Use physical particles, not arbitrary standardized feature coordinates.
Derive the jet axis and radial scale from HLT alone for both sources:

- Axis: direction of summed authenticated HLT p4. For summed transverse
  momentum above 1e-12 in the authenticated momentum unit, use
  eta=asinh(pz/pT) and phi=atan2(py,px). Otherwise use eta=phi=0 and the
  degenerate radial-scale fallback below.
- Coordinates: delta_eta and wrapped delta_phi relative to that HLT axis.
- r: their Euclidean radius; theta=atan2(delta_phi,delta_eta), wrapped to [-pi,pi).
- R_H: pT-weighted 90th-percentile HLT radius, floored at 0.05 radians.
- Inner region: r < R_H/2; outer region: all remaining particles.

The outer region has no upper cutoff; retain the complete authorized collection.
For the weighted percentile, sort radii ascending and select the first radius
whose cumulative nonnegative physical pT reaches 90% of the total. Equal-radius
ties give the same radius; do not interpolate. Empty HLT, zero total pT, or
the degenerate summed axis uses R_H=0.05. Never use an OFFLINE axis as a fallback.
Nonfinite physical values remain invalid inputs, not a reason to invoke a
fallback. Perform geometry and boundary decisions in FP64.

| Scale | Partition | Regions per jet |
| --- | --- | ---: |
| Coarse | Inner / outer | 2 |
| Medium | Each radial band split into two angular half-planes | 4 |
| Fine | Each radial band split into four angular quadrants | 8 |

The three scales contain 14 queries. Partitions are nested, cover the source
exactly once at each scale, and use the same HLT-defined boundaries for
HLT and OFFLINE. No constituent match or truth correspondence is required.
Medium angular intervals are [-pi,0) and [0,pi); fine intervals divide these
at -pi/2 and pi/2. Map +pi to -pi, and put exact radial-boundary points in the
outer band. Enumerate each scale in inner-then-outer order, with angular bins
ascending within a band.

The query descriptor is fixed:
four-way one-hot [whole,coarse,medium,fine], inner flag, outer flag,
sin(angular_center), cos(angular_center), and angular half-width divided by pi.
This is nine components. Whole-jet query has its whole flag, zero band flags,
and full-circle descriptor (0,1,1). Coarse queries use that same angular
descriptor; other centers and half-widths follow their intervals. Descriptors reveal geometry,
not OFFLINE occupancy, particle identities, class or selected teacher responses.

### 7.2 Train TP so regional vectors describe local inputs

Train one shared native ParT encoder on both HLT and OFFLINE sources. Each
global update deterministically chooses one source and one partition scale,
balanced across the two sources and three scales. Source and scale are the
same on both ranks; they are not network inputs.

For that batch:

1. Encode the full selected-source jet and apply the native classifier.
2. Encode every crop of the chosen partition independently with the SAME
   encoder and pooling weights.
3. Average the 128-vectors of nonempty crops, with equal weight per nonempty
   crop, and apply the SAME classifier to this aggregate.

The teacher objective is:

    L_TP = CE(full_jet_logits,y) + CE(classifier(mean_valid_crop_embeddings),y)

The global term keeps ordinary discrimination active. The aggregate term gives
local representations a classification role without assigning every individual
crop the whole-jet label. All-empty sources follow the native empty-input
convention and contribute only the full-jet CE.

Recompute native crop features from the crop's own particles only. Do not leave
the full-OFFLINE jet's axis, total pT, multiplicity, or normalization in crop
features. Relative placement is specified by the HLT-defined query.
A crop cannot attend to particles outside itself. Do not derive the target
by pooling full-jet contextual tokens, which could repeat the whole-jet label
everywhere despite different region names.

Use deterministic empty-crop sentinels so all DDP ranks execute the same forward
and collective sequence. Mask empty outputs out of the aggregate and losses.
Chunking/activation checkpointing may control memory only with tested gradient,
RNG and normalization parity; never run unequal SyncBatchNorm call sequences.

Select TP by mean validation AUC over eight declared outputs: full-jet and
each of three crop aggregates, for each of HLT/OFFLINE. Tie on mean CE, then
earlier epoch. This is a target-teacher selection rule, not the final student's
criterion. Report each component; poor local representation quality is not
a technical failure.

### 7.3 Frozen target banks and normalization

From selected TP, export native pooled 128-vectors for each of 14 regions for
TRAIN OFFLINE and HLT, plus full-OFFLINE jet vectors for P9. No teacher forward
runs during a student optimization step.

Fit one shared regional mean/std across both TRAIN sources with equal source,
scale, and per-jet weight; within a scale average nonempty regions. Exclude
empty vectors from statistics. Use FP64 population moments and floor1e-3,
then store FP32 targets. A wholly empty source/scale has zero contribution,
with effective counts recorded. An entirely empty bank gets zero mean and
floored scale. Nonempty constant vectors retain their measured mean and floored
scale. Both remain scientific results.

Share these statistics across both sources and all scales so P8 does not
introduce a separately chosen coordinate weighting. Whole-jet targets for P9
use their own TRAIN statistics. Record the distinction in comparisons.

Empty-region targets have exact zero stored vectors plus a target-only validity
mask. The student receives neither that mask nor OFFLINE counts. It predicts
all registered queries. No extra occupancy/count task is introduced in v1.

### 7.4 Student objective and ten-row grid

One training-only predictor reads [h_HLT, nine-component query descriptor],
uses two width128 GELU hidden layers, and outputs 128 numbers. All P rows use
the same predictor architecture; P9 evaluates its whole-jet query only.

Average SmoothL1 over coordinates, valid regions within each scale, available
scales within a jet, then all global-batch jets. A jet with no valid targets
has zero new loss and still receives CE/U100. Fine scales do not get more
weight merely because they contain more regions. In distributed training,
reduce these per-jet losses using the global sample weighting.

| Row | New targets | Maximum lambda |
| --- | --- | ---: |
| P0 = C0 | None | 0 |
| P1 | Coarse OFFLINE | 1 |
| P2 | Medium OFFLINE | 1 |
| P3 | Fine OFFLINE | 1 |
| P4 | All three OFFLINE scales | 0.3 |
| P5 | All three OFFLINE scales; preferred main recipe | 1 |
| P6 | All three OFFLINE scales | 3 |
| P7 | All three OFFLINE scales | 10 |
| P8 | All three HLT scales from the same TP | 1 |
| P9 | TP full-OFFLINE jet embedding only | 1 |

P5/P8 test privileged source information with shared encoder/query geometry.
P5/P9 ask whether regional supervision adds value beyond this new teacher's
whole-jet embedding; target normalization and query count differ and are
reported. P1-P3 test spatial granularity. P4-P7 test strength.

This is an adaptation of representation prediction at multiple spatial scales,
not a claim to reproduce I-JEPA or a physical particle-matching algorithm.
Small/noisy regions can hurt; region sizes here are proposed fixed defaults.

## 8. Diagnostics, comparisons, and reporting

The primary matched baseline for all 22 HLT rows is fresh FP32/DDP C0.
R0/P0 alias that same result. Do not replace a missing C0 with an older BF16
result. Historical M0HLT/OFFLINE/U100/S2IT/S6 results are labelled separately.

For historical continuity:

    recovery_percent = 100 * (AUC_model - AUC_M0HLT)
                             / (AUC_OFFLINE - AUC_M0HLT)

Authenticate the original anchors (historically 0.96056122 and 0.96452569).
Report raw AUC and delta_C0 alongside recovery; do not replace the denominator
with TR, TP, or an assistant. Report values outside 0-100 without clipping.

Each epoch records native validation metrics, selected-so-far epoch/AUC/recovery,
same-epoch C0 contrast when available, component/weighted losses, LR, throughput,
memory, and target coverage. Completed results use the selected checkpoint,
strict native reload, and authenticated ordered predictions.

On epochs 1/3/10/20/30, measure on fixed diagnostic batches:

- CE, U100, and new-loss shared-encoder gradient norms and pairwise cosines;
- target prediction error versus a TRAIN-mean predictor;
- coordinate variance, response energy, region validity/occupancy by scale;
- assistant original-target/predecessor errors and accumulated drift.

Diagnostic passes restore RNG, buffers and optimizer state. They never change
weights, targets, coefficients, sample schedules, or qualification decisions.
No automatic teacher pruning, loss-weight adjustment, early stopping or
cancellation based on finite scientific quality.

Report native macro/per-class AUC, CE, accuracy and existing QCD rejection
metrics from the same selected checkpoint. Pair bootstrap by authenticated
jet IDs where used, labelled conditional on these selected models; it does
not account for seed variation, adaptive selection, or unverified event-level
independence. Original50k remains overlapping descriptive output.

Rank finalists using the registered validation metric, examine rejection and
failure patterns, then design fresh-seed and FullStat confirmation. Do not
choose a promising epoch on original50k or promote a small-data gain into a
high-data claim. The first screen has no combinations of L/R/P objectives.

## 9. Two-GTX1080Ti execution and resource qualification

Reuse the execution principles in
[the two-GPU plan](LUKA_SUBCLASS_DDP_HYDRA_PLAN.md) and
[contract](../contracts/LUKA_SUBCLASS_DDP_HYDRA.md), with a NEW campaign recipe.

GPU allocations: Hydra/gpus/default/normal, one node, one Slurm task,
six CPUs, 48000 MiB RAM, and gpu:geforce_gtx_1080_ti:2.
Use exactly two torchrun ranks, FP32 and SyncBatchNorm. GPUs replicate the
model and split the batch; their memory is not pooled into one larger device.
No node pinning or promise of current GPU availability is implied.

Global padding is established before the rank split. Scale each local mean
loss by world_size*local_rows/global_rows before DDP averaging. Keep all tail
rows. Validation shards gather unique indices and restore exact original
order. Only rank0 publishes. Teacher branch loops and empty regions must
execute compatible collectives on both ranks.

A setup allocation (initial ceiling 12h, same device/memory profile) qualifies
input readers, genuine production two-rank miniatures and all distinct model/
loss paths on real selected rows. Technical target teachers and assistant
banks are labelled miniature-only, discarded for science, and forbidden as
scientific parents. Setup profiles full/partial batches, largest particle
support, all regional scales, source transitions, validation and target export.

Each scientific task also validates its now-available actual parent banks and
runs a disposable actual-target startup check before fresh initialization.
Setup estimates cannot certify the quality or identity of a teacher that has
not yet been scientifically trained.

Measure per-card peaks and aggregate host peaks including ranks, preprocessing
children and launcher. Use <88% physical GPU and <85% requested host RAM,
and measured full-fit time with 2x safety plus 30min fixed allowance <=48h.
Include all validation, diagnostic, checkpoint and export work.

The old subclass benchmark's 0.164-0.168s updates and 6.6-7.2GiB peaks justify
trying this hardware, not estimating these new teacher jobs. TP has multiple
encoder calls and OFFLINE may have longer support. Never truncate particles,
reduce epochs, change batch semantics or switch precision to pass admission.
Exact recomputation optimizations require parity tests. If the intended
recipe cannot fit, preserve diagnostics and revise the plan before queuing it.

### Storage and export

Use bounded sharded/memory-mapped raw/cache readers, not six resident copies of
every view in each rank. Cached compact targets may be shared read-only at the
host level; do not duplicate all banks in GPU memory.

Approximate uncompressed TRAIN FP32 target payloads:

| Bank | Dense payload estimate |
| --- | ---: |
| One 128-vector per jet | 0.153 GB |
| Six assistant output banks | 0.919 GB |
| Six TR condition vectors | 0.919 GB |
| Fourteen TP region vectors, both sources | 4.290 GB |
| Full-OFFLINE TP vectors | 0.153 GB |

These decimal-byte estimates exclude IDs, masks, statistics, validation
diagnostics, physical views, checkpoints, temporary exports and existing
dependencies. Derive response differences on read from six TR vectors.
Do not reserve storage using these payload estimates alone.

Registration inventories exact serialized files and peak temporary space,
checks personally available quota, and prints the required headroom. Target
exports publish one shard atomically and finish with a completion manifest.
Never overwrite or delete an old partial root to restart registration.

## 10. Automatic queueing, ordering, and recovery

Implemented DAG: **33 Slurm jobs** = setup + durable CPU release + 30 scientific
GPU tasks + final CPU report. Scientific teacher/assistant jobs include their
outgoing target exports; do not count hidden reducer jobs in this total.
If measured export/report time requires separate jobs, version and display
the revised operational DAG before initial authorization.

The queue helper reviews the complete bounded DAG, source, task recipes,
resource ceilings and dependency map once. It submits setup and an afterok
CPU release. After successful setup accounting and authenticated acceptance,
release submits all science and the afterany report without another terminal
interaction. No login-host polling process is required.

True dependency edges:

    LS_D050 -> LS_D000
    LM_D066 -> LM_D033 -> LM_D000
    LL_D075 -> LL_D050 -> LL_D025 -> LL_D000
    TR -> each of R1..R9
    TP -> each of P1..P9
    C0 has no scientific dependency

Every root consumes the original frozen U100 bank as declared. Dependencies
also require successful outgoing-bank publication, not just saved weights.
A finite weak teacher completes and releases its children. Technical failure
blocks only its dependent branch. Unrelated branches proceed.

Preferred root order: C0, LS_D050, TR, TP, LM_D066, LL_D075.
Submit descendants in a topological interleaving that places LS_D000, R5 and
P5 early among available work, then broader weights and source controls.
For R/P, preferred within-family order is 5,6,4,7,1,2,3,8,9.
No family-wide serial lane, per-family cap, hold, or scheduler priority
modification is part of the plan. Slurm controls actual start order.

The report uses afterany on all 30 science IDs and handles missing parents/
failed/cancelled descendants explicitly. Live results/progress are read-only
and do not wait for that report. Configure invalid-dependency handling so
technically blocked jobs become terminal and the report can finish.

Use exact campaign-bound numeric IDs, no-requeue, exclusive submission claims,
write-ahead journals and per-job comments. On ambiguous submission, preserve
evidence and reconcile exact IDs before any retry. No automatic resubmission
or deletion of claims.

CPU release/report use compute/default/normal with scheduler CPU rounding
accepted. Registered bounds: release 1 CPU/4000MiB/1h; report
2 CPUs/8000MiB/2h. Stream predictions and hashes so report memory is bounded;
measure these paths before releasing the actual scientific submission plan.
Setup uses a disposable CPU process to authenticate the real setup inventory,
replay all 30 miniature receipts, and time the reporting metrics on complete
validation predictions. Its report estimate explicitly extrapolates the
2-epoch receipt work to 30 epochs and counts all teacher-component metrics.
It adds the measured authentication cost, applies 2x safety plus ten minutes,
and enforces both CPU walltime and memory bounds. This is a measured estimate,
not a claim to have timed future scientific outputs that do not yet exist.

Existing AGENTS/transfer guidance names Tigris acceptance. For this explicitly
Hydra-targeted campaign, this plan uses real two-GPU Hydra production-worker
miniatures as the equivalent required research qualification. Local tests
establish readiness to queue setup, not remote production acceptance.

## 11. Implementation layout and artifact contracts

The additive `luka_embedding_bridge_hydra` namespace has one
registry/trainer/execution path for all three families. Implementation
consolidates the initially proposed small modules as follows; scientific
definitions above are unchanged:

| Module | Responsibility |
| --- | --- |
| contracts.py / registry.py | Immutable 30-task registry, recipes, aliases, dependencies and budgets |
| campaign.py / transfer.py | Authenticated parents, exact source, physical exports, role-separated creation and storage inventory |
| inputs.py | Raw/cached readers, quarter coordinates, old-view replay, field substitutions |
| response.py | Common scales and response/interaction algebra |
| regions.py | HLT frame, partitions, independent crop frontend and masks |
| banks.py / storage.py | Typed memory-mapped banks, bounded exports, TRAIN statistics and evaluation-only readers |
| model.py / losses.py | Native adapter, isolated auxiliary heads and exact reductions |
| distributed.py / training.py | Two-rank updates, validation, diagnostics, selection and deployment |
| resources.py / qualification.py / execution.py / worker.py | Disposable technical checks, allocation/parity/replay, real measurements and scientific task publication |
| submission.py / reporting.py | Dry/live DAG, durable release, exact-ID recovery, progress and completed contrasts |

Thin script/Slurm wrappers follow the existing absolute-PROJECT_DIR pattern.
Set PYTHONNOUSERSITE=1 and PYTHONDONTWRITEBYTECODE=1 and prepend the registered
environment lib to LD_LIBRARY_PATH. No Fresh_check import or legacy mutation.

New contracts use LUKA_EMBEDDING_BRIDGE_HYDRA_*/v1. The concrete artifact
inventory is in the [contract](../contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md).
In particular, TRAIN normalization is part of each typed TARGET_BANK rather
than a separate NORMALIZATION file; region definitions are hashed in RECIPE;
DATA/VIEW/PHYSICAL record prepared arrays. TRAINING_REPORT/FIT/TASK distinguish
fit evidence from final task publication, and RESULTS is the campaign report.
The [source map](../LEGACY_SOURCE_MAP.md) records exact donor files and commit.

Every bank binds producer checkpoint, native-versus-predicted representation
meaning, input view, exact role, ordered identities/labels, normalization,
dtype/precision, shape, query/region catalog, parent hashes and per-file hashes.
An assistant-output bank cannot be consumed as a native teacher embedding.
Validation banks cannot satisfy TRAIN readers. Technical banks cannot satisfy
science tasks. All publications are atomic and corruption fails closed.

Source donors at 45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08:

- luka_subclass_ddp_hydra: allocation, distributed weighting, diagnostics,
  native replay, resource accounting and durable submission.
- luka_u100_embedding and luka_u100_contrastive: original target normalization,
  native predictor, optimizer and recovery reporting.
- luka_selective_teacher_hydra: authenticated field splicing and teacher-bank
  publication; reuse its semantics, not its independently trained teachers.
- luka_fullsim/stage2, cms_proxy_ladder/views and literature_context/transform:
  original coordinate construction and native frontend.
- luka_next_supervision_hydra: role-restricted physical exports and bounded
  preparation patterns where compatible.

Donor runtime is unchanged. All campaign runtime is additive, and execution
requires the complete exact pushed source snapshot.

## 12. Implementation milestones and required checks

1. **Registry and input preparation.** Freeze all scientific defaults above,
   30-task count, exact dependencies, roles, target scales, seeds and source
   imports. Inventory/export missing quarter-view and physical material.
2. **Common FP32 two-rank C0.** Establish original U100 update and native reload
   parity before adding family losses.
3. **Ladder family.** Test fixed axes, anchored loss, all three paths, frozen
   parents, selection and outgoing-bank integrity.
4. **Response family.** Test teacher conditions, common statistics, algebra,
   query averaging, R9 control and mixed-condition training.
5. **Regional family.** Test partitions, crop isolation, teacher aggregation,
   empty cases, masks, source control and equal-scale weighting.
6. **Execution and reporting.** Exercise exact dry/live fake scheduler DAG,
   partial publication, technical failures, no quality gates and all reports.
7. **Real qualification.** Exact pushed source, full dry review, genuine
   two-GTX1080Ti setup and per-task actual-parent checks precede science.
8. **Readout and confirmation.** Report every row; select a later replicated
   FullStat study only after reviewing native classification and cost.

Meaningful tests must include:

- D000/U100/third-view byte parity, exact quarter fractions, and rejection of
  changed assignment/source provenance.
- D050 paths cannot be accidentally deduplicated.
- Assistant zero-predecessor/ramp endpoints, loss-weight sum5, no gradient into
  parents, and no normalization drift across outgoing banks.
- Response identities d_J=d_IT-d_I-d_T before scaling; zero changes give zero
  responses; separate-teacher/checkpoint subtraction is rejected.
- Region coverage, nesting, boundaries, permutation invariance, wrapped phi,
  empty cases and identical HLT-defined boundaries across sources.
- Changing OFFLINE particles outside a crop cannot change its target when the
  HLT frame and TP checkpoint are fixed. Cropping cannot retain full-source
  derived features.
- Mean-over-regions/scales/jet reductions; P8 source binding; P9 query routing;
  masks cannot become predictor inputs.
- One-rank/two-rank deterministic loss, gradient, buffer and AdamW-update parity,
  including uneven global tails and empty local validation shards.
- Normal-dropout diagnostics leave RNG/buffers/updates unchanged.
- All 22 deployed HLT models load into the original native ParT and infer
  without banks, query descriptors, source metadata or teacher code.
- Complete synthetic full-30 fits and miniature teacher/export/student chains;
  actual installed-Weaver forward/backward tests separately verify the model.
- Corrupt bank/spec/source, mismatched role/checkpoint, duplicate jet IDs,
  nonfinite values, resource overruns, interrupted export and ambiguous sbatch
  acceptance fail safely while preserving evidence.
- Poor finite AUC, collapsed finite targets, tiny responses, and ineffective
  regions still finish all registered rows and appear in reports.

Passing local tests and outstanding remote checks are enumerated in the
[validation record](../LUKA_EMBEDDING_BRIDGE_HYDRA_VALIDATION.md). In particular,
local one-GPU fixtures cannot establish two-rank NCCL/SyncBatchNorm parity.

## 13. Scientific limits and decisions for the next stage

The design improves interpretability and protects the established U100 path,
but it does not guarantee better recovery.

- Ladders trade extra teacher compute for potentially easier targets; the
  anchoring and shorter path are deliberate protections against drift.
- Responses can emphasize irreducible missing information. Broad weights and
  absolute-target controls test that risk rather than assuming it away.
- Local crops may be weak or redundant. The aggregate teacher objective,
  coarse-to-fine comparison, HLT control and whole-jet control address this.
- A teacher trained on its own TRAIN rows may overfit. Validation diagnostics
  assess this; the screen does not claim out-of-fold assistant targets.
- Single-seed selected-checkpoint gains remain provisional. The user explicitly
  chose screening here before FullStat confirmation.
- Thirty fits do not isolate every possible confound. A later direct predicted-
  embedding teacher control, matched-compute chain, pairing shuffle or extra
  seed would require a separately enumerated follow-up.

Highest-priority readouts are LS_D000 versus C0, R5 versus C0/R9, and P5 versus
C0/P8/P9. Also compare all three ladder endpoints and retain high-weight failures.
No winner is chosen from auxiliary error alone.

Before execution registration, freeze actual source paths/hashes, inherited
seed integers, native selection tie implementation, verified source/export
availability, installed-Weaver revision, resource measurements and personal
quota. These are authentication/execution facts, not open invitations to tune
scientific defaults after looking at results.

## 14. References and present status

Teacher-assistant distillation motivates intermediate transfer across capacity
gaps; the information-view and fixed-embedding adaptation here is proposed:
[Mirzadeh et al., Improved Knowledge Distillation via Teacher Assistant](https://arxiv.org/abs/1902.03393).

Regional representation prediction is related to predicting target-region
embeddings from context, but this supervised cross-view jet design is different:
[Assran et al., I-JEPA](https://arxiv.org/abs/2301.08243).

All three families, registration/export, measured setup, automatic release
and readouts are implemented. Local fixtures exercise real training and target
publication, but no research data export, source commit/push or Slurm submission
was performed here. The next task is a scoped commit/push, authenticated input
transfer and registration following the runbook. Production setup must pass on
two GTX1080Ti before automatic scientific release.

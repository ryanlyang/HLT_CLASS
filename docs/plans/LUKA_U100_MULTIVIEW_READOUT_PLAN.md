# U100-centered multiview embedding and classifier-readout study

Date: 2026-10-29. Status: **implemented as a staged debug campaign; remote
acceptance must pass before science submission**. See the
[runbook](../LUKA_U100_MULTIVIEW_RUNBOOK.md) and
[versioned contracts](../contracts/LUKA_U100_MULTIVIEW.md).

2026-10-30 execution repair: job 21983825 completed the CONCAT miniature, then
failed while loading a missing historical native-OFFLINE TRAIN cache block for
FUSION. SPEC/v2 permits reconstructing absent OFFLINE blocks in RAM from the
same raw selected rows, only with exact original block-byte hashes and complete
input-fingerprint replay. It never replaces corrupt present blocks. Scientific
RECIPE/v1, teacher identities, inputs, losses, seeds and schedules are unchanged.
The failed SPEC/v1 execution is preserved; a fresh corrected acceptance must
complete before science. This does not claim that all remote checks have passed.

This document specifies the first SPORC/debug screen agreed in chat. The new
additive runtime implements these choices. It does not authorize submission or
cancellation or claim that any proposed method improves recovery. Architectural
details below are registered defaults, not proven winners.

## 1. The question and the intended best versions

Can an HLT-only student improve over the existing **CE + 5 * U100 embedding
prediction** result by either:

1. also predicting a teacher that sees untagged HLT-plus-OFFLINE particles;
2. also predicting an HLT/OFFLINE cross-attention fusion teacher;
3. making its predicted U100 embedding useful to U100's frozen classifier; or
4. retaining its predicted U100 embedding as an input to its final classifier?

The first campaign deliberately separates those interventions. All-three-target
prediction is included, but readout and retained-predictor changes are initially
tested with **U100 only**. Combining successful interventions is a later study.

The recommended choices favor useful, stable supervision without changing the
student optimization recipe in every arm:

- CONCAT: a genuinely untagged particle multiset, one shared feature space,
  full support, and the ordinary ParT inductive bias.
- FUSION: HLT-primary, one-way OFFLINE-to-HLT cross-attention at four depths;
  the target is an offline-enriched HLT jet representation.
- Frozen readout: retain the coordinate-matching loss and add a modest,
  warmed-up task loss through the exact frozen classifier.
- Retained predictor: a small residual-equivalent combined classifier with an
  exact native-classifier initialization and no additional hidden classifier layer.

These are hypotheses about transfer, not a promise of 45--55% high-statistics
recovery. The approximately 65% small-data results and the 28--32% high-data
results belong to different regimes and must not be pooled.

## 2. Scope, authority, and historical references

This is an additive **small-data SPORC/debug** design. Its baseline semantics
come from [the direct U100 plan](LUKA_U100_EMBEDDING_PLAN.md) and
[U100/v1 contracts](../contracts/LUKA_U100_EMBEDDING.md). The old plan's tier3
single-job authorization does not authorize this new debug graph or its new
oracle architectures. Generic production acceptance rules still apply.

Explicit differences from earlier campaigns:

- CONCAT has **no source tags**. Do not reuse the tagged concatenation recipe.
- Ordinary embedding/readout students deploy the unchanged HLT ParT. Only the
  retained-predictor arm and its matched capacity control use expanded inference.
- No meta adapter, student warm start, new U/D ladder, SupCon, relational loss,
  token prediction, neighborhood target, or independent auxiliary logit head.
- No direct KD on the native classification logits. The readout-KD arm acts
  only through the predicted embedding and a frozen teacher classifier.
- The new teachers are trained from scratch; the already trained U100 is reused.

Use the exact original **regular single-encoder COARSE_U100**, from fixed30
parent commit `74aa58c582ff6f497a2c6d91c503f5489fac3e9a`. It must be the teacher
behind the historical trial U100-w5 result, not c50 UE100, a FullStat teacher,
a fusion U100, or a newly retrained substitute. Authenticate its checkpoint,
parent spec, prepared views, report, predictions, receipt, and original source.

Historical locations are discovery hints, never sufficient proof of reuse:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/
  luka_fixed30_74aa58c5_r1/science/train_COARSE_U100/
  luka_u100_direct_ed6d0af4_tier3_r1/
```

User-reported full-validation reference values, to be authenticated before use:

| Reference | AUC | Role |
| --- | ---: | --- |
| M0HLT | 0.96056122 | Historical native CE-only HLT anchor |
| OFFLINE | 0.96452569 | Fixed historical oracle anchor |
| U100_EMBED_W5 | 0.96206974 | Historical matched-target reference |
| COARSE_U100 | approximately 0.964945 | Privileged target teacher |
| K1_E40_PLAIN | 0.96315997 | Stronger, different-objective benchmark |

K1/E40 is useful context, not the matched control for these single-teacher
changes. A fresh U100-w5 baseline is a required first-campaign fit.

## 3. Frozen common student recipe

| Item | First-campaign choice |
| --- | --- |
| Population | Authenticated ALLJETS 299,234 TRAIN / 74,895 validation |
| Student input | Exact existing D000/HLT views, unchanged in every arm |
| Classes / features | Existing 11-class order and 17-feature interface |
| Initialization | Fresh; exact matched M0HLT initialization and sampler seeds |
| Training | 30 complete epochs; physical batch 128; no accumulation |
| Optimizer | AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 0.01 |
| Learning rate | Existing warmup3/cosine30: peak 3e-4, floor 1.5e-5 |
| Main CE coefficient | 1 |
| Embedding coefficient | Total 5, allocated as in the run table |
| Student predictor | Independent Linear(128,128)-GELU-Linear(128,128) per target |
| Precision | Existing BF16 student forward; FP32 predictors and losses |
| Teacher export | Eval mode, frozen weights, FP32 without autocast |
| Selection | Full-validation macro OVR AUC and exact existing tie-breaks |
| Early stopping | None; poor finite metrics never skip a registered run |

The executable recipe must inherit and hash the original configuration rather
than copy only the numbers printed here. Preserve parameter-group exclusions,
all row ordering, final partial batches, and the native dropout configuration.
Tie-breaks are lower validation CE, higher macro log QCD rejection at 50%, then
earlier update, following `runner.selection_key`.

The U100 predictor keeps the historical isolated CPU initialization seed
`20261019`. Recommended isolated seeds for new heads are CONCAT `2026102901`
and FUSION `2026102902`. Creating extra heads must not advance the backbone,
U100-head, sampler, or training RNG streams. Each teacher retains its own head;
never average unaligned teacher embedding coordinates.

Full validation selects epochs; original50k is overlapping, report-only data.
This historical ALLJETS protocol has **no independent test**. Do not open any
separately sealed final-test role or present original50k as confirmation.

## 4. New teacher C: untagged particle concatenation

### 4.1 Exact view meaning

For each paired jet, construct the multiset union of its native HLT and native
OFFLINE particles. Retain all real particles from both views, including two
measurements of the same underlying particle. No matching, deduplication,
assignment features, or source-dependent downweighting is used.

Concatenate the **common physical particle records first**, then build the common
17 numerical features once on that union. The jet axis, relative momentum and
energy features are derived from the union's summed four-vector.

Implementation reconciliation: inspection of the original ALLJETS runtime
showed that the design's generic JetClass raw14/frontend reference was wrong
for these historical Luka data. Decode stored fields using their authenticated
physical-unit conventions, preserving unknown PID and tracking applicability;
union the resulting `cms2jc2_response.bridge.Particles`; apply
`literature_context.transform.build_inputs` once. This is the same frontend
used by the authenticated D000/OFFLINE/U100 caches, including the tracking
transforms. Do not substitute `jetclass2_delphes.inputs.build_inputs` or
concatenate independently normalized feature tensors. The source-blind union,
full support, common axis and absence of source cues remain the scientific design.

This union is a two-view oracle representation, not a physical reconstructed
jet: its energy may count the same underlying energy twice. Do not reinterpret
the summed vector as a new detector-level truth quantity.

No source bit, type embedding, separator, source-specific numerical encoder,
source-specific normalization, or explicit positional identity is permitted.
Use a deterministic source-blind ordering (descending pT with raw-field
lexicographic ties); identical records remain interchangeable. No positional
embedding may reveal that one source occupied the first half of a sequence.
Provenance remains in preparation manifests, outside the model input.

Naturally different measured properties may still reveal particle origin.
"Untagged" means no explicit provenance cue, not statistical indistinguishability.

### 4.2 Recommended architecture and fit

Use the existing 17-input/11-output native ParT: width 128, eight particle blocks,
eight attention heads, two class-attention blocks, standard four pair features,
and a final normalized 128-D class-token embedding followed by Linear(128,11).
Trim is disabled. No source-dependent attention mask or additional branch.

Train with CE coefficient 1 only, for 30 epochs using the common fixed30
optimizer. Use the parent OFFLINE initialization/sampler seeds for this fresh
oracle. Select by its own full-validation AUC and retain its exact final
pre-classifier tap. No student validation metric selects the teacher epoch.

Audit `n_HLT + n_OFFLINE` on every allowed row and size batches to their actual
maximum support. No inherited single-view 512-slot assumption, real-token crop,
or longest-jet exclusion is allowed. Quadratic attention cost must be measured.

**Rationale:** this is the cleanest realization of the user's same-space idea:
all measurements can interact in one encoder, without requiring source labels
or particle associations. It may nevertheless learn redundant or untransferable
features; stronger oracle AUC is not proof of a better student target.

## 5. New teacher F: HLT-anchored cross-attention fusion

### 5.1 Recommended first topology

Use two width-128, eight-particle-block encoders with separate parameters:

- primary: the unchanged HLT view;
- context: the paired native OFFLINE view.

Each stream uses the existing 17-feature transform for its own view. Unlike
CONCAT, stream roles are explicit in this architecture; there is still no
particle-matching index or learned per-particle provenance field.

For missing native-OFFLINE prepared payloads, use the exact replay policy in
SPEC/v2 and `luka_u100_multiview/offline_cache.py`. Reconstructed blocks must
match the original manifest bytes and full prepared-input fingerprint before
use. This restores the same context view; it is not a new input definition.

After primary blocks **2, 4, 6, and 8**, inject cross-attention from the OFFLINE
context state at the corresponding depth:

```text
HLT queries attend to OFFLINE keys/values
message = MHA(LN(HLT), LN(OFFLINE), LN(OFFLINE), geometry_bias, padding_mask)
HLT <- HLT + sigmoid(gate) * output_projection(message)
```

Use eight heads, pre-normalization, zero attention dropout, zero-initialized
output projections, and scalar gate logits initialized to zero. These are the
existing stable residual-fusion defaults. A zero output projection initially
preserves the HLT path; the projection receives gradients immediately and the
context path begins receiving gradients once it becomes nonzero. Test this
explicitly rather than accidentally detaching or permanently disabling context.

Use native four-vector cross-pair geometry and proper padding masks; no
cross-particle correspondence is assumed. Retain the existing full pair-embedding
batch-normalization population when adapting its implementation.

After block 8, apply the primary's two class-attention blocks and final norm.
Its 128-D **offline-enriched HLT** class vector is `e_F`; its Linear(128,11)
classifier is `C_F`. There is no second offline class head, bidirectional exchange,
late probability ensemble, or concatenated 256-D target in this first topology.

### 5.2 Teacher optimization and boundaries

Train the complete active fusion graph from scratch with CE1 for 30 epochs.
Initialize the primary from the same random seed as the CONCAT encoder; use a
separate forked seed, parent OFFLINE initialization seed + 200000, for context
and cross modules. This matches the common primary initialization, not every
parameter or compute cost between two different architectures.

Both streams learn; no frozen imported OFFLINE encoder, warm start, privilege
withdrawal, modality dropout, or SupCon is introduced in this first screen.
The fusion coefficient is fixed at 1. Select by fused-oracle validation AUC.

**Rationale:** enriching an HLT-primary representation is my preferred fusion
target for transfer: its representation is organized around HLT tokens, while
accessing offline evidence at several depths. This is a prior, not a guarantee
that it is HLT-predictable or that one-way fusion beats symmetric fusion.

Full fused weights and readout must remain available for authenticated export.
Extracting only its primary branch would create a different teacher and is
not a valid substitute. Existing fusion donors are reusable patterns, not
authorization to import their old dataset, population, or checkpoint semantics.

## 6. Common target bank and embedding objective

For each teacher `j` in `{U, C, F}`, export the selected frozen model's raw final
pre-classifier embedding `e_j` on TRAIN, joined to HLT by exact identity/label.
Compute independent FP64 TRAIN population moments:

```text
mu_j = mean_TRAIN(e_j)
sigma_j = max(population_std_TRAIN(e_j), 1e-3)   # coordinatewise
z_j = (e_j - mu_j) / sigma_j
z_hat_j = g_j(h_HLT)
E_j = mean_over_jets_and_128_coordinates SmoothL1(z_hat_j, stopgrad(z_j), beta=1)
```

Store FP32 targets and FP64 normalization evidence. No per-batch target
normalization, validation moments, PCA, fitted validation target, or averaging
of coordinates across teachers. Normalization is fixed throughout student fits.

The student never receives a teacher embedding as an input. `z_j` is used only
inside its training loss. Validation and deployment use HLT inputs only, even
when selecting the retained-predictor arm.

## 7. Registered first-screen student runs

Run IDs below are registered scientific identities in the new contracts.

| Arm | U weight | C weight | F weight | Additional objective / classifier | Deployable graph |
| --- | ---: | ---: | ---: | --- | --- |
| `u100_w5` | 5 | 0 | 0 | Native CE1 | Native HLT ParT |
| `u100_concat` | 2.5 | 2.5 | 0 | Native CE1 | Native HLT ParT |
| `u100_fusion` | 2.5 | 0 | 2.5 | Native CE1 | Native HLT ParT |
| `u100_concat_fusion` | 5/3 | 5/3 | 5/3 | Native CE1 | Native HLT ParT |
| `u100_readout_ce` | 5 | 0 | 0 | Native CE1 + warmed-up frozen-readout CE | Native HLT ParT |
| `u100_readout_kd` | 5 | 0 | 0 | Native CE1 + warmed-up frozen-readout KD | Native HLT ParT |
| `u100_use_prediction` | 5 | 0 | 0 | Combined-classifier CE1 | HLT + predictor + combined classifier |
| `hlt_capacity_control` | 0 | 0 | 0 | Same combined-classifier CE1; no teacher supervision | Same expanded HLT graph |

There are **seven core student fits plus one required capacity control**.
The two fresh oracle teachers make **ten scientific fits**. Setup, exports,
reporting, and optional owned-bank cleanup are additional jobs, not extra
student experiments. U100 is not retrained. No other fits are silently added.

### 7.1 `u100_w5`: reproduction anchor

`L = CE(native_logits, y) + 5 E_U`.

Reproduce the existing U100 training kernel and U100-head initialization as
closely as possible. Run in debug but authenticate the historical tier3 result;
record hardware/kernel differences and do not require equal AUC for admission.
This fit is the shared control for both scientific families.

### 7.2 `u100_concat`: same-space teacher addition

`L = CE(native_logits, y) + 2.5 E_U + 2.5 E_C`.

Use two independent, equally small prediction heads. Preserve the U100 head's
initial state exactly. The CONCAT head must learn an entire jet embedding, not
particle tags, source classification, or a reconstructed pair of views.

This is the most direct test of the untagged union idea at fixed total weight.
Do not use a weak CONCAT oracle score to omit this registered student.

### 7.3 `u100_fusion`: HLT-anchored rich-teacher addition

`L = CE(native_logits, y) + 2.5 E_U + 2.5 E_F`.

The second target is the selected fusion teacher's primary class vector after
all four cross-attention injections. This is my preferred new-teacher arm,
because the target is HLT-anchored rather than a source-agnostic latent union.
Its benefit remains uncertain and is measured against `u100_w5`, not inferred
from the oracle's own AUC.

### 7.4 `u100_concat_fusion`: complementary views

`L = CE(native_logits, y) + (5/3) * (E_U + E_C + E_F)`.

Use exact registered rational allocations before conversion to runtime floats;
do not round each coefficient to 1.67. Independently normalized heads avoid
averaging incompatible coordinates. All three teachers are frozen.

This asks whether the teachers are complementary. It may lose to a two-teacher
arm because U100's allocation shrinks and targets may be redundant. It is not
automatically the strongest setup simply because it has more targets.

### 7.5 `u100_readout_ce`: classification-aware embedding prediction

See the [frozen-readout CE code handoff](../LUKA_U100_FROZEN_READOUT_CE_HANDOFF.md)
for the implementation map, gradient flow, artifacts and adaptation notes.

Let `C_U(e) = W_U e + b_U` be the exact frozen U100 classifier. Then:

```text
e_hat_U = mu_U + sigma_U * z_hat_U
r = C_U(e_hat_U)
a(p) = min(p / 3, 1), p = completed-pass coordinate after this update
L = CE(native_logits, y) + 5 E_U + a(p) * CE(r, y)
```

Recommended final readout coefficient is **1**, linearly ramped over the first
three passes. The embedding coefficient remains 5 from the first update.
This short, predetermined ramp is intended to prevent initially random
embedding predictions from receiving a full-strength classifier penalty before
learning the target scale. It does not respond to validation and is not early
stopping. The exact ramp is a new recommended default, not a historical recipe.

Freeze classifier parameters and all teacher buffers. Do **not** detach
`e_hat_U`, wrap its classifier evaluation in `no_grad`, or place the prediction
path inside `inference_mode`: gradients must reach `g_U` and the HLT encoder.
No gradient enters `mu_U`, `sigma_U`, target embeddings, or teacher weights.
Retain canonical FP64 moments in the artifact, but cast them explicitly to FP32
for this FP32 readout computation. Feed de-standardized predictions to the
classifier, never standardized coordinates directly or a second normalization.

Native classification CE remains 1; it is not split into CE/KD shares. There
are two CE terms on different outputs. Keep the embedding anchor so the
predictor cannot satisfy the readout solely with arbitrary classifier-friendly
vectors unrelated to the actual target.

My prior: this is the most promising inexpensive task-usefulness change. It
emphasizes embedding errors that matter to classification, while ground-truth
CE does not explicitly reward reproducing teacher mistakes. The predictor can
still absorb benefits without improving native classification, so select and
report the native classifier, not the better-looking auxiliary readout.

### 7.6 `u100_readout_kd`: teacher-consistent embedding prediction

Use the same readout path, coefficient ramp, seeds, optimizer, and head as CE:

```text
T = 2
q = stopgrad(softmax(original_U100_TRAIN_logits / T))
p_hat = softmax(r / T)
D = T^2 * mean_over_jets sum_over_classes q * (log(q) - log(p_hat))
L = CE(native_logits, y) + 5 E_U + a(p) * D
```

Compute stable FP32 log-softmax/log-probabilities. Apply `T^2` exactly once;
sum over classes and average over jets, not over all batch/class elements.
The teacher is the same U100, on the same paired TRAIN jets, not OFFLINE.
Export its original logits or authenticate reconstruction through the readout
from de-standardized targets. No validation logits supply training targets.

This keeps the soft teacher class relationships that readout CE need not keep.
Using the same coefficient does not make CE and KD gradient strengths equal;
log both instead of retrospectively tuning a hidden normalization factor.

### 7.7 `u100_use_prediction`: retain the learned transformation

The [retained-predictor code handoff](../LUKA_U100_RETAINED_PREDICTOR_HANDOFF.md)
maps this graph, its control and deployment checks to the implementation.

Use the same U100 predictor and target as the baseline, but replace the native
linear classifier with a linear classifier on concatenated features:

```text
r_keep = W_h h + W_z z_hat_U + b
L = CE(r_keep, y) + 5 E_U
```

This is exactly `Linear(256,11)(concat(h, z_hat_U))`, split for clarity. Initialize
`W_h,b` to the matched native classifier's random initial weights, and `W_z=0`.
Initial logits are therefore the native baseline's logits; the new predictor
branch becomes useful only as training learns to use it. The predictor remains
the small 128-to-128-to-128 GELU MLP. No extra hidden layer, trainable fusion gate,
U100 classifier, or teacher readout loss is added in this first variant.

CE gradients flow into both `h` and `z_hat_U` once the new branch is active;
`E_U` also trains the predictor/encoder from the start. Do not detach predictions
before classification. There is only one classification CE, not an additional
native-head CE that changes the comparison's label-supervision budget.

Select using `r_keep` validation AUC because that is the deployed graph.
Save and validate the entire encoder/predictor/combined-head checkpoint; the old
`native_state()` stripping rule is invalid for this arm. HLT-only deployment
means no teacher data or model is needed, not identical native architecture.

### 7.8 `hlt_capacity_control`: architecture without privileged supervision

Exactly match the preceding graph, all parameter initializations, optimizer,
weight-decay groups, data and schedule; use `L = CE(r_keep, y)` only. The extra
MLP is now an ordinary learned HLT feature branch, not an embedding predictor.
Never open an offline/teacher target bank for this control.

Compare `u100_use_prediction` against BOTH this control and `u100_w5`. Their
difference isolates the effect of embedding supervision within the expanded
architecture. Improvement over native M0HLT alone cannot establish that effect.

## 8. Fairness, accounting, and diagnostics

For the four plain embedding arms the total coefficient is always 5. That
controls the nominal loss budget, not target predictability, teacher capacity,
loss magnitude, or gradient strength. The readout arms intentionally add an
objective; the retained-predictor arm intentionally changes deployment.

Log per epoch:

- native/deployed validation AUC, CE, accuracy, selected epoch, running best;
- CE, each raw/weighted embedding loss, readout CE/KD and actual ramp coefficient;
- occasional per-objective gradient norms and cosine similarities on shared
  encoder parameters, on predeclared TRAIN batches, without optimizer updates;
- train/validation time, peak GPU memory, support maxima, current epoch/batch;
- teacher oracle AUC and checked classifier reconstruction errors at export.

Diagnostic gradient evaluations must not alter RNG, batchnorm state, data order,
or subsequent gradients. They do not adapt weights during a run. Keep auxiliary
readout diagnostics on TRAIN; no validation embedding bank is needed.

Use historical recovery consistently:

```text
recovery = 100 * (AUC_student - AUC_M0HLT) / (AUC_OFFLINE - AUC_M0HLT)
```

Import exact saved anchors rather than round them. Never move the denominator
to the new CONCAT/FUSION oracle. Report their privileged status, even if their
own "recovery" exceeds 100%. Add direct AUC delta versus fresh `u100_w5`, and
for the retained arm versus `hlt_capacity_control`. Report per-class AUC and
QCD rejection with zero-background censoring, not only the macro average.

Inference benchmark all three deployed graphs: native, retained-predictor,
and capacity control on identical HLT batches/precision/hardware. Report
parameters, latency and peak memory; do not label an unmeasured cost negligible.

One seed and repeated use of this validation sample are exploratory. Paired
bootstrap comparisons can quantify event-sampling variability but not training
seed variability or validation-selection bias. A second matched seed and a
separately defined high-statistics confirmation are later work, not silent
additions to the first screen.

## 9. Debug execution and run priority

Intended site: SPORC `debug`, account `reu-aisocial`, environment
`/home/ryreu/miniconda3/envs/atlas_kd_sporc`. Request CPU/RAM/GPU and walltime only
after measuring the new teacher graphs. Existing student A100 evidence does not
prove that doubled-support CONCAT or cross-attention FUSION fits the same budget.

Preferred order balances the user's new-teacher priority with early information:

1. authenticate inputs/U100/reference artifacts and export or authenticate U100
   targets; launch the two new teacher fits as resources permit;
2. run `u100_w5` as a reproducibility control;
3. prioritize `u100_concat` and `u100_fusion` when their own teachers are ready;
4. run `u100_concat_fusion` when both teachers are ready;
5. run U100 readout CE, U100 readout KD, retained predictor, and capacity control.

U100-only work may fill otherwise idle capacity while new teachers train.
An independent U100 fit must not wait on a failed unrelated fusion teacher.
Use only true `afterok` producer dependencies; do not make every student depend
on the previous student's success to express priority. A scheduler priority hint
is not a guaranteed cluster-wide start order. If a two-job concurrency ceiling
is needed, implement it explicitly and distinguish it from scientific lineage.

The intended dependency structure is:

```text
authenticated U100 bank ------> baseline / readout CE / readout KD / retained
authenticated HLT data -------> capacity control
CONCAT fit + TRAIN export ----> U100+CONCAT
FUSION fit + TRAIN export ----> U100+FUSION
both new exports + U100 ------> all-three student
terminal student attempts ---> status/results report
all required outputs validated -> optional owned-target retirement
```

The implementation uses **two stages**. `queue_luka_u100_multiview.sh accept`
submits one real debug acceptance job: longest full-support train/validation
stress, exact U100 readback, compact TRAIN export, and two-epoch production-code
miniatures for both new teachers and all eight students. Its temporary teacher
weights are discarded as scientific initializers. `... science` requires that
receipt and a separate review of the measured requests before submitting ten
scientific fits, U100 export and the final report. There is no automatic
science submission after acceptance. This explicitly implements the existing
new-architecture acceptance requirement; it does not claim remote validation
has already succeeded or waive the required dry-plan review.

All workers must set `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, prepend
`${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`, and source helpers from the absolute
project path. No Fresh_check runtime import or modification is allowed.

## 10. Compact storage and reusable artifacts

- Reuse authenticated particle caches read-only; never duplicate the full dataset.
- Build paired/union batches in RAM or bounded owned node-local scratch.
- Export only final 128-D TRAIN targets, moments, identities, labels, manifests,
  and needed U100 logits/readout parameters. No particle-token/attention banks.
- Three FP32 `(299234,128)` arrays total approximately 0.43 GiB before metadata,
  file-format overhead, predictions, checkpoints and temporary publication space.
  Measure actual peak storage; this is not a complete quota estimate.
- Use one shared authenticated bank per teacher, not one copy per student.
- Retain exact selected teacher checkpoints, readouts, scientific receipts and
  selected student artifacts. Save the selected predictor/full training model
  where needed for reconstruction diagnostics or later explicitly authorized
  continuation; do not assume a native-only checkpoint contains that head.
- Do not save every epoch checkpoint or optimizer state by default. Any resume
  support must explicitly bind optimizer, schedule, sampler, RNG and model state.
- Retire only newly owned target payloads after every registered consumer and
  result check succeeds, using an authenticated retirement manifest. Keep failed
  attempts and shared imported files untouched. Never delete teacher checkpoints
  merely because their first export finished: downstream reuse is still planned.
- Respect personally available quota and filesystem free space with headroom;
  a rounded `df -h` value is not an exact available-GiB guarantee.

Bank lineage must bind selected teacher/checkpoint hash, architecture/view recipe,
source commit, input inventory, ordered TRAIN identities/labels and normalization.
Publish arrays atomically and manifests last. Path existence alone is insufficient.

## 11. Code map

Inspected repository revision: `9868f38641d1648e43f519c20e53ec12efe04dc3`.
The additive implementation uses that reviewed donor revision and records
actual adaptations in [LEGACY_SOURCE_MAP](../LEGACY_SOURCE_MAP.md). Existing
campaign runtime and artifacts are unchanged.

| Existing surface | Reuse / caution |
| --- | --- |
| [U100 contracts](../../src/hlt_classification/luka_u100_embedding/contracts.py) | Baseline target identity, recipe, width and coefficient |
| [U100 training](../../src/hlt_classification/luka_u100_embedding/training.py) | Matched loop and objective; old validator only allows weight 5 and one target |
| [U100 targets](../../src/hlt_classification/luka_u100_embedding/targets.py) | TRAIN moments, identity joins, atomic target export and validation |
| [U100 source](../../src/hlt_classification/luka_u100_embedding/source.py) | Original-source replay and exact regular COARSE_U100 provenance |
| [Auxiliary model](../../src/hlt_classification/luka_embedding_aux/model.py) | Native `mod.fc` input tap and RNG-isolated prediction head |
| [Native Delphes model](../../src/hlt_classification/jetclass2_delphes/model.py) | Installed-Weaver 17-input/11-class model with trim disabled |
| [ALLJETS inputs](../../src/hlt_classification/literature_context/transform.py) | Historical common 17-feature transform; CONCAT applies it once to the physical union |
| [Physical particles](../../src/hlt_classification/luka_fullsim/particles.py) | Authenticated stored-unit conversion, unknown PID and tracking applicability |
| [ALLJETS foundation](../../src/hlt_classification/luka_fullsim/alljets_data.py) | Exact historical trial population; not generic 500k/FullStat splits |
| [Fixed30 schedule](../../src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py) | Original learning-rate and fixed-budget semantics |
| [Runner](../../src/hlt_classification/jetclass2_delphes/runner.py) | HLT inference and exact checkpoint-selection tie-breaks |
| [Fusion model](../../src/hlt_classification/jetclass2_delphes/salience_learned_model.py) | Four same-depth one-way injections; use HLT primary / native OFFLINE context |
| [Fusion offload](../../src/hlt_classification/jetclass2_delphes/dzfix_fusion_model.py) | Lossless activation-storage pattern only after parity; not old dz-fix data/contracts |
| [Tagged concat model](../../src/hlt_classification/jetclass2_delphes/native_concat_model.py) | Warning: 18th source channel and learned source embeddings; NOT this design |
| [Tagged concat data](../../src/hlt_classification/jetclass2_delphes/native_concat_data.py) | Warning: per-view preprocessing plus source code; NOT this union transform |
| [Meta embedding plan](LUKA_META_EMBEDDING_PLAN.md) | Adapted teacher-target readout is different from our student-prediction readout |

Implemented NEW namespace: `src/hlt_classification/luka_u100_multiview/`, with thin
CLI/Slurm adapters. Components:

```text
contracts.py         exact arm inventory, recipe/view/model hashes, schema versions
source.py            authenticated import of U100 and historical references
views.py             untagged raw-union construction and full-support audit
teachers.py          CONCAT and HLT-anchored fusion factories and embedding taps
model.py             independent predictors, frozen readout, retained-head variant
targets.py           shared TRAIN banks and exact classifier reconstruction checks
training.py          matched students, teacher CE fits, diagnostics and selection
worker.py            artifact publication, deployed readback and owned storage
campaign.py          immutable task graph, resource acceptance and reference panel
submission.py        dry/execute authorization, exact journal, concurrency policy
acceptance.py        genuine staged debug stress and production-code miniatures
retention.py         explicit exact-payload retirement after all consumers succeed
```

The CLI is `scripts/luka_u100_multiview.py`; the launcher is
`scripts/queue_luka_u100_multiview.sh`; the worker is
`sbatch/run_luka_u100_multiview.sh`. A distinct v1 artifact family preserves old
single-target contracts and pinned source.

## 12. Acceptance checklist before declaring queue readiness

1. Baseline one-target loss/seed/sampler/update parity with historical U100-w5.
2. CONCAT has exactly 17 channels, full sum cardinality, no source inputs and
   source-blind preprocessing. Padding cannot affect real inputs. Permuting
   tokens leaves eval outputs invariant within a justified numerical tolerance.
3. Swapping externally held source labels without changing union raw particles
   cannot change CONCAT inputs/logits. Physical PID/charge fields remain intact.
4. FUSION cross-masks exclude padding, same-depth context is used, and active
   cross modules receive finite gradients. Zero residual gives native-primary
   parity; nonzero residual actually uses offline context.
5. All target banks reject identity/label/source/teacher/view/role mismatches.
   TRAIN-only normalization and the 1e-3 floor replay exactly.
6. `C_U(mu_U + sigma_U*z_U)` reproduces original U100 TRAIN logits/probabilities
   within explicit FP32 round-trip tolerances. Wrong classifier or moments fail.
7. Readout gradients reach student/predictor, not teacher parameters or targets;
   the original classifier is unchanged byte-for-byte after training steps.
8. Readout-KD direction, class summation, temperature and single `T^2` are tested.
   Raw signed diagnostics use justified numerical tolerances, not blanket zeros.
9. Exact coefficients sum to 5 in plain arms. Readout ramps are predeclared and
   identical between CE/KD. No old C25/P75 or main CE0.5 defaults leak in.
10. Retained/control initial logits match native initialization. Complete deployed
    readback preserves predictor/head; native arms truly remove all auxiliaries.
11. HLT-only inference works with every teacher/bank path unavailable. Capacity
    control never reads offline inputs or embeddings, including during training.
12. Registered bad-but-finite results complete; corrupt/nonfinite data fail closed.
    Report missing/failed rows honestly, without treating job disappearance as success.
13. Synthetic complete DAG, source-drift, duplicate/partial submission, failure
    isolation and retirement-ownership tests pass. No model inputs contain labels,
    identities, matching indices or degradation provenance.
14. Authoritative installed-Weaver/native parity and genuine new-architecture
    resource acceptance are recorded separately from local synthetic tests.
    No silent batch/support/precision changes to fit an allocation.

## 13. Follow-ups: deliberately outside the first campaign

If a new target helps, test additive supervision without reducing U100:
U1005 + new2.5 versus U100-only7.5. For all three, U1005 + CONCAT2.5 + FUSION2.5
must be compared with U100-only10 as well as the fixed-total5 arms. These tests
separate diversity from total supervision strength; they are not queued now.

If a readout variant helps on U100, apply it to the promising multi-target mix
with an explicit total readout-weight budget and a same-mix no-readout control.
Do not automatically multiply readout strength by the number of teachers.
If retained predictions help beyond the capacity control, expand their inference
use only with matched architecture controls and measured deployment cost.

Later options, requiring new registration: readout-weight sweeps, symmetric
fusion, modality dropout, SupCon teachers, second seeds, high-statistics ports,
or a fresh native CE control. None is an implicit part of these eight students.

## 14. Implementation handoff

The additive eight-student/two-new-teacher runtime and queue path now exist.
Local evidence is recorded in HANDOFF. The next remote step is the real debug
acceptance stage from an exact pushed clean commit. Its measured envelope must
pass before the science plan can be rendered and reviewed. No remote resource
measurement, full campaign execution, or recovery improvement is claimed by
local tests. Existing campaigns, datasets and Fresh_check remain untouched.

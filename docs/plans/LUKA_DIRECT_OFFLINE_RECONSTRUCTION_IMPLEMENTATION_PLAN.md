# Direct OFFLINE jet reconstruction: twelve-fit Hydra implementation plan

Status: scientific implementation v1, execution R3, 2026-11-06. The additive R scope is implemented;
see the [contract](../contracts/LUKA_DIRECT_OFFLINE_RECONSTRUCTION.md),
[local evidence](../LUKA_DIRECT_OFFLINE_RECONSTRUCTION_VALIDATION.md) and
[Hydra runbook](../LUKA_DIRECT_OFFLINE_RECONSTRUCTION_RUNBOOK.md).
The user requested this plan after
agreeing to twelve direct-reconstruction fits and explicitly prioritizing
scientific benefit and adequate model capacity over training speed.

This is a separate follow-up to the
[seven-candidate campaign](LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md), not a
replacement for its forty rows or an amendment of an existing execution spec.
Its S family already implements conditional set denoising. R instead predicts
the complete measured particle collection directly from HLT information,
without noisy OFFLINE particles as decoder inputs. Existing A/B/C source,
recipes, jobs, holds, claims, results and reference artifacts keep their meaning.

## 1. Scientific question and intended benefit

Can the additional task of reconstructing a jet's measured pure-OFFLINE
particle collection improve classification by the native HLT model beyond
the established U100-w5 embedding-prediction baseline?

An embedding target summarizes a trained teacher. Direct reconstruction asks
the student to retain information about measured energy sharing, geometry,
composition, multiplicity and tracking properties without relying on that
teacher's choice of hidden coordinates. This is a representation-learning
hypothesis, not a claim that every OFFLINE detail is inferable from HLT or that
better reconstruction necessarily improves classification.

Two decoder inputs are co-primary:

- **Global:** the same pooled 128-dimensional HLT jet embedding used by the
  classifier. This directly tests the representation bottleneck.
- **Global + tokens:** that embedding plus the contextual HLT particle tokens
  after particle block 8. Reconstruction gradients reach both the readout
  representation and the particle encoder. This may make reconstruction easier
  but allows information to bypass the pooled vector; it is an intentional
  scientific comparison, not evidence that the pooled embedding alone improved.

All twelve rows retain classification and the original independent U100-w5
head. All new modules are removed at deployment. Neither student attention
weights nor value messages receive predicted OFFLINE quantities.

## 2. Fixed baseline, data roles and comparison rules

Retain the existing Hydra ALLJETS development recipe:

| Item | Required behavior |
|---|---|
| Student | From-scratch native HLT Particle Transformer, eight particle blocks, width 128, 17 inputs, 11 classes |
| Population | Exactly 299,234 authenticated D000 TRAIN jets and 74,895 D000 validation jets |
| Training | Batch 128, full 30 epochs, original optimizer, learning-rate schedule, native initialization and epoch data order |
| Existing supervision | Original standardized COARSE_U100 128D TRAIN embedding bank; independent predictor; weight 5 SmoothL1 unchanged |
| New target | Full paired raw pure-OFFLINE TRAIN collection; full HLT TRAIN collection for R11/R12 |
| Selection | Existing native HLT validation-AUC checkpoint rule, including its exact tie behavior |
| Deployment | Native HLT encoder and classifier only; no decoder, queries, count head, targets or matching algorithm |
| References | Authenticate and reuse compatible C0/C1 by seed; no reference retraining in the twelve-fit budget |
| Evaluation | Exploratory development validation; original50k overlaps validation and is report-only; no independent test |

Reuse the [U100 contract](../contracts/LUKA_U100_EMBEDDING.md),
[Group A input/execution contracts](../contracts/LUKA_NEXT_SUPERVISION_HYDRA.md)
and [Group C measured-set conventions](../contracts/LUKA_NEXT_SUPERVISION_GROUP_C.md)
where their semantics apply. The new R recipe and output contract must be
distinct: a direct set prediction is not an S velocity prediction.

There is jet-to-jet pairing, not true HLT-to-OFFLINE constituent pairing. A
predicted slot is not an HLT particle. The assignment used by the reconstruction
loss pairs model outputs with target measurements solely to score predictions.

Only TRAIN determines transforms, cardinality support, target banks and
reconstruction diagnostics. Do not export raw validation OFFLINE sets, construct
new validation targets, or inspect a sealed test to support this campaign.
Labels remain classification targets and may stratify declared TRAIN probes;
they are never reconstruction-decoder inputs.

R11/R12 are **HLT reconstruction controls on top of U100-w5**, not models trained
without any OFFLINE supervision. They isolate the source of the new
reconstruction objective. Source-specific target normalization is allowed and
recorded; equal loss weights do not guarantee equal encoder gradient magnitudes.

## 3. Exact twelve-run registry

The family ID is R, with a new independent execution scope. Do not append these
rows to the old immutable forty-row registry or alias them to existing S IDs.

| ID | Decoder input | Reconstruction source | Extra whole-jet consistency | Maximum new loss weight | Seed index |
|---|---|---|---|---:|---:|
| R1 | Global | OFFLINE | No | 0.1 | 0 |
| R2 | Global + tokens | OFFLINE | No | 0.1 | 0 |
| R3 | Global | OFFLINE | No | 0.1 | 1 |
| R4 | Global + tokens | OFFLINE | No | 0.1 | 1 |
| R5 | Global | OFFLINE | No | 0.03 | 0 |
| R6 | Global + tokens | OFFLINE | No | 0.03 | 0 |
| R7 | Global | OFFLINE | No | 0.3 | 0 |
| R8 | Global + tokens | OFFLINE | No | 0.3 | 0 |
| R9 | Global | OFFLINE | Yes | 0.1 | 0 |
| R10 | Global + tokens | OFFLINE | Yes | 0.1 | 0 |
| R11 | Global | HLT | No | 0.1 | 0 |
| R12 | Global + tokens | HLT | No | 0.1 | 0 |

R1/R3 and R2/R4 are the main replicated comparisons. The weight sweeps and
consistency/source ablations initially use seed 0; a single favorable ablation
does not establish replication. R11 compares with R1; R12 compares with R2.
R9/R10 add only the stated consistency loss to R1/R2, respectively.

Twelve is the number of new scientific fits. Shared technical preparation,
resource miniatures and reports are additional operational work. C0/C1 and
compatible S results are reference rows, not replacement slots for new fits.

## 4. Data preparation and full-particle support

### 4.1 Reuse authenticated raw measurements

Start from the existing Group A raw export: ordered TRAIN identities and labels,
ragged offsets, physical four-vectors, measured category, charge, d0/dz,
reported errors and explicit validity. Reuse already transferred bytes after
authenticating all hashes and joins. No new teacher or guessed inverse transform
of clipped model inputs is needed.

The Group C `probabilistic_targets.py` functions `hlt_reference`, `measured`
and `prepare_source` are the closest implementation donors. Reuse their
source-convention-aware transformations or extract a tested common helper;
preserve the existing S artifact bytes and numerical conventions. R must not
call C's entire preparation routine and unnecessarily fit embedding mixtures.

For each source, publish a separate R target wrapper over either authenticated
compatible arrays or newly prepared immutable arrays. A C bank can be reused
only after checking its complete semantics and original parent validation.
Path existence, a matching array width or a successful old Slurm job is
insufficient. Registration must work without any C execution being present.

### 4.2 Target fields and observable frame

Use the paired HLT summed-four-vector axis and vector pT as the coordinate
reference, with the existing positive floor 1e-12. This frame comes from the
same authenticated HLT view. It is not an OFFLINE jet axis or an OFFLINE total.
Retain every source particle, including particles beyond the native HLT input
support if the raw collection contains more. Audit both counts explicitly.

| Group | Scored quantities | Missing-data treatment |
|---|---|---|
| Kinematics | log(pT/HLT-pT), log(E/HLT-pT), eta offset, sin(phi offset), cos(phi offset) | Follow audited physical-source validity and log floors |
| Composition | Six-way measured category including unknown; three-way charge -1/0/+1 | Unknown is a genuine declared category, not a zero placeholder |
| Tracking | asinh(d0/mm), asinh(dz/mm), log(d0err/mm), log(dzerr/mm) | Score a value only when its effective source validity is true |
| Tracking validity | Four binary validity indicators | Always supervise for a real target particle; errors also require positive reported error |
| Presence/count | Real particle count and which predicted slots are assigned | Target-only supervision; never passed to the decoder |

Continuous target means/scales use source-specific, full-TRAIN masked FP64
moments and standard-deviation floor 0.001. Freeze them before scientific fits.
Invalid values contribute neither statistics nor regression loss. The numeric
22-channel storage layout can match S, but R uses categorical/Bernoulli losses
for categories and validity rather than regressing their noised encodings.

Stream preparation into memory-mapped ragged arrays; store no dense
TRAIN-jets-by-maximum-particles tensor. Retain compact raw aggregate targets for
R9/R10 and per-jet frame metadata needed for physical reconstruction in the loss.
Publish the manifest last after all payload hashes and ordered joins verify.

### 4.3 Fixed output-slot capacity, never target-dependent forward shape

Freeze one common capacity for all twelve rows:

```text
K = max(maximum OFFLINE TRAIN count, maximum HLT TRAIN count, 1)
```

All decoder forwards predict all K slots. **Do not trim output queries, mask
them, or choose their number using the current target's true count.** Otherwise
the direct predictor would receive part of its answer. This restriction differs
from S, where noisy target rows are explicitly allowed inputs.

The common K keeps the decoder support and count-head width matched between
OFFLINE and HLT reconstruction controls. Its derivation is recorded as
training-population architecture metadata, not per-event privileged input.
There is no leading-32/64 crop, top-pT selection, rare-jet removal, sampled
particle loss or target-dependent dynamic query budget in this recipe.

Record count quantiles, both maxima, total target particles, native input support
and projected matching/attention costs. Empty source collections, if admitted
by the source contract, use count zero and presence-only supervision. Invalid
collections fail according to the source contract; do not drop their jets.

## 5. Native forward and removable decoder

### 5.1 Tap representations from one native forward

Use the existing pre-classifier tap in `luka_embedding_aux/model.py:head_input`
for the final normalized jet embedding h. For token-conditioned rows, capture
the block-8 output T before class-token aggregation using the installed native
module hook pattern in `luka_particle_teacher_hydra/model.py:ParticleStudent`.

Verify T's actual shape, particle order and mask against the pinned installed
Weaver implementation. Capture the effective native mask if a trimmer can alter
support; never assume an untrimmed external mask still aligns. Require exactly
one tap of each declared representation. Remove hooks in a `finally` block and
clear captured tensors after the training step.

Do not reimplement Weaver attention, run the encoder twice, detach the training
signal accidentally, or update native BatchNorm/dropout state during decoder
probes. The classifier and original U100 head run exactly as before. Targets
and reconstruction predictions never enter their forward path.

### 5.2 Preferred decoder capacity

Use the following initial R/v1 architecture, common across scalar-weight variants:

- K learned query vectors of width 256, initialized with normal std 0.02.
- Four pre-LayerNorm decoder blocks; eight attention heads; feed-forward width
  1024; GELU; zero auxiliary dropout; no auxiliary BatchNorm.
- Each block has query self-attention, cross-attention to permitted HLT memory,
  and a feed-forward layer. An h-conditioned scale/shift provides global
  conditioning at every block; initialize that scale/shift to identity.
- Global memory is a learned 128-to-256 projection of h. Token-conditioned
  memory adds projected block-8 HLT tokens with their real native padding mask.
  No particle-rank or absolute source-ID embeddings are added to that memory.
- The output-slot queries are learnable slots, not particle identities. They
  self-attend to support coordinated particle configurations.
- Separate output heads predict kinematics, category logits, charge logits,
  four track values, four validity logits and one presence logit per slot.
  An h-only count head predicts categories 0 through K for both decoder modes.

Global and token modes share width, depth, heads, output support and losses.
The token-memory projection is an additional training-only component; report
its parameter cost rather than claiming exact training-parameter equality.
R11/R12 match their respective OFFLINE decoder architectures exactly.

Preserve native initialization and data RNG. Initialize new CPU parameters in
an isolated RNG context with proposed root seed 2026110601 plus the existing
replicate offset 1000003. Initialize K-dependent queries/count parameters before
optimizer construction. Use small output weights, zero standardized-regression
biases, uniform categorical logits and initial presence probability 0.1;
encode exact values and module initialization order in the new recipe.

Start with an FP32 auxiliary decoder, assignment costs and losses. Native
BF16 training remains the original recipe. Do not silently shrink the decoder
or change precision to pass a resource check; any necessary execution change
gets its own documented version and numerical/gradient qualification.

### 5.3 Output representation

Regress standardized log-pT, log-energy and eta-offset coordinates. Predict a
single angular coordinate and derive sine/cosine before applying the frozen
target normalization, avoiding a free two-vector with an undefined angle at
zero. Category and charge use separate softmaxes. Tracking values regress their
transformed coordinates; validity and presence use separate sigmoids.

The inverse coordinate map uses only the known HLT frame and frozen constants,
outside the learned decoder, to recover p4 for aggregate losses/probes. Real
OFFLINE coordinates are supplied solely to the target/loss side. This direct
regression need not produce a physically valid p4 at every intermediate update:
log the fraction with E < |p| and do not claim a calibrated detector simulator.
Nonfinite required values are errors; finite poor reconstructions are results.

## 6. Permutation-invariant assignment and exact loss reductions

### 6.1 Per-particle field cost

For output slot i and target particle j, define nonnegative costs in FP32:

```text
kin(i,j)   = mean SmoothL1(beta=1) over the five standardized kinematic values
comp(i,j)  = 0.5 * [CE(category_i, category_j)/log(6)
                 + CE(charge_i, charge_j)/log(3)]
track(i,j) = 0.5 * mean BCEWithLogits(validity_i, validity_j)
            + 0.5 * masked_mean SmoothL1(beta=1) over four track-value fields
field(i,j) = [kin(i,j) + comp(i,j) + track(i,j)] / 3
```

The masked track-value mean is zero when no track values are valid. Validity
supervision remains active. Mask before arithmetic so sentinels/NaNs in unused
padding cannot contaminate a loss. Equal particle weighting is deliberate:
low-pT charged particles must not disappear behind energy weighting. Average
within each jet before averaging jets, so large multiplicities do not dominate.

### 6.2 One-to-one matching, including presence costs

For N real targets, solve a rectangular K-by-N linear assignment. Every real
target has exactly one assigned slot; a slot can match at most one target.
Use `scipy.optimize.linear_sum_assignment` on the detached CPU float64 view of
the finite FP32 cost matrix. Assignment is discrete and has no gradient;
recompute/gather matched losses from graph-connected predictions afterward.
Never detach the actual reconstruction losses.

Presence is supervised on all K slots, with absent-slot weight w0 = 0.1:

```text
positive_i = softplus(-presence_logit_i)
negative_i = softplus( presence_logit_i)
D_presence = N + w0*(K-N)

L_fields   = mean_j field(assigned_slot(j), j)           # zero if N=0
L_presence = [sum_assigned positive_i
              + w0*sum_unassigned negative_i] / D_presence
```

Use a matching cost consistent with these reductions:

```text
assignment_cost(i,j) = field(i,j)/N
                    + [positive_i - w0*negative_i]/D_presence
```

The omitted sum of all absent-slot costs is constant with respect to assignment.
Negative assignment costs can therefore be valid. For N=0 skip assignment and
field loss; K is at least one and presence/count remain defined. For N=K no
absent-slot term is present. Weighted presence probabilities are not assumed
to be calibrated particle-existence probabilities.

Canonicalize targets using their measured fields and validity for deterministic
solver tie behavior, without supplying that order to the network. Do not add
epsilon perturbations that could change the optimum. Test target permutation,
identical duplicates and tied optima explicitly; compare optimal loss when a
tie admits several scientifically equivalent assignments.

Do not substitute nearest-neighbor Chamfer matching, many-to-one assignments,
approximate transport or truncated cost matrices inside this registered recipe.
If exact matching proves too costly, investigate execution changes first and
register any changed scientific loss separately.

### 6.3 Complete reconstruction and training objective

```text
L_count = CE(count_logits(h), N) / log(K+1)
L_reconstruction = L_fields + L_presence + 0.1*L_count

L_R = L_reconstruction                              # R1-R8, R11-R12
L_R = L_reconstruction + 0.1*L_consistency            # R9-R10

lambda(t) = lambda_max * min(t/3, 1)                 # fractional epoch
L_total = CE(native_logits, class)
        + 5*SmoothL1(original_U100_predictor(h), original_U100_target)
        + lambda(t)*mean_jets(L_R)
```

All coefficients are explicit starting choices, not demonstrated optima. Log
raw and weighted components separately. The three-epoch ramp applies to the
whole new objective; the original w5 branch is unchanged. Poor losses or low
recovery never shorten the registered thirty-epoch budget.

## 7. Whole-jet consistency for R9/R10

These rows test whether supervising the collection as a whole adds information
beyond matched local fields. They use the same forward, target support and
matching as R1/R2. The extra loss does not affect discrete assignment.

Use all K predictions with differentiable soft presence weights p_i, not a
target-count-selected subset or a hard presence threshold:

```text
predicted_p4_sum = sum_i p_i * predicted_p4_i
predicted_charged_energy_fraction =
    sum_i p_i*E_i*(1-P_i(charge=0)) / max(sum_i p_i*E_i, epsilon)
```

Targets use all actual source particles and their measured charges. Compute
sums and invariant-mass-squared in FP64 for numerical stability, with gradients
preserved through prediction casts. Let q be the positive HLT-frame pT scale:

- Four-vector term: mean SmoothL1 over `asinh(P_mu/q)` for E, px, py and pz.
- Mass term: SmoothL1 of `asinh((E_sum^2-|p_sum|^2)/q^2)`.
- Charged-energy-fraction term: SmoothL1 of the fractions, with an energy
  denominator floor tied to q and the fixed 1e-12 relative epsilon.

Average those three terms equally to obtain L_consistency. The signed
mass-squared transform deliberately scores negative predicted values instead
of hiding them behind a clamp with zero gradient. Report physical jet mass
only when its mass-squared is nonnegative; disclose invalid predictions.
This is measured-collection consistency, not an assertion that OFFLINE and HLT
jets must have equal energy, multiplicity or particle content.

## 8. Memory management without weakening the experiment

Performance preference means retaining the complete targets and a capable
decoder. It does not make compute or memory limits disappear. Measure costs
honestly, especially the common K, exact CPU assignment and decoder backward.

Preserve one native batch-128 forward and one optimizer update. Do not split
the native forward into microbatches, which could alter its BatchNorm behavior.
Decoder-only microbatches are allowed because the auxiliary decoder has neither
BatchNorm nor dropout and the loss is a mean of independent jet losses.

Preferred bounded-memory implementation if full-batch decoder backward is too
large:

1. Run the native forward once, retaining its graph for logits, h and optionally T.
2. Create temporary leaf views of h/T for auxiliary calculation only.
3. Process decoder microbatches, calculate exact assignments, and backpropagate
   each reconstruction contribution scaled by `microbatch_jets / actual_batch_jets`
   and lambda. Accumulate decoder parameter gradients and gradients on the
   temporary conditioning leaves; release each decoder graph immediately.
4. Inject the accumulated h/T gradients into the original native graph together
   with CE+w5 backward. These injections are essential; detached conditioning
   without them would not train the HLT encoder.
5. Check all required gradients and perform one optimizer step.

Use an explicit vector-Jacobian backward with both h and T outputs for token
rows, so direct token supervision and the h-to-token path both contribute.
Prove numerical gradient/update agreement against an unchunked implementation,
including the interaction between these two paths. Only then enable this path
in production. Decoder activation checkpointing is another measured option.

Microbatching may trim HLT-memory padding using its own observable mask. It
must never trim output queries using target count. Matching may be solved one
jet at a time with bounded CPU buffers. Avoid unbounded process pools, repeated
raw decompression, full-TRAIN particle copies and accidental BLAS oversubscription.
Include any subprocess proportional memory in the host-memory measurement.

## 9. Proposed implementation map and donor lineage

Extend `src/hlt_classification/luka_next_supervision_hydra/` additively. Keep the
current shared trainer, readers, site adapter, journals and reporting rather
than creating another unrelated campaign stack.

Planning baseline inspected: `505b07364041d4f51c42ac9e6a2f85353258606a`.
The paths below exist at planning time except those explicitly marked new.
Record actual donor paths and exact commits in `docs/LEGACY_SOURCE_MAP.md` when
code is migrated; this document itself does not migrate runtime code.

| Existing implementation | Reuse or change required |
|---|---|
| `data.py`, `inputs.py` | Authenticate original raw TRAIN export, D000/U100 joins and role capabilities |
| `probabilistic_targets.py` | Reuse audited frame/transforms, streaming storage and masks without changing old S semantics |
| `luka_embedding_aux/model.py:head_input` | Capture global h from the actual native classifier forward |
| `luka_particle_teacher_hydra/model.py:ParticleStudent` | Donor pattern for block-8 hooks and native-only export; do not import a privileged teacher |
| `model.py`, `training.py` | Add R dispatch, live token capture, reconstruction objective, bounded backward and diagnostics |
| `group_c.py`, `campaign.py`, `reuse.py` | Donor registration/parent-worktree validation; add separate R scope and preserve A/B/C recipes |
| `resources.py`, `worker.py` | Add genuine full-support R setup, six measured paths and R-bound receipts |
| `submission.py`, `reporting.py` | Independent R review/release; R labels, same-seed controls, losses and provisional epochs |
| `luka_particle_teacher_hydra/submission.py`, `worker.py` | Adapt durable bootstrap, exact-ID release and observed Hydra CPU allocation checks into `reconstruction_release.py` |
| `scripts/luka_next_supervision_hydra.py` | Proposed thin `create-r` entry point and existing review/submit/results/progress dispatch |
| Existing queue/Slurm scripts | Reuse scoped authorization/environment handling with an R job prefix |

Proposed new modules:

```text
reconstruction_contracts.py   # R registry, architecture, losses, immutable schemas
reconstruction_targets.py     # direct-set target wrappers, common K, aggregates
reconstruction_model.py       # learned queries, global/token decoder, output heads
reconstruction_loss.py        # exact assignment, mixed-field/presence/count losses
reconstruction.py             # R registration/checking and preparation dispatch
```

Proposed tests: `tests/test_luka_direct_reconstruction_hydra.py` plus focused
regressions in the existing A/B/C suites. Proposed supporting documentation:
`docs/contracts/LUKA_DIRECT_RECONSTRUCTION_HYDRA.md`,
`docs/LUKA_DIRECT_RECONSTRUCTION_HYDRA_RUNBOOK.md` and a validation record.
These are planned paths, not currently implemented commands or contracts.

## 10. Artifacts and compatibility boundaries

Introduce explicit R/v1 content-hashed artifacts for RECIPE, SPEC, TARGETS,
SET_BANK or reused-bank wrappers, SUPPORT, NATIVE_PARITY, RESOURCE_CHECK,
ACCEPTANCE and TRAIN_PROBE. Use a distinct reconstruction contract prefix and
scope `direct_reconstruction_only`; bind existing common fit/epoch/report
envelopes to the R recipe/spec/target hashes. Version common envelopes if new
required semantics cannot be represented without changing their meaning.

SPEC must bind exact clean pushed source, environment, data identities, raw
manifest, U100 target hash, reference reuse proofs, registry/order, model/loss
recipe, site profile and declared storage/compute limits. Preparation-derived
K/transforms have their own immutable SUPPORT/TARGETS hashes; acceptance and
every scientific worker bind these. Do not mutate SPEC after setup.

Original A/B/C recipe hashes and commands must remain stable. Old artifacts
are validated by their own pinned implementation where necessary; do not
import a current source validator while claiming it is executing inside an
older worktree. R registration needs existing input/reference proofs, not
successful A/B/C scientific fits or their newly prepared target banks.

Use fresh immutable roots and exclusive task claims. Publish payloads atomically,
validate references before reuse and preserve partial failures for diagnosis.
No unreviewed resume, overwrite, claim deletion or automatic resubmission.
Runtime must not import `Fresh_check`.

## 11. Implementation phases and verification

| Phase | Deliverable and required checks |
|---|---|
| 1: registry/data | Exact twelve rows; separate scope; authenticated full TRAIN preparation, common K, masks and aggregate targets; no validation target capability |
| 2: decoder | Native h/T taps and both decoder modes; unchanged native/U100 initialization; no target or count access in forward |
| 3: objective | Hand-checked field/presence/count reductions, rectangular assignment, aggregate losses and gradients |
| 4: trainer | Full/partial batch paths, exact decoder chunking, full30 fits, diagnostics, native-only selected export/replay |
| 5: orchestration | Source/parent validation, exact setup receipts, measured resource plans, twelve independent dry-run commands, partial-journal recovery and reports |
| 6: Brown qualification | Installed-Weaver parity and real warmed production-worker miniatures on the chosen Hydra GPU profile; no scientific conclusion from miniature accuracy |

Focused tests must include:

- Target-permutation invariance, output-slot permutation equivariance of loss,
  duplicate particles, assignment ties, N=0/N=1/N=K and unequal source maxima.
- A constructed example in which many-to-one nearest-neighbor matching would
  give the wrong answer; verify exact one-to-one assignments and total cost.
- Agreement between the assignment cost and the stated field/presence objective.
- Padding/invalid tracking ignored before arithmetic; valid zeros remain values;
  charge/category CE and error-validity conditions; all-invalid track support.
- No target-count query trimming; identical decoder predictions when only target
  values, ordering or counts change with fixed HLT inputs/model state.
- No class, source identity, matching index, target mask, target frame or OFFLINE
  coordinate enters the prediction forward. HLT frame metadata is allowed only
  for deterministic physical coordinate interpretation and declared diagnostics.
- Correct h and block-8 gradients; token modes propagate beyond the pooled tap;
  encoder update/RNG parity at auxiliary weight zero; temporary-leaf injection
  matches direct autograd on a nontrivial graph where h depends on T.
- Presence-weighted aggregate calculations against hand-computed p4, zero-energy
  denominator behavior, signed negative mass-squared, wrapped angles and units.
- Full/partial CUDA steps with production decoder widths and real maximum K in
  the remote miniature, including matching, backward and probes.
- Strict native-only checkpoint load/replay after removing all new modules.
- Corrupt target/source rejection, wrong scope/role/seed rejection, read-only
  reports and idempotent submission journals. Poor finite scientific metrics
  and prediction quality do not reject, cancel or skip registered rows.
- Existing A/B/C regression tests, unchanged recipe hashes and no broadened
  final-test or raw-validation access.

## 12. Hydra resources and independent queue workflow

R3 execution correction after setup153208 on57789d36: real target support is
K=298. The setup recorded2.030seconds per full training step and a conservative
4791-minute request, above the2880-minute limit. Matching accounted for roughly
0.84-0.90seconds/full batch; decoder forward was about0.13seconds. The failure
was technical runtime admission, not scientific quality or memory exhaustion.

R3 retains the registered objective and batch/decoder shapes. Validate and
canonicalize targets on CPU, form the eight-jet chunk's full FP32 field costs
in batch, transfer detached FP64 costs once, and solve each full rectangular
assignment with SciPy using at most six threads. Recompute matched losses and
consistency in batch with per-jet masking/reductions. No padded column reaches
the solver; the decoder receives neither target counts nor padding. Preserve
the scalar implementation as the loss/gradient/assignment reference. This is
an execution change with SPEC/v3, not a larger runtime allowance or changed
scientific recipe. New genuine six-path setup remains necessary.

Use Brown Hydra's RTX A6000 profile as the initial intended platform, following
the current campaigns. Its availability is not established by this plan.
Start profiling with one GPU, six CPUs and the existing 32,000 MiB host request,
but do not assume R fits inside the old resources because S or P did.

One genuine setup prepares targets and qualifies six distinct paths:
global/OFFLINE, tokens/OFFLINE, global/OFFLINE+consistency,
tokens/OFFLINE+consistency, global/HLT and tokens/HLT. Identical supports with
different scalar weights or seeds may share timing evidence; bound loss/gradient
behavior at all registered weights in local technical tests.

Measure warmed full and partial batches, representative TRAIN batches, largest
HLT-token tails, largest target tails and track-validity tails. Include both
decoder modes at the common maximum K, exact assignment CPU transfers/solver,
decoder backward, native backward, validation, preparation and once-per-epoch
diagnostics. Record synchronized phase timings and actual peak host/GPU memory.
Discard technical weights before scientific fits.

Use the existing two-times full-fit timing margin plus actual per-fit loading
and 900 seconds as the initial projection rule. One-time shared preparation is
charged to setup, not multiplied by every scientific batch. Budget storage from
actual TRAIN token counts, target reuse/duplication, auxiliary checkpoints and
outputs, with explicit filesystem/quota headroom; do not invent a fixed small
quota before this inventory.

The existing 48-hour/85%-memory envelope is an initial reviewed profile, not a
reason to cut particles, reduce the decoder, skip rows or shorten training. If
exact execution optimizations still do not fit, produce a concrete revised
profile or exact checkpoint-continuation design with its own technical
qualification before release. The user's preference authorizes investigating
greater training cost, not claiming unmeasured resources or altering science
silently. Preserve diagnostics for any technical admission failure.

The R3 execution workflow below retains R2's automatic release and supersedes the original manual second review,
at the user's request. Scientific recipes, targets and twelve-row order are
unchanged. A fresh SPEC/v3 binds the automatic release policy and batched exact
matching kernel; existing registrations are not upgraded or edited in place.

1. Commit/push exact clean source and register a new R-only root from existing
   authenticated inputs/reference proofs. Prefix: `next_R_<commit>_a6000_v3_r1`.
2. Review/authorize the `bootstrap` plan once, including the twelve tasks,
   scientific recipe and resource ceilings. It submits `lukanr_setup` on the
   A6000 and `lukanr_release` on compute with `afterok:<exact setup job>`.
   The release requests one CPU,4000MiB and60minutes, with no GPU. Both jobs
   are owned by Slurm and survive logout. No extra teacher fit is required.
3. The release authenticates its exact job ID, original authorization, pushed
   source, successful setup accounting, full receipts and R-bound acceptance.
   A bounded accounting wait handles publication lag. It then writes and
   scheduler-dry-reviews the complete measured science plan before submitting.
4. All twelve fits queue automatically within the initially authorized bounds,
   with no second confirmation or login process. Token:
   `AUTHORIZE_DIRECT_RECONSTRUCTION_AUTO_V3`; task names `lukanr_R1`, etc.
   Failed setup, missing/corrupt evidence or exceeded resources cannot release
   science. An ambiguous partial sbatch journal requires exact-ID inspection;
   it is never blindly retried. Poor finite quality does not gate admission.
5. Inspect results/progress at any time; assemble the final report on demand.

Exact proposed submission order:

```text
R1 R2 R3 R4 R11 R12 R5 R6 R7 R8 R9 R10
```

This puts both primary strategies and their replications first, then source
controls, followed by paired weight and consistency variants. All scientific
jobs have empty dependency lists: no afterok/afterany chains, family lanes,
holds, nice adjustments, arrays with concurrency caps or waiting for earlier
science results. Submission order encourages mixed coverage but cannot override
Slurm priority/resources. Other campaigns are not canceled, released or requeued.
The only dependency is setup-to-release. There are no fit-to-fit dependencies.
The CPU release may itself wait for compute resources. Submission does not
guarantee immediate GPU starts. Results/report remain available on demand.

## 13. Diagnostics and interpretation as results arrive

Every epoch publishes native HLT accuracy/AUC, historical recovery, CE, original
embedding loss, each reconstruction component, weighted auxiliary contribution,
training time and best-so-far checkpoint. Progress reports show both same-epoch
control comparisons where available and differences from selected same-seed
C0/C1 checkpoints; label those distinct comparisons clearly.

Completed results authenticate selected native checkpoints/probabilities and
report matched same-seed deltas. Preserve unresolved/failed/pending rows.
Do not substitute seed 0 for a missing seed-1 control. Show historical SPORC
U100 and compatible Hydra references with their actual provenance. S denoising
is a secondary cross-method comparison only when its source, population,
schedule and reference compatibility are established; it is not an R rerun.

Use the first TRAIN batch of each epoch for bounded mechanism probes, not every
batch and not new validation target access. Record:

- Matched kinematic/composition/track errors, predicted validity, presence
  precision/recall at a declared diagnostic threshold, count error and slot usage.
- Raw/weighted auxiliary gradient norms at h and, for token modes, T; report
  gradient comparisons without silently introducing adaptive loss weighting.
- Correct, zero and shuffled HLT conditioning on identical targets. In token
  modes shuffle h, tokens and their mask together, or ablate each path with its
  own label. A shuffle of h alone is not a test of full conditioning dependence.
- Within-class/coarse-HLT-kinematics shuffles with actual swapped fraction;
  no-swap groups stay disclosed. For physical aggregate probes, move the
  corresponding HLT frame consistently and re-express fixed targets in that
  frame, or restrict the probe to the standardized field loss and label it so.
- Whole-jet errors, invalid predicted p4 fractions, target multiplicity/validity
  coverage and reconstruction cost. Save only a small deterministic TRAIN sample
  for visual inspection, not a full prediction dump for every epoch.

These are adaptive TRAIN mechanism diagnostics, not independent reconstruction
generalization evidence. Better auxiliary loss, attractive jets or stronger
conditioning are not enough: the primary outcome is native HLT classification
gain over same-seed U100-w5, especially consistency across R1/R3 and R2/R4.
No row is canceled for a poor finite result. Exploratory weight selection and
the absence of independent test data must remain explicit in any claim.

## 14. Implementation realization and readiness boundary

The R registry, targets, decoder, exact objective, bounded backward, reporting
and independent submission now extend the existing Hydra package. Local
verification and its limits are recorded in the validation document. Genuine
Brown setup acceptance is required before the twelve scientific fits; no
runtime or recovery on actual Brown data has been measured locally.

Implementation choices left open above are frozen in v1: decoder chunks of
eight jets, output-head weight std0.001, presence threshold0.5 for diagnostics,
and two aggregate TRAIN examples per epoch. K-dependent modules initialize
on CPU before optimizer construction. R prepares its own complete banks from
the already transferred raw data; it does not depend on a completed C setup or
reuse unverifiable S payloads. This costs a second normalized bank on disk and
is included explicitly in the inventory-derived storage budget. Only native
selected checkpoints are saved, so discarded decoder optimizer/checkpoint
state is not multiplied across epochs. The raw source currently rejects empty
jets; the general set objective is separately verified for N=0.

During implementation, reconcile any necessary change to these loss reductions,
field conventions, decoder support or architecture explicitly in the plan and
new versioned contract before registration. Freeze TRAIN-derived realization
metadata without using validation performance to choose those semantics.

The method references are architectural precedent, not evidence of improved
HLT recovery: [DETR](https://arxiv.org/abs/2005.12872) motivates learned-query
one-to-one set prediction; [set-conditional particle diffusion](https://arxiv.org/abs/2405.10106)
provides context for the existing S comparison. No external implementation is
copied by the implementation.

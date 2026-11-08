# OFFLINE pretext teachers and physical-structure supervision on Hydra

Status: scientific implementation plan v1, 2026-11-08. The additive runtime,
versioned contracts, tests and automatic setup/release workflow are implemented.
The campaign has not been remotely qualified or queued by this implementation
step. See the [runbook](../LUKA_PRETEXT_STRUCTURE_HYDRA_RUNBOOK.md) and
[validation record](../LUKA_PRETEXT_STRUCTURE_HYDRA_VALIDATION.md). This plan
specifies 30 scientific fits and does not change existing campaign specifications.

## 1. Question and preferred design

Can native HLT classification improve beyond CE + U100 embedding weight5 by:

1. Predicting pooled embeddings of OFFLINE encoders trained on particle
   completion, denoising, and structural tasks without jet-class supervision?
2. Predicting explicit OFFLINE energy-flow and branching measurements directly?

The user requested eight differently trained teachers, at least one student
per teacher, additional teacher combinations, and eight structure variants.
Every HLT student retains U100-w5. No student is initialized from an OFFLINE
encoder. All additions are auxiliary training heads; deployed inference is the
original native HLT classifier with the original attention and particle inputs.

The strongest intended bets are `E01`/`E05`/`E07`, the diverse five-teacher
combination `M03`, and the rich structure bundles `S05`/`S07`. These are research
priors, not measured success probabilities. The all-eight combination `M04`
tests breadth; it is not assumed to beat a smaller combination. No 60% or 70%
recovery claim follows from this design or from the earlier ensemble result.

Decisions intended to improve the chance of useful transfer:

- Make the exported 128-dimensional teacher embedding a mandatory information
  bottleneck for every pretext decoder. A decoder that reads particle tokens
  directly could solve its task while leaving the pooled target untrained.
- Train the clean teacher path explicitly as well as its corrupted path, since
  student targets will be extracted from clean OFFLINE jets.
- Recompute corrupted teacher inputs from the visible particles. Hidden total
  momentum, full-jet axes, and full-jet normalization must not leak the answer.
- Prefer relative energy sharing, angular relationships, branching and measured
  identity/tracking structure over an objective dominated by absolute pT.
- Balance hard-constituent and soft-constituent contributions in set losses.
- Give combinations a fixed total extra embedding weight, and introduce new
  auxiliary losses gradually while U100-w5 remains active from the first update.
- Put informative combinations and rich structural bundles early in submission,
  alongside their controls and individual-teacher comparisons.

The previous [teacher sweep](LUKA_TEACHER_SWEEP_HYDRA_PLAN.md) already included
CE plus masked reconstruction and CE plus physical summaries. This campaign's
distinctive test is **zero jet-class loss in all eight new teachers**, task-loss
checkpoint selection, and explicit structural targets in native HLT students.
It must not be presented as the first reconstruction or summary experiment.

User-supplied completed selective-teacher readouts motivate testing additional
information: S2IT reached49.7% full-validation recovery versus46.3% for S00,
while S6 reached52.1%. Those are single-seed selected-checkpoint observations,
not independently confirmed effects. The current width256 capacity results
are preliminary and do not justify changing the native student in this screen.
This plan uses those observations as motivation, not newly authenticated local
scientific evidence or proof that any new target will exceed them.

## 2. Scope, controls, and exact fit count

| Group | IDs | Fresh fits | Meaning |
|---|---|---:|---|
| OFFLINE pretext teachers | T01-T08 | 8 | No jet labels used for optimization or selection |
| One-teacher HLT students | E01-E08 | 8 | U100-w5 plus the corresponding new embedding |
| Multiple-teacher HLT students | M01-M04 | 4 | U100-w5 plus a fixed-budget target combination |
| Direct structure HLT students | S01-S08 | 8 | U100-w5 plus explicit structural supervision |
| Fresh HLT controls | C_U, C_CE | 2 | U100 alone; U100 plus an existing CE OFFLINE teacher |
| Total | | **30** | 8 teacher fits and 22 HLT fits |

Setup miniatures, target preparation, bank export, CPU release and reporting
are operational work, excluded from this fit count but included in resource
estimates. No ninth teacher is fitted: `C_CE` uses authenticated completed T00
from the previous teacher sweep. Two fresh controls are intentional here;
historical C0/S00 results remain references, not substitutes for this matched
training implementation and its new auxiliary ramp.

This is a single-seed exploratory screen. Confirmation with additional seeds
is a subsequent campaign, not an automatic extra fit or validation-selected
replacement for an unsuccessful row. All 30 registered rows remain reportable.

## 3. Population, source, and deployment boundaries

Use the existing Luka FullSim ALLJETS population: 299,234 TRAIN and 74,895
development-validation jets, native D000 HLT input, 17 input features and the
existing 11-class ordering. This is the paired Luka dataset used by the current
Hydra campaigns, not the repository's initial ten-class synthetic HLT-v3 proxy.
Follow [ALLJETS semantics](../contracts/LUKA_FULLSIM_ALLJETS.md) and the existing
[physical conventions](../contracts/LUKA_FULLSIM_STAGE2.md).

The original historical test files have already been repurposed into TRAIN.
There is no independent test in this campaign. Original50k overlaps validation
and is a secondary report at the checkpoint selected on full validation.
Preserve the existing disclosure flags; `final_test_accessed=false` must not be
misrepresented as an untouched test population. No new final-test operation.

Authenticate and reuse:

- Native D000 TRAIN/validation, ordered identities and labels.
- The original U100 TRAIN embedding bank and its unchanged normalization.
- Paired complete OFFLINE physical particles and original native OFFLINE cache
  evidence, source conventions, preparation fingerprints and split memberships.
- Completed T00 CE teacher checkpoint, encoder provenance and target evidence.
- Historical M0HLT/OFFLINE anchors and compatible U100, S00, S2IT/S4/S6 readouts.

The raw reader exposes p4, charge, measured particle-category flags, tracking
values/uncertainties and validity. It does not establish generator ancestry,
true decay-prong labels or truth HLT/OFFLINE constituent matches. This campaign
uses neither generator labels nor constituent correspondence. A branch target
is a measured clustering descriptor, not a claim about the true decay tree.

The existing Group A physical14 export is TRAIN-only. The native OFFLINE
validation transfer is not automatically equivalent to a physical14 export.
Inventory and authenticate available artifacts before requesting a transfer.
If required, export a minimal separate OFFLINE physical14 validation delta from
the original source, with ordered identities, explicit units/validity and
source/preparation hashes. Never invert clipped native features to invent raw
tracking values, reuse TRAIN exports as validation, or relabel old contracts.

Native student inputs are byte-identical across arms. Teacher corruption,
physical summaries, masks, output-slot counts and target validity are accessed
only by training/evaluation code in their declared roles. Student inference
accepts HLT features, p4 and padding masks only. Jet identities are used for joins
and deterministic sampling, never as model features.

## 4. Common optimization and selection

All scientific fits run 30 complete epochs, physical batch128 including the
last partial batch. Student native weights, U100 head and batch order match the
original seed0 U100 recipe. Use the original AdamW groups, exclusions, betas,
epsilon and weight decay by authenticated recipe, without duplicating guessed
defaults. Learning rate warms from zero to 3e-4 over three epochs and follows
the original cosine to 1.5e-5 at epoch30. No performance-based early termination.

Teacher encoders use the native width128, eight particle blocks and two class
attention/pooling blocks. The historical OFFLINE initialization and sampler
seeds plus 200000 are shared across T01-T08. The jet-class classifier is unused
and excluded from their optimizer. Particle-category/count cross entropy is
allowed as a pretext loss; it is not jet-class supervision. No teacher CE
fine-tuning follows pretraining in this campaign.

All auxiliary initialization and masking use isolated RNG domains. Adding a
decoder/head must not advance the native initialization or sampler RNG. A
teacher's embedding predictor has the same initialization in its individual
student and every combination containing it. Identity-keyed corruption depends
on recipe, epoch and draw, never on jet class, validation score or batch order.

Students select by full native HLT validation macro OVR AUC and the existing
selection/tie rule. Teachers select by the lowest full OFFLINE validation
**pretext objective**, using two fixed corruption draws per jet and the clean
objective below; exact ties choose the earlier epoch. Fixed validation draws
are independent of the changing TRAIN masks. Do not choose teachers by an
untrained classification head, downstream student AUC, or a class probe.

Teacher validation particles/targets support evaluation and selection only.
Student validation can evaluate auxiliary errors for diagnostics under an
explicit evaluation-only capability; it never updates parameters, normalizers
or any training bank. Student selection remains classification-only.

## 5. Teacher representation and decoders

### 5.1 The exported representation must learn the pretext task

Let `h_clean` and `h_view` be the native pooled 128D preclassifier states of the
same encoder on clean OFFLINE and corrupted/partial OFFLINE input. Export
`h_clean`, not decoder states, reconstructed particles or a separate projector.

Every teacher minimizes:

```text
L_teacher = L_primary(h_view, task target)
          + 0.25 * L_clean_signature(h_clean, clean structural signature).
```

The common clean signature uses exactly the S07 C2/C3/B/R descriptor families
and equal-family reductions defined in sections 8-9.
All targets are detached measurements, with TRAIN-only fitted transforms.
This low-weight common task trains the exact clean pooling path used at export;
the main tasks differ across teachers. There is no extra embedding consistency,
contrastive, class-label or meta-learning loss hidden in this recipe. Attribute
results to this composite recipe, not to a pure isolated masking objective.

Both main and clean decoders receive only pooled h and fixed learned output
queries. They cannot cross-attend to encoder particle tokens or receive clean
inputs, hidden coordinates, true hidden counts, class labels or teacher logits.
Scalar/vector heads are `128 -> 256 -> output` with GELU. Set decoders have
learned query slots, two width128/four-head query self-attention blocks, FFN
width256, GELU FFNs, dropout0.1, and affine scale/shift conditioning from h at
both blocks. Their field readouts are
separate. There is no autoregressive teacher forcing of true particle values.

A minibatch-shuffled-h decoder diagnostic, shared-encoder/pooling gradient
checks and clean-embedding variance/rank reports detect bypass or collapse.
A wiring bypass is a technical failure; low finite rank or a model learning to
ignore h despite correct wiring is a scientific result, never a release gate.

### 5.2 Masking, frames and completeness

Start from complete physical OFFLINE particles, apply the registered corruption,
then rebuild teacher model inputs from that view with the original frontend.
All input axes and relative scales come from visible/corrupted particles.
Masking already normalized native tensors is prohibited for these new teachers:
their full-jet normalization could reveal removed energy or direction.

Hidden-particle targets use the visible-view axis and scalar pT sum for their
relative coordinates/scales. The six kinematic components are
`log(max(pt/P_visible,1e-8))`, `log(max(E/P_visible,1e-8))`, relative eta,
sin(relative phi), cos(relative phi), and `log1p(m/P_visible)`.
No external clean-jet scalar is supplied to the decoder.
Validate this frame separately from the historical parent-local and HLT-frame
reconstruction conventions. Clean structural targets use their own clean frame.

For N<2, removal is inapplicable: retain the particle, use zero contribution
for the masked task, and train the clean signature. Do not invent a second
particle. Otherwise retain at least one visible token. No top-K input truncation
or hidden output truncation. Set decoder capacity K is the fixed maximum
OFFLINE count in the authenticated TRAIN/validation metadata inventory, declared
before fitting. Capacity uses shape metadata only; it is fixed for all jets,
and a jet's true output count is never supplied as a decoder input.

Use complete ragged targets, padding-only trimming, chunked set decoders and
exact rectangular assignment. Normalize set field losses within each jet, then
average jets. A high-multiplicity jet must not dominate by having more targets.
The match applies to predicted versus removed OFFLINE particles only.

### 5.3 Mixed-field set objective

Each output slot predicts presence; relative log pT/log energy, angular
coordinates and mass; six measured category logits; three charge logits;
tracking values, errors and their four validity bits. Angular phi uses a
sine/cosine pair. Continuous channels use TRAIN-only transforms and population
scales; tracking regression is masked by source validity. Missing is not zero
measurement. Normalize continuous losses separately for kinematics and tracking
so a field with many channels does not acquire an implicit larger coefficient.

Tracking values use asinh(value/1mm) and positive uncertainties use
log(max(error/1mm,1e-8)), under the inherited declared units. Native teacher
inputs keep the registered frontend; these are decoder target coordinates.
Fit primary-task target moments from two fixed TRAIN corruption draws with
task-specific, label-blind RNG. Clean-signature moments come from clean TRAIN.
New epoch masks change training examples, not normalizers. Save the preparation
draw seeds and accumulator support; never estimate these moments on validation.

For assignment, average kinematic SmoothL1(beta1), applicable tracking
SmoothL1(beta1), category CE, charge CE and validity BCE; omit the tracking group
when no target tracking component is applicable. Costs are detached, FP32 and
finite. Exact Hungarian matching resolves the unordered set. Regression/class
loss on the assignment is 0.5 times a uniform target-particle average plus 0.5
times a target-pT-weighted average, with a uniform fallback for zero total pT.
Add 0.25 balanced presence BCE and 0.25 hidden-count CE over 0..K. Presence
positive and negative groups each get half weight when both are present.

This balances soft composition against energetic structure. Separate decoder
and loss-phase timing is required; the previous full-reconstruction runtime
failures prohibit assuming assignment is cheap.

### 5.4 Eight fixed teacher recipes

| ID | Main task and input corruption | Target and intended content |
|---|---|---|
| T01 | Random constituent completion. Remove a uniform subset with q=0.30 or 0.50, chosen with equal probability per jet. | Complete removed mixed-field set; general particle relationships. |
| T02 | Energetic-core completion. Center on the highest-pT constituent; remove it and its nearest angular neighbours until q=0.15 or 0.30 of N is hidden. | Removed mixed-field set; recovering hard structure when the leading cue is absent. |
| T03 | Soft-constituent completion. Uniformly choose 0.40N targets from the lowest-pT 70% of the collection, with deterministic small-N clamping. | Removed mixed-field set; soft radiation, identity and tracking content. |
| T04 | Local energy-flow completion. Uniformly select a constituent as patch center; hide its nearest angular neighbours for q=0.30 or 0.50. | Missing energy/count maps in the visible frame, by particle category; no raw set decoder. |
| T05 | Relational completion. Use T01 masking. | Joint angular-distance/energy-sharing/category distributions of pairs with at least one removed particle. |
| T06 | Multiscale correlation prediction. Uniformly remove 20% of constituents. | Full clean two- and three-point correlation bundle, including moments and cumulative profiles. |
| T07 | Branching prediction. Uniformly remove 20% of constituents. | Full clean primary-branch, soft-drop and dimensionless mass bundle. |
| T08 | Denoising. Uniformly drop 20%; independently multiply retained momentum magnitudes by lognormal noise with log sigma0.10 and smear eta/phi by Gaussian sigma0.01. | Complete clean mixed-field set in the corrupted-view frame; broader information restoration. |

For each fractional mask, use `m=max(1,min(N-1,floor(q*N+0.5)))` when N>=2.
T03 first forms the bottom `max(1,ceil(0.70*N))` pool; cap m by that pool and
N-1. Angular neighbours for T02/T04 use eta-phi distance with wrapped phi and
deterministic physical-key ties.
Permutation of input particles must not change a target except at physically
identical ties, where the output set remains identical. No source row rank or
particle index is supplied to a network.

T08 preserves each retained particle's clean invariant mass when recomputing
energy after momentum changes; charge/category/tracking of retained particles
are not altered. This is a registered denoising augmentation, not a calibrated
HLT simulator or another physical detector endpoint.

T04 uses a 10x10 grid in relative eta/phi, with coordinate edges
`[-inf,-0.8,-0.4,-0.2,-0.1,0,0.1,0.2,0.4,0.8,inf]`. Store removed scalar-pT sum
divided by visible scalar-pT sum and removed counts, separately for six measured
categories. Outer bins preserve all particles; no clipping away a tail.
Apply log1p, TRAIN standardization and equal energy/count family weighting.

T05 uses angular bins with edges
`[0,0.005,0.01,0.02,0.04,0.08,0.12,0.20,0.30,0.45,0.65,1,2,inf]`
and balance bins `[0,0.025,0.05,0.10,0.20,0.30,0.40,0.50]`, where balance is
`min(pt_i,pt_j)/(pt_i+pt_j)`. Maintain charged/charged, charged/neutral and
neutral/neutral channels. Each eligible pair contributes `z_i*z_j`, with z
normalized by full clean scalar pT in the target calculation only. Store both
the normalized histogram and total eligible pair weight. Zero-total pairs have
zero histogram applicability. Average distribution and total-weight losses.
The encoder/decoder never receives the clean normalization scalar.

These detailed tasks are deliberate refinements of the conversational sketch:
T02 removes an energetic region rather than a trivially isolated leading token,
T04 predicts a missing map rather than a full set, and T06/T07 must infer full
structure from a partial view. These decisions fit the same eight-teacher budget.

## 6. Teacher export and individual HLT students

After teacher selection, reload its exact checkpoint and run clean OFFLINE
TRAIN inference with dropout disabled. Export the native pooled 128D h in FP32,
binding encoder, clean frontend, task recipe and selected-checkpoint hashes.
Normalize with TRAIN FP64 population mean/std, floor0.001, using the existing
embedding-bank semantics where truly compatible. Keep new coordinate identities
such as `PRETEXT_T01`; they are not interchangeable OFFLINE or U100 banks.

No PCA, teacher averaging, jet-class probe or decoder output replaces h.
Different teacher coordinate systems need separate prediction heads. Poor
finite teachers still export banks and their dependent students still run.

Let `r(u)=min(1,u/3)` for fractional student epoch u>0. Every student retains
`L_base = CE + 5*mean_SmoothL1(pred_U100, target_U100)`, beta1. For teacher set A:

```text
L = L_base + r(u) * (5/|A|) * sum_t_in_A mean_SmoothL1(pred_t, target_t).
```

Each mean is over jets and all128 coordinates. The first three epochs ramp
only the new supervision; U100 stays weight5 immediately. All extra weights
remain constant after the ramp. There is no adaptive loss weighting or fade.
Log weighted shared-encoder gradient norms/cosines at epochs1,3,10,20,30;
equal loss coefficients do not imply equal gradient sizes.

| Student | New target | Added total embedding weight |
|---|---|---:|
| E01 | T01 | 5 |
| E02 | T02 | 5 |
| E03 | T03 | 5 |
| E04 | T04 | 5 |
| E05 | T05 | 5 |
| E06 | T06 | 5 |
| E07 | T07 | 5 |
| E08 | T08 | 5 |

All predictors are removable `128 -> 128 -> 128` GELU MLPs with independent,
target-stable initialization. They read the student's native pooled h.
Classification does not consume target values or predictor outputs. A scalar
structure head or extra embedding head changes training parameters, not native
deployment capacity. Export `mod.*` only, strict native reload, and verify that
removing every head leaves class probabilities unchanged.

## 7. Four combinations and two controls

| ID | Extra teacher targets | Per-target final weight | Question |
|---|---|---:|---|
| M01 | T01,T05 | 2.5 | Does particle completion complement relational completion? |
| M02 | T05,T06,T07 | 5/3 | Do pair relationships, correlation scales and branching complement one another? |
| M03 | T01,T04,T05,T06,T07 | 1 | Preferred diverse local-plus-global combination. |
| M04 | T01,T02,T03,T04,T05,T06,T07,T08 | 0.625 | Does maximum breadth improve or dilute the useful signals? |
| C_U | None | 0 | Fresh same-seed native CE+U100-w5 control. |
| C_CE | Historical teacher-sweep T00 | 5 | Fresh same-ramp CE+U100-w5+CE-OFFLINE-embedding control. |

The four combinations are fixed before results. Do not choose an additional
best-teacher mixture after inspecting validation and report it as preregistered.
M04/M03 are submitted before smaller combinations when dependencies permit,
honouring the user's interest in more targets as well as fewer targets.

The fixed budget prevents a simple increase in aggregate nominal loss weight.
It does not prove that combinations have identical optimization difficulty,
gradient magnitudes or regularization. Report these differences. Historical
S00 had no new-loss ramp, so its numerical result is context rather than a
replacement for C_CE. S2IT/S4/S6 remain strong historical benchmarks; IT is not
silently added to every new arm. New-teacher-plus-IT and embedding-plus-direct-
structure hybrids are follow-ups outside the30-fit screen.

## 8. Exact physical structure targets

### 8.1 Geometry, units and validity

Compute complete per-jet targets from declared physical OFFLINE particles,
with original units/validity and no jet labels. No learned saliency selector,
validation-ranked observable list or generator truth is needed. For the
HLT-source control, apply the identical definitions to the complete measured
HLT collection. Native classification inputs are still unchanged.

Use `P=sum_i pt_i`, `z_i=pt_i/P`, and wrapped phi differences. Correlation and
clustering distances use rapidity-phi, with rapidity from the physical p4;
T04's raster explicitly uses pseudorapidity-phi. Set reference angular scale
R0=1.0; it is a convention for these descriptors, not an assertion about the
original jet finding radius. Use the summed p4 axis for radial measurements.

Zero-pT constituents contribute zero to pT-weighted geometry, are excluded
from angular clustering, and remain counted in multiplicities. P<=0 is invalid.
Pin finite-coordinate, mass-roundoff and zero-transverse-mass handling in the
versioned recipe, with separate audit counts. A physical-p4 check allows only
roundoff-scale negative mass squared (relative tolerance1e-6 against
max(E^2,|p|^2,1 GeV^2)); larger violations require an explicit source-convention
resolution. No silent fabricated rapidities, units or missing tracking values.

### 8.2 Correlation family C

For beta in {0.5,1,2}, compute exact normalized correlators:

```text
e2(beta) = sum_(i<j) z_i*z_j * (DeltaR_ij/R0)^beta
e3(beta) = sum_(i<j<k) z_i*z_j*z_k
           * (DeltaR_ij*DeltaR_ik*DeltaR_jk/R0^3)^beta.
```

Add beta1 cumulative angular profiles at
`r=[0.005,0.01,0.02,0.04,0.08,0.12,0.20,0.30,0.45,0.65,1,2,inf]`:
e2 restricted to pair distance<=r; e3 restricted to all three distances<=r.
These are campaign-defined cumulative extensions of the correlators, not
standard names for a unique literature observable. Retain total e2/e3 and the
cumulative fractions normalized by their own total. Include log1p(N) and
sum(z_i^2) once in the combined bundle. N<2/3 has zero e2/e3, respectively;
zero-total cumulative fractions have zero applicability, not an invented shape.

Transforms: `log1p(e_n/1e-4)` for totals, raw[0,1] cumulative fractions,
log1p(N), and raw pT dispersion. Fit TRAIN moments separately by channel.
Avoid unstable e3/e2 powers in this first implementation. Provide hand-checked
small-jet fixtures and particle-permutation tests.

Exact three-point targets are precomputed once, not recomputed in every student
batch. A bounded batched matrix implementation may use
`A_ij=sqrt(z_i*z_j)*(DeltaR_ij/R0)^beta`, diagonal zero, so
`e3=trace(A^3)/6`; thresholding A by r gives the stated cumulative e3. Verify
against an independent triple sum. Use FP64 accumulation and chunked memory;
no hidden tuple sampling, leading-particle truncation or approximate replacement
to force the job inside its resource cap.

### 8.3 Branching family B

Build one exclusive angular reclustering tree on positive-pT constituents:
merge the pair with smallest rapidity-phi distance, add p4 in E-scheme, and
continue to one root. There is no beam removal of constituents already in the
selected jet. This uses C/A pair ordering; document the exclusive convention
and validate equivalent cases against a pinned FastJet installation. Stable
physical-key tie breaking must preserve permutation-invariant descriptors.

Decluster along the higher-pT child (the primary branch). For each split define
`z=min(pt_a,pt_b)/(pt_a+pt_b)`, `theta=DeltaR_ab/R0`,
`kt/P=min(pt_a,pt_b)*DeltaR_ab/P`, and `mass_drop=max(m_a,m_b)/m_parent`.
Mass drop is inapplicable when parent mass is zero. Hard splits have z>0.05.

The target bundle contains the first two hard splits in traversal order, their
z, log1p(theta/1e-3), log1p((kt/P)/1e-4), mass drop and presence flags; primary
hard-split count; and normalized ungroomed mass. Count is a clustering count,
not a true decay-prong label. Missing splits train presence; their numerical
fields receive no regression loss. There is no per-particle assignment target.

Add soft-drop descriptors using z_cut=0.10, beta_SD=0: follow the primary
branch until z>0.10. Record split existence, z_g, R_g/R0 and m_g/P, with
log1p transforms for angles/mass. If no accepted split exists, existence is
false, z_g/R_g are masked and the terminal retained-branch mass is recorded.
This convention is explicit; do not insert a fake resolved splitting.

Include an 8x8 primary-branch density over log(1/theta) and log(kt/P), with
finite interior ranges [-2,8] and [-12,0], plus explicit underflow/overflow
bins on each axis. Record count and soft-branch-pT-weighted versions, using
log1p transforms. Ignore zero-kt points in this density and record their count
separately. This is a binned measured branching target inspired by the Lund
plane; it is not a learned teacher embedding or a truth decay tree.

### 8.4 Radial, composition and tracking family R

Radial pT fractions use edges
`[0,0.02,0.05,0.10,0.20,0.35,0.50,0.80,inf]`. Add category number/pT fractions
for all six measured categories, charged-pT fraction, log multiplicity,
effective multiplicity `1/sum(z_i^2)`, and the leading four pT fractions with
presence masks. Keep energy-flow shape and particle content as separate blocks.

Tracking summaries are computed only where the source declares values/errors
valid: counts and pT fractions with |d0/error| or |dz/error| above1,2,4,8,
fractions with positive/negative signed significance, and missing-measurement
fractions. Empty applicable sets retain missingness information and mask the
corresponding measurement summary. These are measurements under inherited
provisional conventions, not vertex/ancestry labels. They are included because
the previous IT result motivates information beyond pure kinematic prongs.

R's blocks are radial shape, identity/energy composition, and tracking. Equal
block weight prevents dozens of tracking thresholds or histogram bins from
dominating a combined objective just by having more outputs.

### 8.5 Normalization and loss

Fit all transforms/moments on TRAIN targets only, by source and descriptor
recipe, using FP64 population statistics and std floor0.001. Constant channels
are marked by a frozen TRAIN-support mask and contribute zero regression loss;
they do not reject the target family or skip the fit. Variable-but-unpredictable
channels remain registered. Save means, scales, support and all definitions.

Each block averages SmoothL1(beta1) over applicable continuous channels and
uses BCE for binary validity/presence outputs. Average applicable channels per
jet, then jets, and average block losses within the family. Missing applicability
is target-side only: never provide an OFFLINE validity bit as student input.
For a completely inapplicable regression block use differentiable zero and
still train its registered presence/missingness outputs.

Student structure heads read pooled native HLT h and use
`128 -> 256 -> descriptor outputs` with GELU. Give C2,C3,B and R blocks separate
heads and equal family weights when combined. The base U100 predictor is
unchanged. The classifier does not directly ingest OFFLINE summaries or
predicted physical values; explicit HLT feature concatenation is outside this
campaign and would need its own deployment/control comparison.

## 9. Eight direct structure versions

```text
L_structure_student = CE + 5*L_U100 + r(u)*lambda*mean(selected family losses).
```

| ID | Target bundle | Source | lambda | Main comparison |
|---|---|---|---:|---|
| S01 | C2 moments and angular profile | OFFLINE | 1 | Two-particle structure alone |
| S02 | C3 moments and angular profile | OFFLINE | 1 | Three-particle structure alone |
| S03 | C2+C3 | OFFLINE | 1 | Joint correlation supervision |
| S04 | B: splits, masses and primary-branch density | OFFLINE | 1 | Branching alone |
| S05 | C2+C3+B | OFFLINE | 1 | Preferred kinematic-structure combination |
| S06 | C2+C3+B | OFFLINE | 0.3 | Does gentler supervision preserve more class information? |
| S07 | C2+C3+B+R | OFFLINE | 1 | Preferred broad structure, identity and tracking target |
| S08 | C2+C3+B+R | HLT | 1 | Same structural task without additional OFFLINE structure targets |

Within these rows C2 and C3 are separate equally weighted families; shared
scalars are supervised once. The shared log1p(N)/pT-dispersion block belongs
to C2 when present and otherwise C3; R's duplicate log1p(N) is omitted when
either correlation family is present. R is the equal mean of its three blocks. The
weights1/0.3 are intentional lower doses than extra embedding weight5: the
losses differ in dimension, applicability and task difficulty, so equal numeric
weights would not be a calibrated comparison. Log their actual gradient sizes.

This refines the earlier eight isolated-observable sketch into eight controlled
versions: individual families, richer combinations, a low-weight version and
a matched HLT-source test. It keeps the requested eight structure fits while
putting more budget into comprehensive targets. S08 still contains U100-w5;
it is not a student without privileged supervision. S07 versus S08 isolates
the source of the added structural targets, subject to normalization/difficulty
differences which must be reported.

## 10. Screening, reporting and decision rules

The target registry above is fixed before results. Earlier discussion proposed
using offline teacher importance and HLT recoverability to discover observables.
Here those are diagnostic hypotheses, not an adaptive admission procedure:
there is no reliable reason yet to assume a feature-importance ranking predicts
student gains, and an attribution search would add a separate selection loop.
Target families are physically motivated, predeclared and all tested.

Report per epoch and for each selected student:

- Full-validation macro/per-class AUC, CE, accuracy, original QCD rejection
  metrics and passing counts; historical recovery and delta to fresh C_U/C_CE.
- Original50k metrics from the same selected checkpoint, explicitly overlapping.
- Raw/weighted auxiliary losses, gradient norms/cosines, elapsed time, update
  count and selected epoch. Compare best-so-far only with clear completion counts.
- Extra-bank predictability and structured-target errors by family. Use a
  deterministic identity-hash TRAIN diagnostic subset of up to8192 rows for
  detailed probes; evaluation-only validation targets may measure generalization.
- Teacher pretext curves, selected objective, clean embedding moments/effective
  rank and corruption-versus-clean diagnostics. Teacher losses are not AUCs and
  cannot be called oracle recovery. No random/untrained classification score.

A frozen linear class probe on teacher h can be an explicitly declared later
diagnostic, fitted on TRAIN only. It is not required for the30-fit campaign,
cannot select checkpoints/teachers, and is not silently added as more training.

Teacher-set comparisons report all individual targets and combinations. A
combination is promising when it improves on C_U, C_CE and its constituent
individual students, with seed confirmation still pending. A low pretext loss
or reconstruction error is not sufficient. Direct structure is more persuasive
when S07 improves on both C_U and S08; S05/S06 distinguish excessive weighting
from a uniformly unsuccessful structural family.

Recovery is `100*(AUC_student-AUC_M0HLT)/(AUC_OFFLINE-AUC_M0HLT)` using the
authenticated historical anchors. Report absolute AUC deltas alongside it.
Do not change anchors to the best new model. Saved-prediction paired bootstrap
intervals may describe validation jet resampling, conditional on selection;
they do not account for the30-row search, training seeds or unknown cross-file
event dependence. No claim of an independently confirmed winner.

Poor finite task quality, constant valid targets, low teacher rank or negative
recovery never fail a job, skip a student or cancel a registered arm. Nonfinite
required computation, wrong roles, corrupt artifacts, source drift or an
unqualified resource requirement are technical failures with preserved evidence.

## 11. Hydra preparation, auto-release and queue order

Intended site: Hydra/Brown `gpus/default/normal`, one typed
`gpu:nvidia_rtx_a6000:1`, six CPUs, 32000 MiB host memory. This explicitly
overrides generic repository Tigris defaults for this campaign. Current free
capacity is not assumed; Slurm chooses start times.

One bootstrap review covers an A6000 `setup`, durable CPU `release` afterok setup,
all30 scientific jobs, and CPU `report` afterany all30: **33 Slurm jobs**.
There is no scientific fit hidden in setup. Setup prepares/authenticates raw
views and structural banks once, validates the installed frontend/Weaver path,
and measures genuine production miniatures before any science release. Setup
may take up to12h; each fit must qualify under48h. CPU release/report request
one CPU, 4000 MiB and one hour; allow Hydra's observed one-requested/two-allocated
CPU rounding with independent controller/accounting evidence.

Successful setup releases all30 jobs automatically from Slurm, so logout is
safe after bootstrap completion. Direct-structure fits and controls can start
while teachers run. Each teacher job publishes its selected encoder and complete
TRAIN embedding bank before completing. Each E student depends only on its own
teacher; M students depend only on their listed teachers. No student depends on
another student's result and no global wait-for-all-teachers barrier exists.

These are genuine artifact dependencies. The older seven-candidate plan's
dependency-free fit policy applied to experiments with already available
targets. It cannot remove a new teacher's required production step; this plan
explicitly registers that difference. Prequeued dependencies can occupy the
queue immediately and become eligible as their banks are completed.

Proposed interleaved submission order (all IDs exactly once):

```text
C_U, S05, T01, E01, T05, E05, S07, T07, E07, C_CE,
T06, E06, S06, T04, E04, T08, E08, S03, T03, E03,
T02, E02, S04, S01, S02, S08, M04, M03, M02, M01
```

Submission order expresses preference only. No holds, nice adjustments,
partition-wide changes, artificial student chains or family concurrency caps.
A failed teacher prevents only students requiring its absent target artifact;
the report lists those blocked/incomplete rows. Finite weak teachers complete
normally and release their students. Do not promise completion in one night.

Setup must cover every distinct encoder-corruption/loss path, set support,
structure bundle and multihead count. Warm representative full128 and partial
batches before timing; measure at least12 steady-state updates separately from
largest-cardinality stress. Include physical preprocessing, exact assignment,
decoder forward/backward, classification validation, two-draw teacher
validation, clean signature passes, target export, startup and diagnostics.
Use the existing2x time safety plus fixed startup/I/O allowances. Preserve
measured per-phase reports before admission checks; do not raise ceilings or
truncate science after a failed setup to force admission.

Acceptance requires peak host memory below85% of the allocation and GPU memory
below88% of actual VRAM. Include child-process memory without double-counting
shared pages. Discard all miniature weights. Because future teacher banks do
not exist during setup, use explicitly technical miniature banks there, then
run a disposable actual-bank startup check in every student before its full fit.

Stream raw/structural data and memory-map compact banks rather than maintaining
all teachers and raw arrays in RAM. Eight FP32 TRAIN 128D banks alone occupy
about1.14GiB; this excludes raw data, structure banks, metadata and checkpoints.
Inventory exact array sizes, complete target support, temporary publication
space and at least2GiB output headroom. Require a reviewed personal-quota budget;
filesystem free space alone is insufficient. Report all transfers/export logs
with progress and do not infer completion from an existing directory.

## 12. Implementation layout and immutable artifacts

Use an additive `src/hlt_classification/luka_pretext_structure_hydra/` package:

```text
contracts.py          exact task registry, recipes, role rules and hashes
campaign.py / reuse.py registration and authenticated old-source imports
inputs.py / transfer.py physical/native capability readers and minimal deltas
structure.py          pure definitions, C/A tree and exact ECF kernels
target_banks.py        streamed preparation, normalizers and ordered joins
teacher_model.py      pooled bottleneck, decoders and label-free objectives
corruption.py         deterministic masks, frames and denoising
teacher_training.py   task-loss selection and clean-embedding export
student_model.py      native classifier, U100 and removable new heads
student_training.py   common fit/selection/export and diagnostics
resources.py          measured paths, inventories and acceptance
submission.py         exact DAG, bootstrap/release and immutable journals
worker.py             thin validated task dispatch
reporting.py          completed/provisional readouts and all comparisons
```

CLI `scripts/luka_pretext_structure_hydra.py`; queue helper
`scripts/queue_luka_pretext_structure_hydra.sh`; worker
`sbatch/run_luka_pretext_structure_hydra.sh`; intended short job prefix `lukaps`.
Required read-only commands include `results`, `progress`, `inspect-submission`
and `teacher-results`. They must not trigger inference or write artifacts.

Proposed new artifacts use `LUKA_PRETEXT_STRUCTURE_HYDRA_<KIND>/v1`: RECIPE,
SPEC, TRANSFER, INVENTORY, STRUCTURE_DEFINITION, NORMALIZATION, STRUCTURE_BANK,
TEACHER_EPOCH, TEACHER_FIT, EMBEDDING_BANK, STUDENT_EPOCH, STUDENT_FIT,
RESOURCE_PROFILE, ACCEPTANCE, PLAN, SUBMIT_CLAIM, RUN_CLAIM, TASK and RESULTS.
These contracts are to be implemented; this document does not create them.
Bind every artifact to exact source, parent receipts, ordered identities,
coordinate definitions, masks/transforms, selected checkpoint and file hashes.
Publish atomically, validate before reuse, and distinguish technical banks.

Reuse audited donor behavior instead of importing a whole old dispatcher.
The inspected source baseline for this plan is
`a6da7a02b773d2397ff0d301bb4b5b400fc7528c`. Candidate donor surfaces:

| Existing source | Intended reuse/adaptation |
|---|---|
| `luka_capacity_hydra/{training,resources,submission}.py` | Matched native training/export, warmed measurement, durable release |
| `luka_selective_teacher_hydra/{model,training,reuse}.py` | Multiple target heads, TRAIN joins and pinned-parent authentication |
| `luka_teacher_sweep_hydra/{inputs,transfer,model}.py` | Native OFFLINE evidence and decoder concepts; replace CE selection and masked-input semantics |
| `luka_next_supervision_hydra/{data,probabilistic_targets,reconstruction_batched}.py` | Complete physical14 source, streamed banks and exact chunked assignment concepts |
| `luka_embedding_aux/targets.py` and `luka_u100_embedding/contracts.py` | Existing target normalization/identity semantics and original U100 recipe |

Confirm actual donor paths and symbols during implementation, pin the exact
donor commit then, and record every reused/adapted file in LEGACY_SOURCE_MAP.
No code is migrated by this planning change. Authenticate historical artifacts
in their original pinned worktree/subprocess; importing an old validator into
the wrong project must not bypass its source guard. Preserve old namespaces,
specs, partial outputs and user work.

Workers use absolute project paths, `PYTHONNOUSERSITE=1`,
`PYTHONDONTWRITEBYTECODE=1` and the selected environment's lib directory first
in LD_LIBRARY_PATH. Exact pushed source and environment hashes are mandatory.
Exclusive claims and append-only submission journals precede sbatch. Fully
completed ledgers are idempotent; partial/ambiguous submissions stop for exact-ID
inspection. No broad cancellations, deletion of claims or automatic retry.

## 13. Implementation milestones and acceptance evidence

1. Implement pure physical structures/corruption and authenticated target banks.
   Verify independent hand calculations and full-population inventory before
   building training orchestration around incorrect targets.
2. Implement the pooled pretext encoder and all eight objectives. Prove hidden
   inputs cannot reach decoders, clean h receives gradients and task-only
   selection/export uses the selected encoder. Exercise every edge case.
3. Implement all22 HLT fits and fixed combinations with original U100 parity,
   isolated heads and exact native inference reload.
4. Implement registration, full dry plan, setup/auto-release, resource reports
   and read-only partial results. Preserve exact scientific counts and DAG.
5. Record local evidence, then qualify the real installed Hydra/data path before
   automatic scientific release. Local GPU fixtures do not certify Brown.

Focused tests must cover: ECF2/3 matrix-versus-enumeration parity; cumulative
monotonicity; permutation and global-phi invariance; tree ties and no-split
cases; soft-drop threshold boundaries; zero-pT/low-count/invalid-tracking cases;
mass and frame conventions; no particle truncation; corruption and RNG replay;
TRAIN-only statistics; label-blind teacher gradients/selection; no decoder
bypass; complete nonfinite detection; exact unordered set matching; mixed-field
loss reductions; gradient flow through clean pooling; selected bank replay;
student loss budgets/ramps; target-stable head initialization; native U100
control parity; removal of all training heads; all30 rows/full dependencies;
poor finite teachers releasing students; stale source/role/hash rejection;
resource limits; CPU rounding; partial/duplicate submission; and report behavior
when fits fail or remain incomplete. Use real miniature production paths.

For docs-only planning validation, check relative links, 30 unique scientific
IDs, every referenced teacher, fixed target budgets and complete queue order.
Do not claim the numerical kernels, physical data or Slurm path have been tested
until code and corresponding evidence exist.

## 14. References and limits of the proposal

- [Masked particle modelling](https://arxiv.org/abs/2409.12589) motivates
  label-free missing-set tasks. Our required pooled bottleneck and transfer
  to HLT are specific adaptations, not performance claims from that paper.
- [Energy correlation functions](https://arxiv.org/abs/1305.0007) motivate
  energy-weighted multi-particle angular structure without constituent matching.
  The explicit cumulative profiles here are a declared campaign extension.
- [Soft Drop](https://arxiv.org/abs/1402.2657) motivates the registered grooming
  descriptors; these are measured jet structure, not true decay assignments.
- [The Lund jet plane](https://arxiv.org/abs/1807.04758) motivates primary-branch
  summaries across angular and momentum scales. Binning/normalization here are
  fixed implementation choices, not evidence of student transferability.

The main risks are pretext tasks learning nuisance detail, target information
being unavailable in HLT, correlated teachers diluting stronger targets,
structural losses dominating classification, and validation selection noise.
The bottleneck, clean-path objective, task diversity, fixed embedding budget,
small structural weights, source control and matched references address these
risks without asserting they solve them. The outcome is the measured HLT
student performance, including negative results.

## 15. Implemented numerical and operational refinements

The v1 contract makes rare-case behavior explicit: if removal would leave only
zero-pT visible particles, retain the highest-pT particle and exchange a visible
zero-pT particle. N<2 has zero primary loss and still receives the clean task.
Rapidity/eta denominators use a fixed 1e-12 floor. Mass-squared is clipped only
within relative 1e-6 rounding tolerance; larger spacelike inputs fail. These
rules complete previously unspecified edge cases; no particle cap is introduced.

The native parameterization, 30-fit registry, losses, targets, ramps and
submission order above are unchanged. The complete-particle C/A implementation
caches branch coordinates and vectorizes candidate distances while retaining
exact heap ordering and physical-key ties. Independent pinned FastJet parity
covers equivalent no-beam cases; ties follow the explicit campaign convention.

Setup has a 32-GiB output allowance with 2-GiB personal headroom after input
transfers. It checks array allocations before writing and records concrete
serialized checkpoint and bank inventories. It retains technical miniature
banks only as setup evidence; scientific students reject those banks. The
resource record separates steady updates from synchronized phase diagnostics,
full validation, target export and probes. Local synthetic qualifications are
not substituted for the real-data A6000 setup admission.

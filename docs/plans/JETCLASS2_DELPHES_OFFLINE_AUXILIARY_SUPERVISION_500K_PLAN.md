# JetClass2 Delphes Offline Auxiliary Supervision: 500k Four-Arm Study

Date: 2026-09-12. Plan revision: 1.

Status: **scientific specification with an isolated implementation and staged
queue tooling**. See the [executable v1 contract](../contracts/JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION.md)
and HANDOFF for evidence. Local tests do not substitute for the mandatory real
installed-Weaver/A100 gate. No scientific fit, source publication, GPU allocation
or change to another campaign is authorized by the existence of this document.

Working study name: `JC2_OFFLINE_AUX_500K`.
Reserved implementation namespace: `jetclass2_delphes/offline_aux/`.
Reserved Slurm prefix: `jc2aux_`.

## 1. Question, hypothesis, and boundaries

Can direct supervision from paired **offline reconstructed jet properties**
improve a single classifier whose forward pass consumes **HLT particles only**?

The offline jet is a source of training targets. It is not an additional model
input, a pretrained teacher, or a simulation-truth labelling service. We jointly
train classification and small auxiliary prediction heads from the same HLT jet
representation, then discard the auxiliary heads for classification deployment.

The four arms are:

| ID | Classification supervision | Auxiliary supervision |
| --- | --- | --- |
| `CE` | Ordinary class CE | None |
| `COMP` | Ordinary class CE | Offline species counts and transverse-momentum fractions |
| `STRUCT` | Ordinary class CE | Offline normalized mass, width, radial profile, pair-angle profile |
| `BOTH` | Ordinary class CE | Both groups, sharing a fixed total auxiliary coefficient |

The hypothesis is that these tasks encourage the HLT representation to retain
class-relevant information about composition and geometry. Neither predictable
targets nor good auxiliary regression guarantees better classification. Negative
transfer and no improvement are valid outcomes. `BOTH` need not beat either
single-group arm, and its differently weighted objective is not a strict test
of statistical interaction between tasks.

This first study contains **22 fresh scientific fits**: ten discovery fits,
followed by twelve confirmation fits. There is no offline-model fit, logit KD,
temperature, C25P75 mixture, ladder, constituent matching, representation KD,
ensemble, pretrained initialization, or classification-head widening. In
particular, CE has coefficient **1**, not 0.25. Single-model improvements are
the primary result; averaging seed predictions is outside this study.

### 1.1 Scientific context, not a promise of effectiveness

ATLAS GN2 motivates multitask supervision, but its track-origin and vertex
targets use information that is not supplied by our reconstructed offline
view. We must not label these summaries as truth origins, vertices, or an
implementation of GN2. See [Transforming jet flavour tagging at ATLAS](https://arxiv.org/abs/2505.19689).

Energy-and-angle observables provide a physically motivated way to summarize
jet structure. Our finite pair-angle histogram is a proposed auxiliary target,
not the complete family of energy correlation functions, nor a demonstrated
optimal target. See [Energy Correlation Functions for Jet Substructure](https://arxiv.org/abs/1305.0007).

Relative task weighting can materially affect multitask learning. We therefore
register an explicit weight screen rather than assume raw losses are naturally
commensurate. Learned uncertainty weighting, GradNorm, and gradient surgery are
not part of revision 1. See [Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115).

## 2. Authority, isolation, and inherited contracts

This plan governs only the new auxiliary study. It inherits the dataset,
selection, canonical identity, native HLT inputs, class definitions, and
integrity rules from:

- [JetClass2 Delphes migration plan](JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md).
- [JetClass2 Delphes contracts](../contracts/JETCLASS2_DELPHES.md).
- [Testing and acceptance requirements](../TESTING.md).
- [Repository transfer plan](../../REPOSITORY_TRANSFER_PLAN.md), for cross-cutting provenance and isolation principles.

Explicit scoped differences from the migration's four-spine execution are the
auxiliary objectives, auxiliary heads, fresh study seed domains, and internal
validation roles. Its matching foundation, teacher banks, graph, CE/KD recipe
artifact, and scientific runtime profile are not automatically suitable for
this study. New versioned artifacts must express the differences; do not insert
auxiliary keys into an old immutable recipe or reinterpret an old validation
report as a report on the new comparison slice.

Use a fresh source-pinned worktree, output root, source lock, command plans,
submission journals, exact job ledgers, and restart-zero recovery. Treat the
existing migration inputs and other campaign artifacts as read-only. No worker
may write, cancel, hold, release, reprioritize, or install packages into another
campaign or its environment. Other scientific campaigns need not finish first.

The selected execution site is SPORC tier3/A100, not historical Tigris/GH200.
Genuine installed-Weaver and selected-site GPU acceptance remain mandatory.
The current migration's debug-to-tier3 profiling exception is narrowly scoped;
it does not automatically authorize an auxiliary-study profile or fits on debug.

## 3. Dataset, selection, and fixed roles

### 3.1 Frozen source and training profile

Use the already frozen September 10, 2026 Delphes snapshot and `TRAIN_500K`:

```text
333 ROOT files; 25,430,379,343 raw bytes
training:           500,000 jets
outer validation: 1,000,000 jets
final test:       1,000,000 jets, sealed

registry content hash:
72e8b76555e3707e90aa7ea46d521c93a2b801e46ee94c35ecee978e324d7d01

outer validation membership:
8f04f54797f7cd181812815eb409b90cb1cf7e503c322061e4718d23f06060b5

final-test membership:
995f747b09a47c2e7cbdc996ba552542562ef0e074b3cb7a9de2b87f7438106e
```

These are content identities, not permission to trust a file solely because its
name matches. Authenticate the registry, exported TRAIN_500K profile, raw
inventory, selected ROOT cycles, schema, and source bytes before use. Bind the
actual training membership hash from the validated profile; do not invent it.
The registry is not modified by this study.

Keep `hlt_matched=True`, the existing supported-label/source-selection policy,
natural selected class proportions, no oversampling, and unweighted CE. No new
kinematic cuts, count cuts, or rejection of difficult offline targets is allowed.
No unused validation/test rows may become extra training rows.

Class order is:

```text
QCD, X_bb, X_cc, X_ss, X_qq, X_gg, X_ee, X_mm,
X_tauhtaue, X_tauhtaum, X_tauhtauh
```

Plots and reports use `X_bb`, `X_cc`, and `X_qq`, not a silent relabelling as
the old CMS benchmark's Hbb/Hcc/Hqq classes. Old FullSim metrics and recovery
denominators are not comparable references.

### 3.2 Study-local validation subdivision

Create a new immutable subdivision **inside** the fixed 1M validation set:

| Study role | Exact rows | Permitted use |
| --- | ---: | --- |
| `TRAIN` | 500,000 | Gradients, target normalization, preprocessing diagnostics |
| `VAL_SELECT` | 200,000 | Every-pass classification validation, early stopping, checkpoint and lambda selection |
| `VAL_REPORT` | 800,000 | One post-lock comparison of the twelve confirmation checkpoints |
| `FINAL_TEST` | 1,000,000 | No particle/model/auxiliary-target access in this campaign |

`VAL_SELECT` and `VAL_REPORT` must be disjoint and their union must equal the
original validation membership exactly. They do not replace its identity in
the shared registry. Fit jobs receive a VAL_SELECT capability, never a generic
validation capability that can silently read all 1M rows. VAL_REPORT inference
requires the reporting lock in section 9.

Deterministic selection, independent of model scores and offline particles:

1. Count each class in the authenticated outer validation membership.
2. Reserve one VAL_SELECT row per class. Apportion the remaining
   `200000 - 11` slots proportionally to the remaining class capacities using
   exact integer Hamilton remainders, ties by class index.
3. Within each class, rank rows by SHA256 of UTF-8
   `JC2/offline-aux/v1/val-select/20260912/{outer_validation_hash}/{row_digest_hex}`.
   Hex is lowercase. Break digest ties by canonical inventory index and entry.
4. Take each class's quota; assign all remaining outer-validation rows to
   VAL_REPORT. Require every class in both slices, else fail before fitting.
5. Publish explicit per-file packed entry masks, class counts, ordered identity
   digests, algorithm/seed, and both membership hashes. Replay with different
   chunk sizes must reproduce the masks exactly.

This is an explicitly **row-stratified internal validation split**, not a new
claim of event independence. Outer train/validation/test file separation is
preserved. Without producer event/group identifiers, related jets can occur in
both internal validation slices, including in the same file. Report that
limitation; held-back comparison is not synonymous with independent final test.
Known cross-slice event overlap discovered before fitting requires a versioned
group-aware revision, not an in-place change to these masks.

The other chat/campaign may use the outer validation set. Record known prior
exposure; do not describe VAL_REPORT as globally untouched or publication-final.
No VAL_REPORT scores, auxiliary diagnostics, or particle distributions from this
study may influence its target design, lambda selection, or checkpoint choice.

### 3.3 HLT-only forward pass

Use native, unmodified HLT constituents with the existing 17-feature transform,
own-HLT constituent-sum axis, own p4, mask, native ordering, and capacity 240.
Pad per batch to its maximum length, at least 16; never truncate. Disable
Weaver random sequence trimming as in the migration model.

Offline axes, species, counts, labels, row digests, file names, source indices,
and matching metadata are forbidden classifier inputs. Joining an offline
target by row identity is allowed outside the model; passing that identity to
the backbone is not. Existing finite displacement/zero-error input handling
is unchanged, but impact-parameter significances are not auxiliary targets.

## 4. Exact offline target construction

### 4.1 Common definitions and numerical rules

Use all valid native offline constituents of the paired selected jet. Do not
use generator particles, unassociated event particles, repaired U/D views,
producer-relative angles, or the HLT constituent count as offline content.
No constituent correspondence or offline classifier is required.

Use the reader's canonical FP32 stored four-vectors and exclusive five-way PID
flags, promoted to FP64 for target calculations and accumulation. Define:

```text
p_i = (px_i, py_i, pz_i, E_i)
pT_i = hypot(px_i, py_i)
S = sum_i pT_i
z_i = pT_i / S
P = sum_i p_i
jet_pT = hypot(P_x, P_y)
eta_i = asinh(pz_i / max(pT_i, 1e-8 GeV))
phi_i = atan2(py_i, px_i)
eta_J, phi_J = the same definitions applied to P
wrapped_phi(a) = (a + pi) mod (2*pi) - pi
r_i = hypot(eta_i - eta_J, wrapped_phi(phi_i - phi_J))
d_ij = hypot(eta_i - eta_j, wrapped_phi(phi_i - phi_j))
```

Require a nonempty jet, finite required data, positive constituent pT/energy,
positive S, valid PID flags, and existing reader validity rules. Invalid data
fail closed; valid large angles or unusual species composition do not. The
small positive denominator floor matches the input convention and is recorded
whenever it activates. Do not turn it into a selection cut.

Targets are permutation-invariant mathematically. Compute in deterministic
native-index order for reproducibility; permutation tests require numerical
agreement, not necessarily identical floating-point summation bytes. Worker
count/chunk size must not alter per-row canonical output bytes or row order.
Persist target arrays as FP32 after FP64 construction; all target gradients
are disabled. Histograms have explicit overflow bins, never discarded tails.

### 4.2 Composition: ten target numbers

Fixed species order:

```text
charged_hadron, neutral_hadron, photon, electron, muon
```

For species k:

```text
N_k = number of offline constituents in species k
c_k = log1p(N_k)
f_k = sum_{i in species k} z_i
```

The five N_k sum to the actual offline count. The five f_k sum to one.
Fractions divide by **scalar constituent pT sum S**, not vector jet_pT.
Use the supplied flags, not truth ancestry or charge-only species guesses.

Predict five standardized c_k scalars and a five-way fraction distribution.
Do not add redundant losses for total count, charged count, and neutral count:
they can be derived for diagnostics, avoiding arbitrary repeated weighting.
Rare electron/muon counts remain part of the fixed group; do not remove a
head after observing its classification effect.

### 4.3 Structure: eighteen target numbers plus one mask

**Two scalars:**

```text
m2 = E_J^2 - (P_x^2 + P_y^2 + P_z^2)
tol_m2 = 1e-6 * max(E_J^2, |P|^2, 1 GeV^2)
if -tol_m2 <= m2 < 0: m2 = 0, with clamp count recorded
if m2 < -tol_m2: fail invalid target input
m = sqrt(m2)
mass_target = log1p(m / max(jet_pT, 1e-8 GeV))
width = sum_i z_i * r_i
width_target = log1p(width)
```

The tolerance handles rounding-scale negative mass squared, not arbitrary
spacelike inputs. No additional winsorization, tail removal, or fitted kinematic
cut is allowed. Normalized mass is still a potentially class-informative mass
proxy; normalization is not proof of mass/pT decorrelation.

**Eight-bin radial profile:**

```text
edges = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.65, +infinity]
R_b = sum_i z_i * indicator(r_i in bin b)
```

**Eight-bin pair-angle profile:**

```text
edges = [0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.65, 0.90, +infinity]
W = sum_{i<j} z_i*z_j
A_b = sum_{i<j} z_i*z_j * indicator(d_ij in bin b) / W
```

All bins are left-closed/right-open; the last includes every finite angle at
or above its left edge. Distances are raw delta-R, with no inferred jet-radius
normalization. Store the final edge as a named overflow boundary, not JSON Inf.
Do not include i=j self-pairs or both orders of an unordered pair.

For one-constituent jets, W=0: write `pair_valid=False` and an all-zero A vector;
mask only the pair-profile loss. Other targets and classification remain valid.
For at least two positive-pT constituents, W must be finite and positive; no
epsilon-defined artificial distribution replaces a failed calculation.

Radial fractions and valid pair fractions must be nonnegative and sum to one
within absolute FP32 tolerance 2e-6. Use bounded per-jet O(N^2) work, never an
all-dataset pair tensor. Pair-angle histograms discard information, including
higher-order geometry; do not claim they reconstruct full substructure.

### 4.4 Train-only scalar normalization

Fit seven independent means and population standard deviations on exactly the
500k training jets: five log counts, transformed mass, transformed width.
Fit from the canonical persisted FP32 target values promoted to FP64, in
canonical row order, with population variance (ddof=0). This avoids fitting
against higher-precision values different from the bank the trainer consumes.
For each scalar j:

```text
scale_j = max(std_j, 1e-3)
t_j = (raw_transformed_target_j - mean_j) / scale_j
```

Record the means, original standard deviations, applied floors, training
membership hash, target-definition hash, source hashes and accumulator
algorithm. Freeze before any scientific fit. A constant target is reported,
not grounds to delete a task. No class-conditioned normalization or fitting on
either validation slice is allowed. Distributions are not standardized.

Targets store the pre-standardized transformed scalars and fractions; the
shared normalizer is applied at batch assembly. For count diagnostics, invert
the transform with expm1 and clamp negative predicted counts to zero only in
the labelled physical-count diagnostic, never in the regression loss. Report
clamp frequency. Do not round predictions before computing count MAE.

### 4.5 Excluded targets in revision 1

Absolute offline jet pT/energy, impact-parameter significance, vertex labels,
track origin, truth subjets, per-particle reconstruction, arbitrary pseudo-
correspondence, and learned offline representations are excluded. These need
separate scientific definitions and comparisons. Zero displacement errors do
not prevent constructing the registered p4/PID summaries.

## 5. Model and auxiliary heads

Build the same from-scratch `DelphesParticleTransformer` as the CE reference:
17 features, four pair inputs, embedding [128,512,128], pair embedding [64,64,64],
eight attention heads, eight particle blocks, two class-attention blocks, GELU,
11 class logits, no random trimming. Pin the full resolved Weaver configuration,
installed source bytes, and dropout defaults; partial architecture lists alone
are insufficient provenance.

Tap the final normalized 128-dimensional jet/class-token representation that
feeds the existing class classifier. Obtain class logits and that representation
from **one backbone forward**. The auxiliary path must not change the class
head, insert a new shared bottleneck, use class labels as inputs, or detach the
shared representation before its auxiliary heads.

Five independent heads, instantiated only where used:

| Head | Output width | Used by |
| --- | ---: | --- |
| Counts | 5 unconstrained standardized scalars | COMP, BOTH |
| Composition fractions | 5 logits, softmax in loss/diagnostics | COMP, BOTH |
| Structure scalars | 2 unconstrained standardized scalars | STRUCT, BOTH |
| Radial profile | 8 logits | STRUCT, BOTH |
| Pair profile | 8 logits | STRUCT, BOTH |

Each head is `Linear(128,128) -> GELU -> Linear(128,out)`, with bias in both
linears, no dropout, normalization, residual path, or learned uncertainty weight.
Use pinned PyTorch Linear reset semantics with each head's separate seed.
Identically named heads start identically across lambda values and BOTH versus
the corresponding single-group arm for a given replicate.

The installed Weaver adapter must expose this surface without copying an
unverified forward implementation. An explicitly tested feature-return adapter
or scoped classifier-input tap is acceptable; hooks must be removed and cannot
accumulate across batches. This is an implementation choice only if forward
and gradient parity hold. Do not persist hidden embeddings.

Selected checkpoints retain the backbone, class head and active auxiliary
heads as one selected-weight payload for diagnostics. Ordinary classification
loading extracts only backbone/class-head state in RAM; deployment must not
require target files or auxiliary modules. A separate stripped weight file is
optional future packaging, not another mandatory checkpoint copy per fit.

## 6. Exact losses and weight screen

All losses are FP32 under BF16 model forward; targets are detached. There is
no label smoothing, class weighting, teacher temperature, or KD term.

For scalar residual e, use Huber with delta=1:

```text
h(e) = 0.5*e^2                 if abs(e) <= 1
       abs(e) - 0.5           otherwise
```

Average over target coordinates and batch rows, not a sum over five counts.
For a target fraction vector q and predicted logits a:

```text
KL(q || softmax(a)) = sum_{k:q_k>0} q_k * (log(q_k) - log_softmax(a)_k)
```

Use stable log-softmax, sum over categories/bins and average over rows. Do not
divide the distribution KL by its number of bins, invent pseudocounts, or take
log of zero targets. Nonfinite logits/loss/required gradients fail closed.

Define:

```text
L_CE = mean row-wise class cross-entropy
L_count = mean Huber over rows and five normalized counts
L_fraction = mean composition-fraction KL over rows
L_comp = (L_count + L_fraction) / 2

L_scalar = mean Huber over rows and two normalized structure scalars
L_radial = mean radial-profile KL over rows
L_pair = mean pair-profile KL over pair_valid rows
         (exactly zero with no head gradient if no valid row in this batch)
L_struct = (L_scalar + L_radial + L_pair) / 3
```

The structural denominator remains three when a batch has no valid pairs.
This rule does not drop any jet or renormalize the remaining structural tasks.
Record valid-pair counts so loss reduction can be audited.

| Arm | Objective |
| --- | --- |
| CE | L_CE |
| COMP | L_CE + lambda * L_comp |
| STRUCT | L_CE + lambda * L_struct |
| BOTH | L_CE + lambda * (L_comp + L_struct) / 2 |

Lambda is constant for the entire fit. Discovery grid is the exact rational
set `{1/10, 3/10, 1}`. The BOTH division prevents simply doubling its nominal
auxiliary budget. It does not equalize actual gradient magnitudes; different
groups may prefer different lambdas. No automatic loss schedules or manual
mid-run reweighting are permitted.

Report all constituent losses and their coefficients. On the first training
batch of passes 1, 3, 10, 30, 60, 75, 90, and 100 when reached, measure the
norms and cosine between CE and weighted auxiliary gradients **at the shared
jet representation**. Use the same forward graph without applying an extra
optimizer update or overwriting accumulated parameter gradients. Label these
as representation-gradient diagnostics, not whole-backbone gradient norms.
Zero-norm cosines/ratios are null, not execution failures. Diagnostics do not
adapt the objective or determine which models are evaluated.

## 7. Training recipe and paired randomness

### 7.1 Common optimizer and schedule

Every scientific fit uses the complete 500k membership once per pass, without
replacement, including the last partial batch. One process and one A100; global
batch 256; no gradient accumulation, DDP, LR scaling, or automatic batch change.
There are `ceil(500000/256)=1954` updates/pass; the last batch has 32 rows.
Loss means use the actual current batch size. No class-balanced sampling.

Use AdamW, betas (0.9,0.999), epsilon 1e-8, weight decay 0.01; exclude only the
canonical class token from weight decay as in the baseline adapter. Auxiliary
parameters use the same optimizer, LR and weight-decay policy. No gradient
clipping is added. Resolve and record optimizer backend flags consistently
across all four arms; do not allow an arm-dependent fused/foreach policy.
Revision 1 explicitly uses `foreach=False`, `fused=False`, `amsgrad=False`,
`maximize=False`, `capturable=False`, and `differentiable=False` for every arm.
This pins execution rather than inheriting an installed default that might
vary with parameter-group composition. Profile this exact optimizer path.

Let u be the one-based completed optimizer-update number and x=u/1954. Set
the LR immediately before that update:

```text
0 < x <= 3:   3e-4 * x/3
3 < x <= 45:  3e-4
45 < x <= 60: 1.5e-5 + (3e-4 - 1.5e-5)/2 * (1 + cos(pi*(x-45)/15))
60 < x <=100: 1.5e-5
```

Maximum 100 passes; minimum 60; VAL_SELECT classification validation every
completed pass. BF16 forward and FP32 losses, ordinary full-precision AdamW
states in RAM. No optimizer resets at schedule boundaries, fine-tuning stage,
head-only warmup, or CE-only tail. All trainable parameters learn from pass one.

This matches the existing schedule mathematics, not a claim that it is optimal
for Delphes or auxiliary learning. All arms have the same budget and stopping
rule but need not complete the same number of passes. Report realized compute.

### 7.2 Checkpoint selection and stopping

After each pass, rank its VAL_SELECT classification metrics lexicographically:

1. Higher full-precision macro 11-class OVR AUC.
2. Lower class cross-entropy.
3. Higher macro log-R50; null/censored sorts below finite for this tie-break.
4. Earlier optimizer update.

Keep the exact best weights in RAM, including auxiliary heads. AUC differences
are not rounded before selection. Auxiliary losses, auxiliary quality, training
loss and VAL_REPORT are never checkpoint-selection criteria.

Maintain a separate patience reference: first validation initializes it. Reset
its pass and AUC only on `current_auc > reference_auc + 5e-5`. Stop after any
pass >=60 if `pass - last_significant_pass >=15`; otherwise stop at 100. Patience
accumulates before pass 60, not starting at 60. Restore the exact selected
checkpoint even if its AUC improvement was below the patience threshold.

Persist one selected weight payload at successful completion. Keep final-pass
metrics in the report, not final-pass weights. No rolling, last, optimizer,
resume, or per-pass checkpoint files. A crash loses the fit's progress; recovery
starts it from zero, preserving the original failed attempt for audit.

### 7.3 Seed register and RNG isolation

Replicate IDs: `DISCOVERY`, `CONFIRM_01`, `CONFIRM_02`, `CONFIRM_03`.
These designate stochastic replicates, not data resplits.

Define an unsigned 32-bit seed as the first four bytes, big-endian, of SHA256
of the exact UTF-8 string:

```text
JC2/offline-aux/v1/{replicate}/{domain}
```

Domains are `shared_init`, `sampler`, `train_rng`, and one of
`aux_init/counts`, `aux_init/fractions`, `aux_init/scalars`, `aux_init/radial`,
`aux_init/pair`. Record actual integer values in the campaign spec. Reject
accidental collisions between independent replicate domains rather than
silently accepting duplicate replicates.

Within a replicate, omit arm, lambda, job ID, physical path, and target values
from shared initialization, sampler and training-RNG seeds. All four arms start
from bit-identical backbone/class-head state, checked by an initial-state hash.
Construct auxiliary modules in isolated/forked RNG contexts so they cannot
consume the common model's RNG stream. Same head means same initial head state
in BOTH and the single-group arm.

At each pass p, use
`np.random.default_rng(np.random.SeedSequence([sampler_seed,p])).permutation(500000)`.
At pass start, seed Python, NumPy global RNG, and Torch CPU/CUDA training RNG
from the same hash algorithm with domain `train_rng/pass/{p}`. Auxiliary heads
contain no stochastic layers; diagnostics and validation cannot advance the
next pass's training stream. Bounded preprocessing is deterministic and cannot
draw from the training RNG. Record per-pass order hashes.

Matched seeds control stochastic variation; they do not guarantee bit-identical
trained weights on different hardware or after different objectives change the
optimization trajectory. Pin the numerical environment and deterministic policy
and report any nondeterministic kernels rather than promise universal exactness.

## 8. Stage A: ten discovery fits and configuration lock

Scientific task order:

```text
DISC_CE
DISC_COMP_L010     DISC_COMP_L030     DISC_COMP_L100
DISC_STRUCT_L010   DISC_STRUCT_L030   DISC_STRUCT_L100
DISC_BOTH_L010     DISC_BOTH_L030     DISC_BOTH_L100
```

All ten use DISCOVERY seeds and fresh weights. After common data/target/source
and actual-GPU acceptance gates pass, they may train independently in parallel.
No fit depends on the CE baseline score or completion. The baseline supplies a
matched reference, not a condition for declaring other fits operationally valid.

Only VAL_SELECT may be scored. Select exactly one lambda for each auxiliary
arm using its selected checkpoint's classification selection tuple in section
7.2, then lower lambda if all fields tie. Do not select by visual preference for
R50, by auxiliary fit quality, or by improvement over baseline. Even if all
three settings lose to CE, the highest-ranked one advances and that negative
discovery outcome remains in the report.

Publish `CONFIGURATION_LOCK/v1` binding all ten successful fit/report/weight
hashes, the ordered grid, exact metrics and selection rules, the three chosen
lambdas, both validation masks, normalization, seed register, and scientific
source hashes. Missing/invalid fits block this lock until exact recovery; poor
scores do not. This lock cannot be edited after confirmation begins.

## 9. Stage B: twelve confirmation fits and controlled reporting

Run `CE`, selected `COMP`, selected `STRUCT`, and selected `BOTH` under each
of CONFIRM_01/02/03: twelve fresh fits. They share the same TRAIN and VAL_SELECT
rows as discovery but use independent replicate seed domains. Each starts from
scratch; no discovery checkpoint is loaded. The twelve fits can run in parallel
after CONFIGURATION_LOCK. Discovery results are not pooled into confirmation
means, standard deviations, or confidence intervals.

All twelve use VAL_SELECT for checkpoint selection and stopping. **No
VAL_REPORT scoring occurs until all twelve selected weight hashes are frozen.**
Publish `REPORTING_LOCK/v1` binding these checkpoints, configuration lock, metric
definitions, pT-bin definitions, comparison membership and evaluation source.
It permits exactly those checkpoints on VAL_REPORT; it does not unlock test.

Then run twelve independent HLT inference/evaluation jobs, one per checkpoint,
on the same 800k rows. Retain compact classification probability shards for
paired statistics; no ensemble combination. The auxiliary diagnostics
specified below use the active selected heads and a separately gated compact
VAL_REPORT target bank, never training or parameter updates.

Aggregate all twelve rows and finish independently of whether the hypothesis
succeeds. A deterministic evaluation may be recovered with identical parents
without reselection. Once VAL_REPORT is inspected, any new targets, lambda
values, or schedules are a new exploratory study; this slice is no longer a
fresh confirmation set for those choices. Final-test execution needs a separate
approved finalist/execution protocol and remains unimplemented here.

### 9.1 What is and is not counted as a fit

```text
Stage A:  1 CE + 3 arms * 3 lambdas = 10 scientific fits
Stage B:  4 arms * 3 new seeds     = 12 scientific fits
Total:                              22 scientific fits
```

Preparation, GPU miniature/profile, lock/reduction tasks, and selected-model
inference are additional jobs, not additional scored fits. No stage-C retrain,
all-1M-validation reselection, best-seed selection, offline reference fit, or
test evaluation is hidden in these totals. Authorize/dry-run the two science
stages separately because Stage B's exact lambda commands require Stage A's
immutable lock. Do not auto-submit another campaign when a gate completes.

## 10. Metrics, uncertainty, and interpretation

### 10.1 Primary and secondary classification measurements

Reuse the 11-class Delphes metric definitions without old class-count defaults:

- Primary: unweighted macro OVR AUC across all eleven classes.
- Secondary: accuracy, CE, macro R50, and per-signal QCD R50 for all ten signals,
  highlighting X_bb, X_cc and X_qq without hiding the others.
- Report both achieved signal efficiency and background passing/total counts.

For a signal, score only its signal and QCD rows with
`p_signal/(p_signal+p_QCD)`; both-zero is score zero. The 50% signal threshold
is descending rank `ceil(N_signal/2)`, ties included. Zero passing QCD yields
null/censored rejection plus the labelled one-event resolution, not JSON
infinity. Macro R50 is the geometric mean over ten signals and is null if any
component is censored. A scientifically censored metric is not a failed task.

Per replicate, report each auxiliary arm minus **its same-seed CE** in raw
accuracy, AUC, and linear R50. Across the three confirmation replicates report
mean paired delta, sample standard deviation (ddof=1), individual deltas, and
the number of seeds improved. Do not compare the best auxiliary seed against
the average CE seed or present the standard error as seed standard deviation.
Report improvements of STRUCT versus COMP and BOTH versus each as secondary,
not three extra primary claims selected after seeing scores.

No offline-gap recovery percentage is defined by these 22 fits: they contain
no matched fresh offline oracle. Print `recovery: not evaluated` rather than
borrowing an old U000 or a differently selected baseline. An optional later
authenticated same-data oracle comparison must name its population, schedule,
selection slice, seeds and limitations and is not a dependency of this study.

### 10.2 Kinematic and auxiliary diagnostics

Freeze five HLT jet-pT bins at the TRAIN HLT-pT 20/40/60/80 percentiles using
FP64 NumPy linear quantiles, with unbounded outer intervals. Reject duplicate
interior edges before fits rather than silently changing the bin count.
HLT pT is the native HLT constituent vector-sum pT, never offline pT. Store the
actual edges in the preparation lock. Report classification metrics and class
counts in each bin; metrics with missing classes are null with a reason.
These bins are diagnostic and do not affect selection or training weights.

For every active auxiliary family, report transformed scalar MAE/RMSE,
physical count MAE, and distribution KL; retain pair-valid counts. CE and
inactive heads have no learned auxiliary prediction: report not applicable,
not an invented zero loss. Compare active predictions against:

1. A constant predictor using TRAIN target means (distribution means on valid
   rows; transformed means for standardized scalars).
2. The exact same observable computed from native HLT constituents, with the
   offline TRAIN normalization applied when comparing transformed scalars.

These are nontrained diagnostic predictors, not extra classifier arms. For
HLT pair-profile diagnostics require both the offline target and HLT profile to
be valid, report the joint count, and never substitute a fabricated single-
particle profile. Prediction diagnostics do not introduce an HLT-target
training control implicitly. Good target prediction is not a pass/fail gate.

### 10.3 Finite-sample uncertainty versus seed variation

Primary robustness evidence is the three matched-seed deltas, with the clear
limitation that three seeds provide a coarse estimate. Do not infer a universal
or seed-independent gain from a very narrow jet bootstrap interval.

After REPORTING_LOCK, run 1,000 deterministic paired **source-file cluster**
bootstrap resamples on VAL_REPORT using the same file multiplicities for all
models and seeds. Seed domain: `JC2/offline-aux/v1/report-bootstrap/20260912`.
Use the same SHA256-to-uint32 conversion as section 7.3. Sample the F distinct
source files F times with replacement per replicate, include all selected rows
of each drawn file with that multiplicity, and recompute metrics on that
resample. Implement multiplicity-weighted ranks/counts or bounded RAM row
expansion with numerical equality tests; no disk-expanded predictions.
Reuse stable per-model score orderings/tie groups across resamples where valid,
and bound concurrent resamples and RAM. Profile the statistics stage separately;
its CPU cost is not included in the 22 fit count or hidden in GPU walltime.

Report percentile 2.5/97.5 bounds for the mean paired delta conditional on the
three fitted seeds. This is not an interval accounting for all possible training
seeds. Also provide per-seed intervals if inexpensive. A resample missing a
class or with censored R50 produces a null for that metric; do not redraw until
it becomes convenient. Report valid replicate count; suppress an interval with
fewer than 900 valid values. Never convert zero-background censoring into an
arbitrarily large finite value for a confidence interval.

Files are the provisional event-containment units, not established independent
generation units. File bootstrap is a dependence sensitivity analysis, not proof
of cross-file event independence or correction for selection/report correlation.
No formal superiority claim based solely on these intervals or three seeds.

### 10.4 Interpretation rules

Report all arms, lambdas and seeds. Discovery is selected-on-validation;
confirmation uses new random seeds and study-held-back validation jets, with
the event/prior-exposure caveats above. AUC and R50 can disagree legitimately.
Call small mixed gains inconclusive, not a failure of data integrity. Improvements
on this reconstructed simulation do not establish detector-generalization or
publication rights. Offline-derived auxiliary supervision is not necessarily
the best possible supervision, even if all three tested arms fail.

## 11. Data flow, artifacts, and storage

Minimal execution flow:

```text
frozen raw inventory + TRAIN_500K profile
    -> study role masks + target definition
    -> compact TRAIN/VAL_SELECT targets + TRAIN normalization/pT bins
    -> data/source acceptance + actual A100 model/resource acceptance
    -> 10 discovery fits -> CONFIGURATION_LOCK
    -> 12 confirmation fits -> REPORTING_LOCK
    -> gated VAL_REPORT target preparation and 12 selected-model evaluations
    -> paired aggregate + completion
```

HLT caches are native HLT only, process-local RAM, built once per fit. Load only
TRAIN and VAL_SELECT caches during fitting. VAL_REPORT cache exists only in
post-lock evaluation jobs; no reason to hold all 1.5M ordinary HLT rows in every
fit. Reuse authenticated read-only source/profile metadata, not old train
caches or fabricated matching locks. A new lightweight native-view cache binding
must avoid requiring assignments merely because the ladder's cache did.

Targets are computed once per role and shared read-only by all fits; they are
compact derived labels, not hidden-state banks. Persist 28 FP32 targets per row
(ten composition, eighteen structure), one pair-valid byte and one 32-byte
identity digest. Array payload is **145 bytes/row**: about 101.5 MB for
TRAIN+VAL_SELECT, or 217.5 MB after all 1.5M ordinary targets exist. Shard at
most 100,000 rows (14.5 MB payload); use a separate small manifest for counts,
checksums, role, shapes, schema and lineage. Avoid per-row JSON.
The 28-column order is five log counts, five species fractions, mass_target,
width_target, eight radial fractions, then eight pair fractions, each in the
orders defined in section 4. Pair validity and row identity are separate arrays.

Use the existing scalar labels/identity catalog with an authenticated join;
do not trust array position alone or duplicate raw particles. Write once via
temporary-file/fsync/atomic publication, validate before reuse, count both
temporary and final files in storage estimates. Temporary compact publication
is allowed; particle/hidden-state/optimizer spills are not.

Selected VAL_REPORT probability payload across twelve fits is approximately
`12 * 800000 * (11*4 + 32) = 729600000` bytes before metadata, plus shared labels
and pT-bin membership. Persist no per-pass validation probability banks or
discovery prediction matrices; selected/full histories are compact metrics.

Default storage envelope: projected steady generated artifacts <=4 GiB and
no individual generated file >64 MiB, with raw ROOT files excluded from that
generated budget. This is a preflight constraint, not a measured claim. Measure
the selected model/head checkpoint size and aggregate/temporary footprint;
refuse an incompatible envelope rather than silently exceed it. Require free
space for twice projected generated bytes plus 1 GiB. Count retained failed
attempts and other existing output in subsequent checks. No broad cleanup or
deletion of another campaign is authorized.

No rolling resumes, optimizer files, full particle reconstruction banks,
per-jet dense pair matrices, retained backbone embeddings, ROOT copies,
per-pass weights, or memmap fallbacks. Compact targets/probabilities are the
explicit exceptions to RAM-only transient state. Logs include phase/progress
and pass summaries, not per-particle or per-update dumps.

### 11.1 Proposed new artifact families (not implemented by this MD)

Use the prefix `JETCLASS2_DELPHES_OFFLINE_AUX_`, all initially `/v1`:

| Family | Required bindings |
| --- | --- |
| `STUDY_SPEC` | Plan revision, stages, arms/grid/seeds, source, profile, class/input/target/recipe identities |
| `ROLE_SPLIT` | Outer membership and exact select/report masks, access policy, provisional event-group status |
| `TARGET_DEFINITION`, `TARGET_SHARD`, `TARGET_BANK` | Equations/bin edges/dtypes, role/source/identity joins, checksums and complete coverage |
| `NORMALIZATION`, `PREPARATION_LOCK` | TRAIN-only moments, diagnostic predictors, pT bins, all preparation/source parents |
| `MODEL`, `RECIPE`, `EXECUTION_ACCEPTANCE` | Shared/tap/head contracts, RNG/optimization, installed-Weaver parity and real measured A100 evidence |
| `STAGE_SPEC`, `COMMAND_PLAN`, `SUBMISSION_LEDGER` | Exact concrete discovery/confirmation/evaluation commands, explicit authorization, actual job IDs |
| `TRAINING_REPORT`, `SELECTED_WEIGHTS` | Initial/selected state hashes, passes/history, losses, gradients, source and role/recipe lineage |
| `CONFIGURATION_LOCK`, `REPORTING_LOCK` | Deterministic lambda choice and frozen twelve-checkpoint report capability respectively |
| `EVALUATION_REPORT`, `AGGREGATE`, `COMPLETION` | Raw counts, metrics, paired statistics/caveats, complete registered coverage independent of score |
| `MONITOR`, `RECOVERY` | Exact attempts/terminal states, output authentication, unfinished-task restart from zero |

Artifacts have canonical content hashes, explicit parent hashes, producer source
lineage and atomic immutable publication. Timestamps are metadata, not scientific
identities. The actual implementation must add validators/tests and a reusable
contract companion before claiming any family is executable.

## 12. Site, acceptance, and non-interference

Production intent:

```text
submission host: sporcsubmit
partition:      tier3
account:        reu-aisocial
QoS:            qos_tier3
GPU:            one A100 per fit/evaluation, one node and one process
Conda prefix:   /home/ryreu/miniconda3/envs/atlas_kd_sporc
raw data:       /home/ryreu/atlas/datasets/jetclass2_10M_20260910/jetclass2
```

Use actual validated site metadata, not an assumption that resources are free.
Begin the nonscientific A100 profile with 8 CPUs/workers, 72 GiB host RAM and
four hours; these are **unmeasured starting requests for this study**. Measure
native HLT cache residency, worker/IPC copies, the BOTH model and backward at
capacity 240/batch 256, full 500k training pass and 200k validation pass, and
800k inference cost using permitted training/select timing extrapolations before
the reporting lock. Do not read report particles early just to benchmark them.
Real ordinary-role profiling must not fit normalization or learn from report/test.

Reserve at least 25% host RAM after accounting for persistent caches, targets,
workers, optimizer, selected RAM checkpoint and temporary batches. Keep worst-
capacity measured GPU peak within 85% of available GPU memory. Failing a
resource gate authorizes reporting the failure, not silent data truncation,
batch reduction, view spilling or target removal.

Let C be measured fit preparation seconds and P the conservative measured
training-plus-VAL_SELECT seconds/pass for the active objectives. Request at
least `ceil(1.75*(C+100*P)/60)` minutes, minimum 60, initial ceiling 2880 minutes.
For evaluation use twice conservative preparation+800k inference time, minimum
30 minutes. Excess above the site/user envelope blocks submission for an
explicit resource decision; do not clamp the estimate down. Measure CPU target
preparation on bounded TRAIN data and select finite per-shard requests before
materializing its dry run. No promise that all fits occupy GPUs simultaneously.

Every worker verifies source and allocation, sets `PYTHONNOUSERSITE=1` and
`PYTHONDONTWRITEBYTECODE=1`, prepends the chosen Conda lib to LD_LIBRARY_PATH,
sources helpers through absolute PROJECT_DIR, and isolates inherited paths from
other projects. Bounded process workers use one numerical-library thread each;
there is no uncontrolled multiplicative thread pool.

Required actual acceptance includes:

1. Canonical HLT input byte equality across arms and exact TRAIN/VAL_SELECT
   target identity coverage; reader refusal of test/report during fitting.
2. Installed-Weaver FP32 logits, input/parameter gradients, and state parity
   for the unmodified CE model versus the auxiliary wrapper with auxiliary
   coefficient zero, including training-mode paired RNG checks. Require
   float comparisons at atol=1e-6, rtol=1e-5 and exact masks, parameter names
   and initial shared-state bytes; separately label BF16 checks as nonauthoritative
   for FP32 parity. Do not loosen tolerances after observing a failure.
3. Genuine A100 production-worker forward/backward for all objectives, nonzero
   auxiliary-to-backbone gradient path on a nondegenerate fixture, actual BF16
   finite execution, batch/capacity/resource checks, and selected-state restore.
4. Export/HLT-only inference with offline targets, labels and auxiliary files
   physically absent from the inference interface; perturbing offline data
   must not change fixed-weight class logits.
5. Bounded output inventory proving no dense target, optimizer, resume or
   particle cache was persisted.

A short real acceptance exercise is infrastructure validation, not a scored
pilot or a substitute for any of the 22 fits. Prior migration acceptance may
be reused only for explicitly identical validated subcomponents; it cannot
prove the new head/loss/role gates. Local toy models are not Weaver evidence.

Use exact `afterok` dependencies, `--no-requeue`, campaign-local stdout, durable
submission intent/receipt journals, and canonical dry ledgers. Live submission
requires explicit user authorization for the concrete stage. Lost acknowledgements
block blind retry. Recovery reuses only authenticated completed tasks and starts
unfinished fits at zero after their exact old attempts are terminal; never
cancel/release other jobs by name. No scientific-score threshold can fail a
gate, prevent confirmation, or suppress a registered report.

## 13. Implementation map and provenance

Add isolated reusable modules and thin entry points; names below are reserved
design destinations, not claims that files exist:

```text
src/hlt_classification/jetclass2_delphes/offline_aux/
  contracts.py       # new families and access checks
  roles.py           # deterministic 200k/800k internal masks
  targets.py         # exact p4/PID summaries; bounded CPU work
  normalization.py   # train-only moments/pT bins/constant predictors
  banks.py           # compact target shards and identity validation
  model.py           # parity-preserving shared feature tap and five heads
  losses.py          # explicit group losses and registered coefficients
  training.py        # paired RNG, selected-only weights, live diagnostics
  reporting.py       # gated 800k evaluation and paired statistics
  campaign.py        # two-stage specs, configuration/report locks
  execution.py       # real acceptance, measured allocation and submission
  recovery.py        # exact-ledger restart-zero logic

scripts/jetclass2_delphes_offline_aux.py
sbatch/run_jetclass2_delphes_offline_aux.sh
tests/test_jetclass2_delphes_offline_aux_*.py
docs/contracts/JETCLASS2_DELPHES_OFFLINE_AUXILIARY_SUPERVISION.md
```

Existing reusable surfaces to inspect and adapt narrowly:

| Surface | Reuse boundary |
| --- | --- |
| `jetclass2_delphes/schema.py`, `reader.py`, `splits.py`, `split_registry.py` | Native fields/identities and selected masks; add explicit study-role capabilities, no label/source-policy changes |
| `jetclass2_delphes/inputs.py`, `cache.py` | Native HLT transform and bounded RAM pattern; avoid mandatory constituent-matching dependencies |
| `jetclass2_delphes/model.py` and canonical Weaver factory | Baseline config/parity; add an opt-in shared representation surface without changing old forward behavior |
| `jetclass2_delphes/campaign.py`, `runner.py` | LR/selection/sampling reference mathematics; do not reuse C25P75 loss or generic validation access |
| `jetclass2_delphes/reporting.py` | 11-class metrics/censoring; bind new exact population and paired report rules |
| `jetclass2_delphes/execution.py`, `submission.py`, `provenance.py` | Site/source/journal patterns; new auxiliary acceptance and separate stage authorization |
| `data/cache_contracts.py` | Canonical hashing, atomic publication and checksum primitives |

Repository HEAD inspected while drafting: `82032e35177f83436741d7fa1b9d38fbc4b3efc7`.
This is not a clean-worktree assertion; unrelated work already exists. No donor
source was copied by this documentation task. During implementation, record
actual reused/migrated files, donor commits and any dirty semantic-byte hashes
in LEGACY_SOURCE_MAP rather than treating the drafting HEAD as a future pin.
No Fresh_check or sibling-worktree runtime imports.

## 14. Implementation milestones and required tests

| Stage | Deliverable | Evidence required |
| --- | --- | --- |
| A | Roles, targets, normalizer, compact banks | Synthetic equations/invariants, deterministic masks/joins, sealed-role refusal and source/byte checks |
| B | Model tap, heads and group losses | Exact architecture/initial-state parity, hand losses/gradients, RNG isolation, deployable extraction |
| C | Training and reports | Schedule boundaries, sampler/partial batches, early-stop/best distinction, no-resume outputs, paired metrics |
| D | Staged orchestration | Exact 10/12 fit expansion, deterministic configuration/report locks, no mutation, failure/recovery and storage tests |
| E | Real SPORC acceptance | Installed-Weaver and genuine A100 production-worker evidence, measured resources and absence of forbidden files |
| F | Queue handoff | Exact pushed source, clean worktree, full concrete stage dry run, explicit separate live authorization |

Required focused regression cases include:

- Zero counts for a species; pure-species and mixed jets; fractions summing to
  one; all counts summing to native multiplicity; no filename/label dependence.
- One constituent, collinear constituents, phi wrapping near +/-pi, zero width,
  tiny tolerated negative m2 versus invalid m2, denominator-floor accounting,
  nonfinite inputs, large angles, exact bin-edge and overflow behavior.
- Pair profile against a hand-computed three-particle example, no double/self
  pairs, normalized positive weights, one-particle mask, all-invalid batch loss.
- Positive global p4 rescaling invariance for fractions/normalized structure
  away from declared numerical floors, azimuthal-rotation and particle-
  permutation agreement, bounded worst-capacity target memory.
- Chunk/worker/relocation invariance; no train/selection/report overlap;
  exact 500k/200k/800k membership; failed class-coverage allocation; hash corruption.
- Normalization replay using TRAIN only; constant targets; validation mutations
  cannot change means, scales or pT bins; target/HLT identity permutation fails.
- Huber boundaries, KL with zero target bins, actual-batch reduction,
  BOTH's one-half coefficient, and lambda-zero CE update/gradient parity.
- Identical shared/head initialization across paired arms, sampler parity,
  isolated head RNG, single backbone forward and gradient reachability;
  fresh confirmation seeds and no discovery warm start.
- Exact LR at x=3/45/60/100 and neighboring updates; 1954 updates with last
  32-row batch; min60/max100/patience15; smaller-than-patience-delta best updates
  still retained; null R50 tie-break without scientific failure.
- Configuration lock including all discovery losers; missing reports block but
  poor metrics do not; no report inference before all twelve weights are frozen;
  attempts to supply all 1M validation rows to a fit fail.
- Per-class metric identity/censoring, paired file-bootstrap multiplicity
  equality against explicit repetition, missing-class/censored resamples,
  seed variance separate from conditional sample intervals.
- Real selected-state reload, identical class logits with/without auxiliary
  files, no labels/offline quantities reaching model inputs, no stale hooks.
- Isolated paths, duplicate/ambiguous submission refusal, exact old-attempt
  terminal checks, restart zero, storage/worker bounds, no unrelated job calls.

Run focused local tests before and after each implementation block. Keep actual
Weaver/GPU gates distinct from CPU synthetic tests. Update the handoff with
observed evidence, not an aspirational ready status. Scientific implementation
must also resolve the baseline shared-feature tap against the actual installed
Weaver source; a shape-compatible fake model does not certify it.

## 15. Deferred follow-ups and decisions already frozen

The following are **not blockers for writing this plan**, but real acceptance,
measured walltime/memory, exact source pins and instantiated artifact hashes are
blockers for live submission. Producer event grouping and charged zero-error
semantics remain accurately labelled provisional; this study adds no error-
significance target and cannot settle those producer questions.

After the registered comparison, separately consider:

1. The same auxiliary objectives generated from HLT instead of offline, with
   matched heads/budget, to isolate richer-target benefit from generic multitask
   regularization. The four current arms alone cannot establish that distinction.
2. More confirmation seeds if effects are small or inconsistent.
3. Combining a selected auxiliary method with logit KD, using a factorial
   control rather than adding both and attributing the gain to one.
4. Absolute offline pT calibration, higher-order/particle-level structure,
   alternative loss balancing or schedules, and training-size scaling using
   the same registered outer memberships.

None is automatically launched, inserted into the 22-fit budget, or used to
change this study after VAL_REPORT exposure. Current frozen choice: 500k
training, four arms, fixed 28-number offline summaries, three lambdas, one
discovery replicate, three new confirmation replicates, 60--100-pass common
schedule, HLT-only deployment, compact targets, RAM-only views, and no resumes.

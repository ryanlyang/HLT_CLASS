# U100-supervised particle and pair corrections: four-run debug plan

Date: 2026-11-02.

Status: **additive implementation and debug queue tooling implemented** in
`src/hlt_classification/luka_u100_pair_correction/`. See the
[runbook](../LUKA_U100_PAIR_CORRECTION_RUNBOOK.md) and
[v1 contract](../contracts/LUKA_U100_PAIR_CORRECTION.md). Clean pushed source and
genuine shared-setup resource acceptance are required before scientific fits;
no remote submission or existing-job alteration was performed locally.
Defaults are experimental choices, not measured optima or a promise of recovery.

Implementation clarifications: validation always evaluates final alpha1;
training alpha and rho retain the three-epoch ramps below. At angular
separation below1e-8 the differentiable pair companion keeps the exact native
forward value but stops the square-root derivative. In particular, exact
collinearity keeps native kT=0 before the log floor: flooring delta first would
incorrectly change log(kT) for pt>1. The native Weaver path is unchanged.

## 1. Decision, scope and leading hypothesis

Keep the successful foundation: a fresh HLT-only Particle Transformer trained
with **CE coefficient 1 plus original U100 jet-embedding prediction coefficient
5**. Add a small, retained correction module that uses only HLT observations.
Supervise it with the richer U100 measurements on authenticated corresponding
particle slots. Use its predictions to modify later particle attention.

**All four are additions to U100-w5.** Each keeps CE1 and U100 jet-embedding
prediction5 throughout training; none is a correction-only alternative. The
primary comparison asks whether the new supervision and retained correction
improve AUC and recovery beyond the completed U100-w5 reference.

**Reuse the completed reference; do not rerun it.** Import its authenticated
saved results and compatible existing U100 target bank. The new campaign has
four student fits and no reference-training task, including after a restart.
If reference evidence cannot be authenticated, diagnose it rather than
automatically scheduling a replacement fit.

Run exactly **four new scientific student fits**, all targeting **U100**:

| ID | Short name | Prediction | Extra training supervision |
| --- | --- | --- | --- |
| A | pair_u100 | Direct corrections to four pairwise interaction features | U100 pair features |
| B | p4_u100 | Corrected particle four-vectors; calculate pair features from them | U100 particle kinematics |
| C | p4_pair_u100 | The same particle predictor as B | Particle kinematics AND pair features calculated from its predictions |
| D | dual_u100 | Independent particle and direct-pair predictors | Particle targets for one branch; pair targets for the other |

The critical distinction is **C has one predictor and two losses; D has two
predictors and two losses**. D does not silently become C with another name.

My leading hypothesis is **C, p4_pair_u100**: predict one coherent corrected
particle collection, then require it to reproduce useful pair relationships.
Particle supervision provides local guidance, pair supervision measures the
consequences of those corrections, and classification gradients teach their
use in attention. A is a strong competing hypothesis because a direct pair
predictor need not reconstruct every particle accurately. D tests whether
combining those inductive biases is useful. B isolates the benefit of local
particle supervision alone.

This is a mechanistic hypothesis, not evidence that the added objectives will
beat U100-w5, the contrastive study, or the K1/E40 models. In particular, four
single-seed runs at one new-loss budget cannot establish an optimal architecture
or a high-data recovery improvement.

The user's latest choice of four U100 runs supersedes the earlier discussion
of including D066/D033 and additional oracle/capacity fits in this first screen.
Do not append those jobs, weight sweeps, teacher fits or a fifth baseline fit.
Those remain explicit follow-ups if this screen warrants them.

## 2. Authority and scientific boundaries

This is an additive **small-data SPORC/debug** study. Its inherited scientific
meaning comes from the [direct U100 plan](LUKA_U100_EMBEDDING_PLAN.md),
[U100 contracts](../contracts/LUKA_U100_EMBEDDING.md), and the authenticated
historical Luka ALLJETS population. The [paired identification plan](LUKA_U100_OFFLINE_CONTRASTIVE_PLAN.md)
provides a recent matched U100-w5 reference and reusable execution patterns;
its matching objective is not part of this study.

Differences from the inherited U100 recipe are explicit:

- The main classifier still consumes the final 128-D student jet vector.
- New correction parameters participate in classification and remain deployed.
- The separate U100 jet-embedding prediction head remains training-only.
- No main-logit KD, auxiliary-logit KD, OFFLINE identification, SupCon, fusion,
  concatenation, neighborhood or token-embedding loss is added.
- No student warm start. Completed student checkpoints are comparison evidence,
  not initial weights. Reuse the frozen U100 teacher and its compatible bank.
- U100 **particle measurements** supply the new targets; they are not U100
  learned token embeddings, learned attention maps, or DE033 representations.
- Matching/source indices, labels, row identities, match-validity bits and
  richer tensors are never deployable inputs.
- No independent final test exists for this historical ALLJETS study. Preserve
  its original exploratory and retired-test flags. Original50k overlaps full
  validation and is report-only.

The [D-attention contract](../contracts/LUKA_D_ATTENTION.md) is useful for slot
lineage, but its attention target, all-pair loss, deployment, and submission
requirements are not copied as this new experiment's scientific contract.
The [REPLACE_GATE handoff](../LUKA_REPLACE_GATE_STRATEGY_HANDOFF.md) describes a
related retained attention correction. Its learned gate target, rank-four
factorization, 25/75 CE/KD loss and block placement are different experiments.
Do not mutate either old campaign to implement this one.

## 3. Fixed population, schedule and references

| Item | First-screen setting |
| --- | --- |
| Population | Original authenticated ALLJETS 299,234 TRAIN / 74,895 validation |
| Student view | Exact existing D000/HLT particles, 17 features, four-vectors and mask |
| Target view | Same prepared U100 slots and original regular COARSE_U100 teacher |
| Classes | Existing 11-class order |
| Support | All real particles, capacity 512, original padding; no truncation |
| Native model | Installed-Weaver ParT, 128-D token output, eight particle blocks, eight heads |
| Initialization | Fresh matched historical initialization and epoch-order seed |
| Training | 30 full epochs, physical batch 128, no accumulation |
| Optimizer | Inherit original U100 AdamW groups; betas 0.9/0.999, eps 1e-8, decay 0.01 |
| Learning rate | Exact warmup3/cosine30, peak 3e-4, floor 1.5e-5 |
| Existing losses | Main CE1 and original standardized jet-embedding SmoothL1(beta=1) weight5 |
| New auxiliary budget | Total coefficient 1 at full strength, as defined in section 8 |
| Precision | Existing BF16 native backbone; new heads, geometry, losses and bias construction FP32 |
| Selection | Full-validation deployed-model macro OVR AUC and original deterministic tie-breaks |
| Early stopping | None; finite poor performance completes all 30 epochs |

Import the actual historical recipe and hash its parent; the table is not a
replacement for all optimizer, dropout, ordering and transform details.
Original tie-breaks are lower validation CE, higher macro log rejection at 50%,
then earlier update. Do not substitute earliest epoch for an exact AUC tie if
the original tie-breaks would select a different checkpoint.

Use the original COARSE_U100 teacher behind U100-w5, not c50 UE100, a FullStat
teacher, a retrained model, or a similarly named checkpoint. Discovery hints:

~~~text
/home/ryreu/atlas/HLT_Classification/checkpoints/
  luka_fixed30_74aa58c5_r1/science/train_COARSE_U100/
  luka_u100_direct_ed6d0af4_tier3_r1/
~~~

Authenticate teacher checkpoint/report/receipt, original source, prepared views
and bank bytes before reuse. Current compatible U100 target exports may be
referenced in place with their original parent hashes; do not re-export them
merely because this is a new campaign.

The primary reused comparison is the completed contrastive campaign's
zero-identification arm **u100_w5_s0, job 21993056**. Its user-reported full
validation AUC is **0.96209280, 38.6% recovery**. Historical U100-w5 is
**0.96206974, 38.1%**. Authenticate the campaign/job binding and exact stored
values rather than hardcoding the rounded terminal output.

Historical anchors are approximately M0HLT **0.96056122** and OFFLINE
**0.96452569**. The reported mixed-identification result **0.96232198, 44.4%**
is useful context, but it uses another objective and is not this study's
matched baseline. None of these is a result on the separate 3M/500k population.

## 4. Correspondence, targets and missing information

Use the existing prepared matching assignments and persistent HLT slot order
from [views.py](../../src/hlt_classification/cms_proxy_ladder/views.py).
U100 replaces matched HLT slots with their corresponding offline measurements,
retains unmatched HLT slots, and removes the unmatched offline tail.

This is **constructed correspondence**, not guaranteed generator-level
identity between detector objects. Equal particle counts alone prove nothing.
Authenticate per-jet counts, masks, ordered identities, assignment hashes and
view-builder source. Never independently sort or rematch teacher/student slots.

For TRAIN jet j, define supervision-only masks:

~~~text
particle_supervision[i] = real_HLT_slot[i] AND recorded_mapping[i] >= 0
pair_supervision[i,k] = particle_supervision[i] AND particle_supervision[k] AND i < k
~~~

Keep all real HLT particles in the encoder and classifier, including unmatched
ones. At deployment the predictor runs on **all real HLT slots/pairs** and is
not told which were matched during training. Unmatched pairs are therefore
unsupervised extrapolation for the correction branch; report their behavior.
Do not use the supervision mask to gate their inference correction.

A jet with zero matched particles contributes zero new particle loss; a jet
with fewer than two matched particles contributes zero new pair loss. It still
contributes CE and U100 jet-embedding loss. Average per-jet losses across all
jets, counting these zeros, and publish the eligible-jet/particle/pair counts.
Do not drop or resample jets to make the auxiliary coverage look better.

Only paired TRAIN richer four-vectors and matching masks are needed for new
supervision. Full validation classification must work with HLT inputs alone.
Use TRAIN probes for target errors and gradient diagnostics; do not create
validation target banks or use validation to fit scales, choose clipping,
adjust losses, or stop/revert training.

## 5. Exact geometric targets and numerical conventions

### 5.1 Pair features

Use the four quantities and ordering of the **pinned installed Weaver**
pairwise Lorentz-feature function:

~~~text
Phi(p_i,p_k) = [log(kT), log(z), log(Delta), log(m_pair_squared)]
kT = min(pT_i,pT_k) * Delta
z  = min(pT_i,pT_k) / (pT_i+pT_k)
Delta_squared = (rapidity_i-rapidity_k)^2 + wrap(phi_i-phi_k)^2
~~~

Use **rapidity**, not pseudorapidity, where the actual Weaver function does.
The preceding informal chat's angular-separation language must not become a
different implementation. Pin epsilon/floor conventions and feature order.
Do not replace the installed physics function with an approximation by accident.

For matched pairs:

~~~text
r_H = Phi(p_H_i, p_H_k)
r_U = Phi(p_U_i, p_U_k)
delta_r_target = r_U - r_H
~~~

Calculate targets from the stored physical four-vectors, not the clipped
17-dimensional input features or learned teacher tokens. Preserve raw stored
units; inherit the original producer-convention caveats. No unit conversion
is inferred from the variable names.

Only i < k participates in new geometric supervision; mirror predicted
corrections for symmetry, and set the added diagonal bias to zero. The native
model's diagonal attention and native pair bias remain intact.

### 5.2 Particle coordinates and decoder

Predict **four kinematic quantities**, not all 17 feature channels in this
screen. Leave charge, PID, impact parameters, uncertainties and validity fields
in the native HLT input path unchanged.

A concrete physically constrained decoder uses:

~~~text
q(p) = [log(pT), y, phi, mu]
mu = log(1 + mass_squared / pT_squared) >= 0
mass_transverse = pT * exp(mu/2)
p_x = pT*cos(phi); p_y = pT*sin(phi)
p_z = mass_transverse*sinh(y); E = mass_transverse*cosh(y)
~~~

It predicts residuals in log(pT), rapidity and wrapped azimuth around the HLT
measurement. For mu, use a nonnegative softplus parameterization initialized
near the HLT value:

~~~text
delta_a[c] = 6*tanh(raw_head[c]/6)
log_pT_hat = log_pT_H + s_p[0]*delta_a[0]
y_hat      = y_H      + hlt_eta_sign*s_p[1]*delta_a[1]
phi_hat    = phi_H    + s_p[2]*delta_a[2]
mu_hat = softplus(inv_softplus(mu_H + 1e-8) + delta_a[3])
~~~

Here hlt_eta_sign is the +/-1 reflection convention already computed from
the summed **HLT** jet four-vector by the input adapter (zero is positive).
The head predicts the rapidity correction in that reflected frame, then the
decoder returns to the stored physical frame. This prevents the eta-reflected
native features from asking the head to guess a missing orientation sign.
No richer-view axis or matching information enters this calculation.

The mass channel uses a residual in its latent positivity coordinate, while
its **loss** compares the resulting physical mu using s_p[3]. Implement
inv_softplus with a stable log/expm1 formulation. The 1e-8 floor is
dimensionless and only stabilizes the latent mass coordinate; it is not an
assumed physical mass. At zero head output this gives an approximately
identity four-vector, with only this declared floor. Native model identity at
initialization is established by the zero attention projector, not by pretending
this floor is absent.

Use wrapped angular differences for the phi loss. Evaluate geometry in FP32
outside backbone autocast; compute stored scale statistics in FP64. Audit
finite gradients for near-massless particles and record mass-channel sensitivity:
softplus near its floor may make mass corrections harder to learn. Treat a weak
mass response as a diagnostic result, not a reason to silently change the decoder
for only one arm.

Before target publication, audit positive pT/energy and mass-shell consistency
on real HLT/U100 four-vectors. For negative mass-squared from roundoff, permit
projection to zero in q only within
1e-6 * max(E^2, |p|^2, 1 raw-unit-squared), and report the count. A material
violation needs a documented target-contract decision before launch; never
silently repair the native classifier inputs. Preserve raw four-vectors for
native Phi targets and follow its authenticated floor.

### 5.3 Scales that preserve zero correction

Fit zero-centered **RMS residual scales**, rather than subtracting a residual
mean. A zero predicted residual must continue to mean no requested correction.
For each coordinate, compute mean squared target residual within each eligible
TRAIN jet, average over eligible jets, then take the square root:

~~~text
s_pair[c] = max(RMS_TRAIN(delta_r_target[c]), 1e-3)
s_p       = max(RMS_TRAIN(q_U-q_H), [1e-3, 1e-4, 1e-4, 1e-4])
~~~

Use wrapped phi residuals in these statistics. Store counts, moments, floors
and scale hashes. Do not fit separate scales for A/B/C/D. This normalization
is independent of, and does not change, the original U100 embedding standardizer.

For input to the direct-pair network, fit a separate TRAIN-only mean/std of
native HLT pair features, with std floor 1e-3 and equal weight per jet.
This mean-centering applies only to network inputs, not output residuals.
Store it in the same target-statistics artifact.

## 6. Shared model placement and inference

### 6.1 Compute once after two native particle blocks

Use one native encoder pass:

~~~text
native HLT input embedding
  -> native particle blocks 0,1 with unchanged attention
  -> capture their output h_i, width128
  -> predict corrections once from these contextual HLT states
  -> native blocks 2,...,7 with added predicted interaction bias
  -> unchanged class-attention blocks and final Linear(128,11)
~~~

Zero-based block 1 is the tap; zero-based blocks 2 through 7 receive the same
correction tensor. No correction enters class attention. This placement
provides context for prediction while leaving six blocks to exploit it.
It is a chosen first-screen default, not a measured best depth.

The new modules take the captured HLT states and HLT geometry only. They have
no access to labels, U100 arrays, matching masks or indices. Do not detach the
captured states. Each forward clears cached tensors even on exceptions.

Preserve installed Weaver's normal block and classifier implementations.
Use an audited forward hook for the tap and attention pre-hooks for adding
the bias, following the existing gate wrapper's interface checks. Add to the
floating attention bias without overwriting its native term or padding mask.
Support only explicitly tested installed attention implementations; never switch
the backend silently to bypass a gradient bug.

### 6.2 Particle head, shared by B/C/D

~~~text
LayerNorm(128) -> Linear(128,64) -> GELU -> Linear(64,4)
~~~

Initialize the final linear weight/bias to zero. Use identical head parameters
at initialization in B/C/D, with no branch-specific random consumption. Decode
the four-vectors as in section 5.2; calculate:

~~~text
c_particle[i,k] = (Phi(p_hat_i,p_hat_k) - r_H[i,k]) / s_pair
~~~

This is a coherent set of predicted particles. It is not a calibrated detector
reconstruction product or a claim of recovering event-specific missing truth.

### 6.3 Direct-pair head, shared by A/D

Compress each contextual token once, then use symmetric pair inputs:

~~~text
v_i = GELU(Linear(128,32)(LayerNorm(h_i)))
t_ik = concat(v_i+v_k, abs(v_i-v_k), v_i*v_k, standardized_r_H[i,k])
       # 32 + 32 + 32 + 4 = 100 components
c_direct = Linear(32,4)(GELU(Linear(64,32)(GELU(Linear(100,64)(t_ik)))))
~~~

Use no batch normalization or dropout in new heads. Initialize the final
Linear(32,4) to zero. Its output directly represents normalized pair
corrections. A and D must have identical initial pair-head parameters.
Construct only valid off-diagonal pairs and mirror; do not infer a direction
for symmetric kinematic targets.

### 6.4 One common bounded attention projector

All four arms use the same shape and initialization for a bias-free
Linear(4,8), W, initially zero:

~~~text
c = c_direct                       # A
c = c_particle                     # B,C
c = (c_particle+c_direct)/2         # D: fixed equal blend BEFORE projection
c_safe = 6*tanh(c/6)
delta_bias^head = alpha(update) * log(2) * tanh(W(c_safe))^head
alpha(update) = min(1, update/(3*steps_per_epoch))
~~~

Thus each new head's added bias is bounded by +/-log(2). This bounds its
multiplier on an unnormalized attention weight between 1/2 and 2; it does not
bound the normalized attention probability by the same ratio. The common
bound prevents D from receiving twice the attention budget.

Set corrections on padding and the diagonal to zero. Do not row-center after
the bound, which would change that bound and the diagonal convention. A
key-independent row constant does not affect softmax; no loss tries to fit one.

W=0 gives exact native forward behavior at initialization in evaluation mode.
The heads learn immediately through their auxiliary losses; W begins learning
as their outputs become nonzero. Verify that this is not a permanently dead
gradient path. Never initialize every layer of a predictor to zero.

The fixed ramp, bound and blend are shared across arms. They are not adjusted
using validation or target quality. D does not receive a learned blend in this
first screen, a third supervision term, or a consistency penalty between heads.

### 6.5 Differentiable geometry and the deployment checkpoint

A routine designed for measured input vectors may detach geometry or operate
under no_grad. Merely passing predicted vectors into it is not sufficient.
Create a differentiable companion for the new branch and verify native
forward-feature parity with the exact installed function on valid inputs.

For the differentiable square-root branch, floor squared distance before sqrt.
Below that floor use the detached native distance for the forward value, so
exact collinearity retains native kT=0 before its logarithmic floor. Never
compute a singular square-root derivative and then multiply it by zero.
Exclude the diagonal from the new branch's computation.
Test collinear off-diagonal particles, massless limits, phi wrapping, padding
and mixed precision. Avoid in-place operations on views used by autograd.

The deployed state contains the native model, active correction head(s), W,
and needed normalization buffers. It excludes the jet-embedding prediction
head and all target/matching payloads. Do not reuse native_state() or the old
U100 save/restore routine unchanged: they would discard necessary parameters.
Likewise, capture=False must disable only embedding-target capture, never the
inference correction. Reload and evaluate the exact deployed model for selection.

## 7. What each variant must actually do

| Arm | Attention correction | Particle loss | Pair loss |
| --- | --- | --- | --- |
| A / pair_u100 | c_direct | None | Predicted direct pair features versus U100 |
| B / p4_u100 | c_particle | Corrected particle coordinates versus U100 | None |
| C / p4_pair_u100 | c_particle | Same as B | Phi of those corrected particles versus U100 |
| D / dual_u100 | Fixed average of c_particle and c_direct | Same as B | Independent direct-pair prediction versus U100 |

For D, **do not apply the supervised pair loss to the blend or to c_particle**.
That would add C's supervision and change the agreed independent-branch test.
Classification and U100 embedding gradients still reach both branches through
the attention blend.

For B, derived-pair errors can be logged without gradients as diagnostics; they
must not be added to the optimizer objective. For A, do not invent a particle
reconstruction loss or a hidden physical decoder.

## 8. Losses and recommended initial weights

Define normalized errors and SmoothL1(beta=1):

~~~text
particle error = (q(p_hat)-q(p_U)) / s_p, with wrapped phi
direct pair error = c_direct - (r_U-r_H)/s_pair
derived pair error = (Phi(p_hat_i,p_hat_k)-r_U)/s_pair
~~~

Average coordinate losses over the four fields, average over eligible
particles or unique pairs within each jet, then over all jets including the
zero-eligible cases. Keep particle and pair reductions separate. A jet with
many constituents must not dominate simply because it has many pairs.

With rho(update)=min(1, update/(3*steps_per_epoch)):

~~~text
L_A = CE + 5*L_U100_embedding + rho * L_direct_pair
L_B = CE + 5*L_U100_embedding + rho * L_particle
L_C = CE + 5*L_U100_embedding + rho * (0.5*L_particle + 0.5*L_derived_pair)
L_D = CE + 5*L_U100_embedding + rho * (0.5*L_particle + 0.5*L_direct_pair)
~~~

Total new full-strength supervision weight is **1**, chosen as a cautious
first screen after per-field normalization. This is a proposal to freeze in
the new implementation, not a coefficient established by prior results.
There is no extra factor for number of particles, pairs, heads or blocks.
Do not rescale CE or the existing embedding objective.

Equal normalized coefficients do not imply equal encoder-gradient norms.
Log the norms and cosine similarities of CE, weighted embedding, and weighted
new supervision on fixed TRAIN probes. Do not adapt their weights online.
If this budget is ineffective, a later matched sweep can test it; a negative
first result is not proof that the strategy cannot work.

## 9. Seeds, references and interpretability

Initialize the native model with the authenticated original node seed, and
the U100 jet predictor with the original seed **20261019**. In isolated CPU
RNG scopes, use **2026110201** for the particle head and **2026110202** for the
direct-pair head. W is identically zero. Creating unused modules or TRAIN
diagnostics must not consume the backbone's dropout or data-order stream.

All four rows use the same original per-epoch row permutation and final
partial batch. Do not make a new sampling seed merely because the architecture
changed. Different architectures can still diverge numerically; verify common
initialization, RNG preservation and input sequence rather than claiming
training trajectories are identical.

Reuse the completed matched U100-w5 control after authenticating that its
initialization, optimizer, target bank, schedule, inputs and validation IDs
match. If authentication fails, stop and diagnose the missing evidence; do
not silently fall back to a different teacher, split, or control.

These four fits compare complete methods, including their parameter/compute
differences. They do not separately establish the causal benefit of privileged
supervision over extra capacity. The previously discussed architecture-only
and true-U100-pair oracle fits remain useful **later** ablations, outside this
four-fit launch. A true richer-feature oracle is not a deployable HLT model.

## 10. Efficient data, computation and diagnostics

### 10.1 Store particles, not quadratic pair banks

Reuse the existing authenticated TRAIN U100 embedding bank. Build a compact
new ragged target bank containing:

- ordered jet identities/labels and offsets/counts;
- U100 physical four-vectors in the exact persistent HLT slot order;
- a supervision-only matched-slot mask tied to the prepared assignments;
- source/view/input fingerprints and all TRAIN scale statistics.

Derive pair targets on demand from this O(N) payload. Do not persist N-by-N
targets for every jet. Source indices need not be model-adjacent arrays; the
bank manifest can bind their authenticated assignment files.

Load the existing HLT caches read-only. If an old required U100 cache payload
was deliberately retired, reconstruct it only through an explicit provenance
path from the identical selected raw rows and matching assignments. Check the
original expected hashes/fingerprints where applicable; a corrupt present
artifact is never treated as a cache miss. Do not modify old manifests.

### 10.2 Bound pair memory without changing the experiment

The direct-pair MLP and attention still have quadratic pair work. This may be
more costly than the current U100-w5 fit. Do not describe it as free or use a
small-token benchmark to estimate maximum support.

Enumerate unique valid pairs in bounded chunks, initially **4,096 pairs per
chunk**, and stream loss sums/counts without retaining whole-population pairs.
Use activation checkpointing of pair MLP chunks where necessary; recomputation
must preserve values, gradients and RNG. Avoid retaining a dense
B-by-N-by-N-by-100 feature tensor through backward.

Reuse one predicted bias across the six receiving blocks. Measure the actual
autograd lifetime and selected-reload memory. Preserve full pair support and
batch128; do not silently reduce tokens, sample a subset of loss pairs, change
the model, or use gradient accumulation to pass admission.

### 10.3 Diagnostics that answer the mechanism

Per epoch, record CE, weighted/unweighted embedding and new losses, actual
rho/alpha, LR, train/validation seconds and peak host/GPU memory.

On a small preregistered TRAIN subset, report:

- error per particle field and pair feature, plus the uncorrected HLT error;
- matched and unmatched prediction/correction magnitudes;
- saturation rates for residual and attention bounds, and mass-floor proximity;
- bias RMS and per-head spread on valid off-diagonal entries;
- CE/embedding/new-loss gradient norms and cosines on shared encoder parameters;
- D's two branch errors, disagreement and blended error;
- finite physical four-vector checks and mask/coverage counts.

Preserve RNG, module modes and running buffers around probes. Poor alignment,
unhelpful corrections or zero useful improvement are scientific observations,
not job-failure conditions. Input corruption, leakage and nonfinite required
values are technical failures and retain diagnostics.

## 11. Proposed code map and contracts

Implementation review baseline for this document:
**431341db5eced8eaa3e52b37ba047f8cd4beed50**. The workspace contains unrelated
uncommitted documentation; preserve it. The eventual source guard must use the
reviewed committed baseline at implementation time, not blindly freeze this hash
after other legitimate work has landed.

Existing reusable pieces:

| Existing location | Reuse or inspect |
| --- | --- |
| [U100 contracts](../../src/hlt_classification/luka_u100_embedding/contracts.py) | Original teacher, embedding weight and inherited recipe |
| [U100 target bank](../../src/hlt_classification/luka_u100_embedding/targets.py) | Authenticate/reuse standardized jet targets |
| [Original auxiliary model](../../src/hlt_classification/luka_embedding_aux/model.py) | Native jet-vector tap, separate U100 head; NOT its deployment filtering unchanged |
| [U100 training](../../src/hlt_classification/luka_u100_embedding/training.py) | Schedule, loss and reduction reference; adapt selected-state handling |
| [Native model adapter](../../src/hlt_classification/jetclass2_delphes/model.py) | Exact installed Weaver configuration and source identity |
| [View builder](../../src/hlt_classification/cms_proxy_ladder/views.py) | Recorded matching and persistent U100 slot construction |
| [Luka stage2](../../src/hlt_classification/luka_fullsim/stage2.py) | Prepared assignments, target lineage and inputs |
| [Luka caches](../../src/hlt_classification/luka_fullsim/regular_cache.py) | Bounded authenticated role/view loading |
| [Existing gate model](../../src/hlt_classification/luka_gating/model.py) | Audited attention-bias injection, masking and backend-gradient concerns |
| [Contrastive submission](../../src/hlt_classification/luka_u100_contrastive/submission.py) | Detached shared-setup dispatcher and two independent lanes |
| [Contrastive resources](../../src/hlt_classification/luka_u100_contrastive/resources.py) | Measured admission pattern, not its resource numbers |
| [Exact Slurm DAG](../../src/hlt_classification/scouting/hcwdl_exact_dag_submission.py) | Exact-ID journals and partial-submission handling |
| [Metric selection](../../src/hlt_classification/jetclass2_delphes/runner.py) | Original selection key and prediction interface |

Proposed **new**, currently absent paths:

~~~text
src/hlt_classification/luka_u100_pair_correction/
  contracts.py     # four rows, geometry/decoder, losses, seeds, deployment
  source.py        # original-source imports, source allowlist, reference reuse
  targets.py       # compact aligned TRAIN targets and residual moments
  geometry.py      # differentiable Phi, q decoder, stable periodic losses
  model.py         # one contextual tap, four variants, shared bias interface
  losses.py        # per-jet masked reductions and fixed budgets
  training.py      # inherited schedule, complete deployed checkpoint selection
  resources.py     # actual four-model miniatures and measured limits
  campaign.py      # immutable registration and storage budget
  submission.py    # detached setup -> four fits -> report
  worker.py        # production tasks, receipts, results and live progress
scripts/luka_u100_pair_correction.py
scripts/queue_luka_u100_pair_correction.sh
sbatch/run_luka_u100_pair_correction.sh
tests/test_luka_u100_pair_correction.py
docs/contracts/LUKA_U100_PAIR_CORRECTION.md
docs/LUKA_U100_PAIR_CORRECTION_RUNBOOK.md
~~~

Keep CLIs thin. No runtime Fresh_check import. Record every copied/adapted donor
file and exact donor commit in LEGACY_SOURCE_MAP during implementation.
No donor code is migrated by writing this plan.

Create a distinct LUKA_U100_PAIR_CORRECTION artifact family at v1. At minimum:
RECIPE, GEOMETRY, TARGET_STATS, TARGET_BANK, REFERENCE, SPEC, ACCEPTANCE,
PLAN, EPOCH, TRAINING_REPORT, DEPLOYMENT, FIT and RESULTS. All need canonical
hashes, parent hashes, role/view/population identity and source lineage.
Do not relabel existing U100, gate, contrastive or cache artifacts.

The deployment manifest must bind the tap/injection blocks, active predictors,
four-vector decoder, normalizers, bias map/bounds and final alpha=1.
Inference must not depend on a TRAIN bank path. Save payloads atomically, then
publish the completed manifest/receipt last. Failed/partial directories remain
diagnostic evidence, not reusable completed fits.

## 12. Debug execution and launch order

Intended first-screen site:

~~~text
partition: debug
account: reu-aisocial
qos: qos_tier3
GPU: one A100 per task
CPUs: 6
host memory envelope: 120000 MiB per task
environment: /home/ryreu/miniconda3/envs/atlas_kd_sporc
maximum concurrent scientific fits: 2
~~~

Use the existing site contract and verify site limits before constructing
commands. These settings do not imply that the new models have been measured.
Proposal: 12-hour shared-setup envelope, up to 24 hours per fit and a 2-hour
report envelope. Derive actual requests from measured loading, full-support
training, validation, probes and selected deployment reload, with explicit
safety factors. If they cannot fit, preserve the evidence and revise resource
planning; do not drop a scientific row or decrease its training budget.

There is **no separate preflight job**. One shared setup authenticates/reuses
targets, builds the compact geometry bank, and runs real production-path
technical miniatures and timing. Genuine new-model resource checks remain
necessary; historical U100 timing alone does not cover these retained branches.

Review the complete prospective graph before submission. Once shared setup
succeeds, a detached submit-host dispatcher automatically queues:

| Submission rank | Scientific task | Lane predecessor |
| ---: | --- | --- |
| 1 | p4_pair_u100 (C) | None |
| 2 | pair_u100 (A) | None |
| 3 | dual_u100 (D) | C, afterany for ordering only |
| 4 | p4_u100 (B) | A, afterany for ordering only |

Then queue report afterany all four. Total **six Slurm jobs**: one shared setup,
four scientific fits and one report. The detached dispatcher is a submit-host
process, not another Slurm job.

Once all jobs are accepted, no submit-host process is needed. Before that,
the detached dispatcher must survive logout and retain its log/journal.
User-visible output must distinguish setup submitted, waiting for setup, and
all four fits queued. Never imply setup success means science was submitted
without checking exact accepted job IDs.

No scientific student depends on another's model. A failed student must not
cancel the remaining independent experiments; a failed shared setup releases
none. No automatic retry of claimed/partial submissions. Cancel only exact
campaign-bound IDs under the user's authorization.

Precompute the target payload size from actual ragged counts. An **8 GiB
campaign budget** is a proposed starting allowance, with at least 1 GiB free
headroom; measure the complete expected bank, reports and checkpoints before
registration. Available quota is capacity, not a demand to reserve all free
filesystem space. Do not delete old results, datasets or Fresh_check.

## 13. Implementation sequence and acceptance

1. **Freeze semantics and import evidence.** Register the exact four rows,
   U100 teacher/bank, matched baseline, population, target masks, decoder,
   losses, seeds and deployment. Authenticate original artifacts without
   importing unrelated unfinished contrastive fits.
2. **Geometry and compact targets.** Implement Phi/decoder and masked RMS
   statistics, prepare a small synthetic ragged bank, then perform the TRAIN
   input audit and deterministic full-bank construction. Reuse the compatible
   existing jet-embedding export.
3. **Retained model variants.** Implement the one-pass tap, symmetric direct
   predictor, particle predictor, common zero-start projector, later-block
   bias injection and deployment serialization. Prove that C and D differ
   exactly as specified.
4. **Losses and production training.** Preserve the original schedule and
   common initialization/order. Select, save and reload the entire deployed
   model. Add immutable epoch metrics and meaningful progress messages.
5. **Reference/results and execution.** Implement read-only partial results,
   model/baseline deltas and historical recovery; wire shared setup, dry plan,
   automatic two-lane submission and exact journal recovery.
6. **Local installed-Weaver acceptance.** Run tensor tests, production-worker
   miniature/reload tests and failure injection. Update HANDOFF with evidence
   and write the real runbook/contract. Do not call design-only paths executable.
7. **Remote debug readiness.** Use clean pushed source and a full dry review;
   shared setup runs the genuine site miniature and resource admission before
   the four scientific fits are released automatically.

Required focused tests:

- Original HLT inputs, all row identities/masks and native parameters stay
  unchanged; shuffled/misbound matching metadata is rejected.
- U100 jet target reuse requires the exact original teacher and full payload
  lineage; existence or equal dimensions alone is insufficient.
- Phi values/order match pinned Weaver, including rapidity, logarithmic floors,
  symmetry, permutation transport and branch-cut cases.
- Decoder physicality, near-massless gradients, finite differences away from
  nonsmooth points, roundoff policy and scale normalization.
- Padding sanitization before arithmetic; no diagonal singularities; zero/one
  matched particle and unequal-cardinality batches; no loss dilution by padding.
- Chunked versus dense values/gradients agree on synthetic cases, including
  reentrant/non-reentrant activation-checkpoint choices actually deployed.
- Disabled added bias exactly matches native FP32 inference; initial W=0
  preserves the forward; auxiliary backward cannot mutate frozen targets.
- With nonzero W, CE gradients reach the correction predictor and earlier
  encoder through differentiable geometry. Test A, B, C and both D branches.
- B receives no pair-loss gradient; C's pair loss reaches its particle head;
  D's pair loss reaches only its independent pair branch through that loss path
  (both share encoder and receive classification/embedding gradients).
- Loss budgets/reductions and three-epoch ramps match the frozen formulas,
  including final partial batches and jets without new supervision.
- Evaluation cannot read a richer target or matching mask; removed embedding
  head does not change logits; retained correction weights do affect logits.
- Selected deployment reload reproduces validation predictions and native
  classifier outputs; no inference reference to training target files.
- Deliberately poor finite AUC/target error still yields all 30 epochs and a
  scientific result. NaNs, forbidden inputs, corrupt artifacts and stale source
  fail closed with saved diagnostics.
- Shared setup failure, resource-cap failure, worker failure, partial sbatch
  acceptance and duplicate launch all preserve exact claims/journals; successful
  setup submits exactly four fits and one report, not extra controls.

Real setup must stress both native largest-support classification batches and
largest eligible-pair backward batches, including the more expensive dual arm.
Measure geometry, diagnostics, target I/O and checkpoint reload, not only a
native network forward. Local synthetic tests cannot establish SPORC timing.

## 14. Results, interpretation and later decisions

Primary metric: selected **full-validation deployed-model macro OVR AUC**.
Also report accuracy, classwise QCD rejection at 30%/50%, and the same fixed
historical AUC-gap recovery:

~~~text
recovery_percent = 100*(AUC_model-AUC_M0HLT)/(AUC_OFFLINE-AUC_M0HLT)
~~~

Do not replace the denominator with U100 or a newly trained oracle.
Show deltas against the authenticated recent U100-w5 control, historical U100-w5
and, separately labeled, the completed mixed-identification result. Report
original50k from the same selected checkpoint, with explicit overlap disclosure.
Do not choose different checkpoints for different reported metrics/subsets.

Persist complete epoch AUC/recovery curves and best-so-far summaries, prediction
metrics and correction diagnostics. Measure deployed parameter count, peak
memory and latency under the same HLT input support as the reference. These
models consume only HLT inputs but have a larger inference graph.

Interpret useful patterns without converting them into stronger claims:

- C over B suggests value in supervising the consequences of particle repairs,
  although their particle-loss allocations differ under the equal-budget design.
- D over C suggests a useful complete alternative with extra flexibility and
  parameters; it does not alone prove that independent supervision is the cause.
- A over B/C may indicate that reconstructing coherent particles is unnecessarily
  restrictive for the classification task.
- Lower target error without higher AUC is not a successful recovery result.
- A null result at this one weight/bound is not a conclusive rejection.

Before a high-data transfer claim, use matched-seed confirmation and useful
capacity/no-supervision controls. A richer-pair oracle can test available
headroom, but is neither HLT-only nor a strict mathematical upper bound.
D066, other field groups, injection depths, loss budgets, prediction-assisted
particle feature inputs, and combinations with OFFLINE identification remain
later studies rather than hidden changes to these four rows.

## 15. External technical references

- [Particle Transformer for Jet Tagging, Qu et al.](https://proceedings.mlr.press/v162/qu22b.html):
  physical pair features and additive attention biases.
- [Upstream Weaver ParticleTransformer implementation](https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleTransformer.py):
  pairwise Lorentz-feature definitions and attention interfaces. This moving
  upstream link is explanatory; the execution must pin and validate its actual
  installed Weaver source.

No external reference establishes the effectiveness of this proposed four-arm
supervision study. The implementation and its scientific results remain future work.

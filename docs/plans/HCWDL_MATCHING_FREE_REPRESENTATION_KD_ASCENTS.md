# HCWDL Matching-Free Privileged Representation-KD Ascents

Status: **local implementation and local verification are complete;
installed-Weaver evidence, genuine Tigris acceptance, measured production
resources, exact pushed source, and explicit pilot authorization remain
pending; this document authorizes no submission or final-role access**.

Short name: **HCWDL-RKD**.

This plan defines four new deployable ascents that reuse the already-registered
HCWDL privileged teacher factory. It is subordinate to the data, matching,
repair, architecture, and final-test invariants in
[`HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md`](HIGH_COVERAGE_COLD_WARM_DISTILLATION_LADDER.md).
Where this plan adds new representation-loss, target-cache, graph, or report
semantics, this plan governs only those new registered-ablation artifacts. It
does not modify, relabel, or reinterpret an existing logit-only HCWDL campaign
or any PMARD execution.

The conceptual motivation is the local note
`Three lenses on the HLT–offline gap.pdf`. That note is a research prompt, not
a runtime dependency or contract. This plan keeps its useful set-distribution
idea while correcting two implementation-critical issues:

1. a loss on raw `deltaR`, pair mass, or `kT` cannot train the student encoder,
   because those quantities do not depend on student parameters;
2. ordinary feature MSE or cross-network MMD assumes compatible latent
   coordinates, which independently trained teachers and students do not
   necessarily possess.

HCWDL-RKD therefore aligns exact paired jets, compares unordered particle
**sets** through a constrained student-to-teacher projection, and compares
learned relational structure through orthogonal-basis-invariant latent
statistics. It
never requires particle correspondence in the representation loss.

“Matching-free” has that narrow, important meaning. D25--D100 teacher views
still use the parent campaign's already-authenticated Shell-Exact assignments
to define their privileged inputs. HCWDL-RKD neither rematches nor consumes an
HLT-token-to-offline-token correspondence in its loss. Native-offline M6 also
supports different charged/neutral particle counts without a correspondence.

## 1. Decision and campaign shape

The existing HCWDL campaign already supplies two primary controls:

- the complete logit-only cold ascent;
- the complete logit-only warm ascent.

HCWDL-RKD adds exactly four full `M1` through `M6` ascents:

| Branch | Canonical strategy | Initialization |
|---|---|---|
| `RSET-cold` | paired jet plus token-set KD | fresh backbone at every rung |
| `RSET-warm` | paired jet plus token-set KD | immediate same-branch predecessor |
| `RREL-cold` | paired jet plus token-set plus latent-relation KD | fresh backbone at every rung |
| `RREL-warm` | paired jet plus token-set plus latent-relation KD | immediate same-branch predecessor |

`RSET` is the conservative, recommended representation formulation. `RREL` is
the best predeclared alternative: it uses the same supervision families and
adds a corrected relational term while reallocating a fixed total
representation budget. `RREL - RSET` therefore tests whether the
relation-inclusive allocation is better than paired jet plus marginal
token-set alignment. It is not an “extra relation loss at no cost” causal
contrast.

The four branches contain 24 new training nodes. They reuse the authenticated
HCWDL roots and downward teachers; they do not train four additional downward
ladders. Together with the two existing logit-only ascents, the scientific
comparison contains six complete ascent families.

Every finite registered node runs its full declared budget regardless of an
earlier rung's performance. The campaign cannot stop a branch, splice a warm
rung into a cold branch, switch representation strategy mid-ascent, or replace
a disappointing child with an attractive unregistered checkpoint.

## 2. Scientific questions and allowed conclusions

The ordered questions are:

1. Does the full paired-jet-plus-unordered-set `RSET` package improve over
   logit-only KD across a complete ascent?
2. At the predeclared M5-cold diagnostic rung, is allocating representation
   supervision to the unordered set better than allocating it only to the
   paired pooled-jet state?
3. Is the equal-total-exposure relation-inclusive `RREL` package better than
   RSET across a complete ascent, and at M5 cold is transferring a quarter of
   the current-rung representation allocation to relations better than the
   corresponding no-relation allocation?
4. Do full-package representation gains compound across the six-rung ascent?
5. Does warm continuation exploit representation KD more effectively than
   repeated cold distillation?
6. Can native-offline structure improve `M6` without projecting native
   offline particles onto the HLT skeleton?
7. At M5 cold, are gains caused by correctly paired privileged structure rather than
   additional optimization terms or generic regularization?

The campaign does not claim that the full ladders separately identify the
causal contribution of every component at every rung. The compact M5 controls
localize jet-versus-set, relation-allocation, and pairing mechanisms at one
strong ordinary-teacher rung; their conclusions do not automatically
generalize to warm trajectories, TOFF, or all six rungs.

Allowed claims are limited to training-time privileged representation
distillation, set-distribution alignment, latent relational alignment, and
HLT-only deployment. A successful student may be said to recover a fraction
of the privileged teacher's classification advantage.

The campaign may not claim that:

- the student reconstructs missing offline measurements;
- MMD identifies corresponding particles;
- a latent relation is a physical particle match;
- a merged HLT candidate has literally been split into offline constituents;
- an HLT representation has become the native-offline representation;
- a performance gain proves the completion-shell assignment is physically
  correct.

Representation KD cannot create information absent from HLT. Its purpose is
to make the HLT ParT use surviving information and learn recoverable
correlations more completely.

## 3. Parent HCWDL artifacts and non-reinterpretation

### 3.1 Shared teacher factory

The representation branches consume the exact selected checkpoints already
defined by the parent HCWDL graph:

```text
D0c, D25c, D50c, D75c, D100, TOFF
D0w, D25w, D50w, D75w, D100, TOFF
```

The cold representation branches use cold downward teachers; the warm
representation branches use warm downward teachers. `D100` and `TOFF` are
shared exactly as in the parent graph. Cross-track substitution is forbidden.

The representation campaign also imports the parent source manifest, split,
row selection, train/validation assignment manifests, assignment lock,
Shell-Exact qualification lock, optimization recipe, and
selected teacher reports. It references their original paths and hashes. It
does not copy them into new files and pretend they were produced by the new
campaign.

The current parent artifact family does not publish the architecture hash this
plan needs. HCWDL-RKD therefore creates a new
`HCWDL_REPRESENTATION_ARCHITECTURE_ATTESTATION/v1` from the canonical model
configuration, installed-Weaver runtime signature, model source hash, tap
signature, parity report, and strict checkpoint-key/shape audits for every
imported teacher/student architecture. It records parent lineage but is not
misrepresented as an old parent artifact.

The attestation **schema and tap specification** are frozen before model edits;
the attestation artifact itself is published only after the named taps exist and
installed-Weaver parity has passed. Parent-import publication therefore follows
surface implementation/parity. “Parent import before any representation target
or student” is a runtime campaign gate, not an instruction to attest code that
has not yet been implemented.

### 3.2 Required prefix-import artifact

Before any representation target or student is built, publish and validate a
`HCWDL_REPRESENTATION_PARENT_IMPORT/v1` artifact containing:

- parent campaign-spec hash and source commit;
- source, split, and exact row-selection hashes;
- train and validation assignment-manifest hashes;
- assignment, recipe, and endpoint-qualification lock hashes;
- parent HCWDL graph hash and the new architecture-attestation hash;
- selected report, selected checkpoint, checkpoint byte hash, node identity,
  track, and domain for every imported D teacher and TOFF;
- selected report/checkpoint hashes for `M0` and every logit-only
  `M1c..M6c/M1w..M6w` node; these are mandatory paired controls, not optional
  imports;
- the parent loss-semantics attestation in Section 3.3;
- proof that every imported artifact validates under its original contract.

An incomplete, stale, cross-selection, cross-track, or path-only import fails
closed. The representation campaign must remain independently monitorable and
recoverable without changing parent artifacts.

### 3.3 Parent loss-semantics blocker and required attestation

The active parent plan defines class weighting for CE and explicitly leaves KD
unweighted. The current implementation of
`src/hlt_classification/scouting/training.py::pmard_loss`, however, multiplies
CE and both KD row losses by the class weight. HCWDL-RKD must not choose one
silently or call the two executions paired.

Before parent import, publish
`HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v1` proving that the parent
runtime, parent reports, and authoritative plan agree on:

```text
CE reduction: class-weighted row mean
KD reduction: unweighted row mean
temperature:  tau^2 forward KL in FP32
```

Under the current documentation authority, the runtime must be corrected and
any affected parent logit controls/teachers required for direct comparison
must be produced under the corrected, versioned semantics. Existing reports
trained with class-weighted KD cannot be relabeled or accepted as paired. The
only alternative is a separately authorized/versioned change to the active
parent scientific plan followed by consistent parent reruns. Until this
attestation validates, target construction and all HCWDL-RKD nodes are
blocked.

Representation per-jet class weighting in Section 18 is a new, explicit
HCWDL-RKD ablation semantic; it does not change the unweighted base KD rows.

### 3.4 Existing logit ascents remain controls

The already-running logit-only cold and warm ascents are never rerun merely to
make their timestamps or producer commit resemble HCWDL-RKD. Comparability is
established by authenticated scientific parents: identical rows, views,
architecture, base optimization recipe, teacher checkpoints, seed policy, and
checkpoint selector. If any of those differ, the report labels the comparison
nonpaired and does not present a direct delta.

## 4. Data populations and access

HCWDL-RKD uses the parent roles without making a new dataset representation:

| Mode | Train | Validation | Final test |
|---|---:|---:|---:|
| Smoke | at most 4,096 | at most 4,096 | inaccessible |
| Pilot | 300,000 | 100,000 | 100,000, sealed until combined locks |
| Production | complete mapped train role | complete mapped validation role | sealed until new locks |

The four representation branches use exactly the same train and validation
identities as the logit controls. The training stream remains natural
population: every selected identity appears exactly once per pass, including
the final partial batch.

Offline or repaired inputs are used only while the frozen representation
teacher is evaluated to create training targets. Once those targets exist,
every `RSET` and `RREL` student consumes only HLT particle inputs, labels, and
identity-joined training targets. No privileged particle array, assignment,
confidence, alpha, or source coordinate is passed to the student forward.

Validation checkpoint selection evaluates only the HLT student on HLT inputs.
Privileged validation representations are not training targets and are not a
checkpoint-selection metric.

## 5. Exact four-ascent graph

Canonical branch IDs are `RSET-cold`, `RSET-warm`, `RREL-cold`, and
`RREL-warm`. Canonical node IDs are explicit registry values, not strings
parsed at runtime:

```text
RSET_M1c ... RSET_M6c
RSET_M1w ... RSET_M6w
RREL_M1c ... RREL_M6c
RREL_M1w ... RREL_M6w
```

For either representation strategy `R`:

```text
cold:
  D0c              -> R_M1c
  R_M1c + D25c     -> R_M2c
  R_M2c + D50c     -> R_M3c
  R_M3c + D75c     -> R_M4c
  R_M4c + D100     -> R_M5c
  R_M5c + TOFF     -> R_M6c

warm:
  D0w              -> R_M1w
  R_M1w + D25w     -> R_M2w
  R_M2w + D50w     -> R_M3w
  R_M3w + D75w     -> R_M4w
  R_M4w + D100     -> R_M5w
  R_M5w + TOFF     -> R_M6w
```

The exact roles are:

| Rung | Student input | Predecessor logit teacher | Representation/logit source |
|---|---|---|---|
| M1 | HLT | none | track-matched D0 on HLT |
| M2 | HLT | same-branch M1 on HLT | track-matched D25 on D25 |
| M3 | HLT | same-branch M2 on HLT | track-matched D50 on D50 |
| M4 | HLT | same-branch M3 on HLT | track-matched D75 on D75 |
| M5 | HLT | same-branch M4 on HLT | shared D100 on D100 |
| M6 | HLT | same-branch M5 on HLT | shared TOFF on native offline |

Only the source in the last column supplies representation targets. The HLT
predecessor supplies logits only. This division is deliberate: predecessor
logits preserve accumulated deployable progress, while the privileged source
introduces new structure. Matching a predecessor representation as well would
overconstrain the child and obscure the purpose of the new rung.

`M1` has a sole D0 teacher. D0 supplies both its temperature-one logit target
and its representation target even though its input domain is HLT. Runtime
routing is based on declared teacher role, not on the mistaken assumption that
a representation source must have a non-HLT domain.

## 6. Cold and warm initialization

### 6.1 Deployable backbone

Cold nodes create a deterministic fresh canonical 21-input/15-output Scouting
ParT backbone using the **parent logit counterpart's backbone-initialization
seed**. Thus `M3c`, `RSET_M3c`, and `RREL_M3c` begin from the same deployable
weights for a fixed replicate seed, while different rungs still receive their
parent-declared distinct states. Only the training-only projections use the
new full representation node ID. This pairing is required for an efficient
strategy comparison.

Warm `M1` loads the selected D0 backbone from its track. Warm `M2` through
`M6` load the selected deployable backbone from the immediate predecessor in
the same representation strategy and track. Optimizer, scheduler, scaler,
sampler, update count, validation history, and RNG state reset exactly as in
the parent warm-start contract.

### 6.2 Training-only representation heads

Representation projections are training-only modules. They never enter the
deployable forward signature and are not carried from one rung to another.
At every cold or warm node they are newly initialized from a node-specific
seed. Bias-free square projections begin at identity wherever dimensions
permit.

This reset is mandatory. A projection learned into D25's latent basis is not a
valid projection into D50's independently trained basis. Warm continuation
therefore loads only canonical deployable backbone/classifier parameters and
creates fresh teacher-specific representation heads.

Resume is different from warm initialization: an exact resume restores the
current node's representation heads and their optimizer state in addition to
the backbone and every ordinary resume field.

### 6.3 Checkpoint separation

Every selected representation checkpoint records:

- a canonical deployable backbone state, loadable by the ordinary HLT ParT;
- a full training/resume state containing representation heads;
- proof that logits from the extracted deployable state equal logits from the
  training wrapper with representation heads inactive;
- the list of training-only parameter names excluded from deployable export.

A descendant warm start loads the deployable state only. Final inference is
physically unable to accept privileged targets because the exported model has
the baseline HLT-only forward signature.

## 7. Named representation surfaces

The implementation exposes representations during the same forward that
produces student logits. It must not run a second student forward, invoke the
Weaver trimmer twice, or advance augmentation RNG twice.

### 7.1 HLT and Shell-Exact ParT

For the canonical eight-particle-block Scouting ParT, expose exactly:

1. `particle_block_2`: masked token states immediately after the second
   particle-attention block, with shape `[batch, tokens, 128]`;
2. `jet_penultimate`: the 128-dimensional output of the class-token
   aggregator immediately before the final 15-class linear head;
3. logits, canonical token IDs, pre-transform family codes, and the
   post-trimmer mask and four-vectors from the same forward.

Canonical token IDs are zero-based HLT-skeleton indices for HLT/D views and
native charged/neutral branch indices for TOFF. Family codes follow Section
9.4. Both are training-time bookkeeping only. The surface forward propagates
them through the **same single trimmer call** as auxiliary channels, strips
those channels before the embedding layer, and proves that they never enter
attention, pooling, logits, or a deployable state. This makes pT ties, M6
family membership, and trimming auditable without running the trimmer twice
or inferring a correspondence after trimming.

The second particle block is deliberately shallow but contextual. Raw input
embeddings cannot use the rest of the jet to infer a missing or degraded
measurement, while very late particle states have already collapsed toward
the classification objective and overlap strongly with logit KD. Capturing
every block is excluded: it overconstrains the student, multiplies storage and
compute, and makes causal interpretation poor.

Weaver trimming and all parent logit-training behavior remain unchanged. The
set loss accepts a variable post-trimmer student set and therefore does not
need trimming disabled. A zero-coefficient representation wrapper must be
logit- and backbone-gradient-equivalent to the ordinary parent model under the
same RNG state.

### 7.2 Native-offline TOFF

TOFF has independent charged and neutral encoders, so their token states do
not inhabit one common coordinate system. Expose:

- `charged_particle_block_2`, `[batch, charged_tokens, 128]`;
- `neutral_particle_block_2`, `[batch, neutral_tokens, 128]`;
- `offline_jet_penultimate`, the 128-dimensional activation after TOFF's
  classifier LayerNorm, first linear layer, and GELU, before its final
  15-class linear layer;
- the two masks, four-vectors, canonical branch token IDs, and logits from the
  same teacher forward.

Charged and neutral token clouds remain separate targets. Concatenating their
unprojected hidden states and calling the result one offline embedding space is
forbidden.

### 7.3 Forward parity

The representation-exposing forward must reproduce the public forward logits
and gradients within the installed-Weaver FP32 parity tolerance with trimming
disabled for the parity fixture. A real miniature additionally exercises the
single-forward path with normal training-time trimming and exact resume.

## 8. Common paired-jet representation anchor

Both strategies include a paired jet loss. For each exact paired jet, let
`g_s` and `g_t` be the student and teacher penultimate vectors. Define
parameter-free LayerNorm as
`LN(x) = F.layer_norm(x.float(), (128,), weight=None, bias=None, eps=1e-5)`.
A bias-free student projection `A_g` maps the normalized student into the
teacher coordinate space:

```text
q_s = normalize(A_g LN(g_s), p=2, dim=-1, eps=1e-12)
q_t = normalize(LN(g_t),       p=2, dim=-1, eps=1e-12)
```

The direct term is:

```text
L_jet_direct = mean_j [1 - sum_d q_s[j,d] q_t[j,d]]
```

The batch-geometry term compares off-diagonal cosine Gram matrices before the
student projection:

```text
K_s = normalize(LN(g_s)) @ normalize(LN(g_s)).T
K_t = normalize(LN(g_t)) @ normalize(LN(g_t)).T
L_jet_gram = mean_{i != j} (K_s[i,j] - K_t[i,j])^2
```

For a one-row final partial batch, `L_jet_gram` is exactly zero. The raw jet
component is:

```text
L_jet = 0.75 * L_jet_direct + 0.25 * L_jet_gram
```

The direct term uses exact jet pairing and gives a strong per-example target.
The Gram term is invariant to an independent orthogonal basis change and
preserves teacher geometry across jets. It prevents the trainable projection
from being the only route by which jet structure is transferred.

At M6, `g_t` is TOFF's named 128-dimensional classifier hidden activation.
The same `A_g` contract applies; no charged/neutral special case is needed at
the pooled jet level.

## 9. Token-set kernel alignment

Both strategies include a per-jet unordered token-set loss on
`particle_block_2`. No token index is compared to another token index.

### 9.1 Student-to-teacher projection

A bias-free 128-by-128 student projection `A_tok`, initialized to identity,
maps student tokens into the active teacher basis. Teacher tokens are detached.
Projected student and teacher tokens are converted to FP32 and L2-normalized
with `F.normalize(..., p=2, dim=-1, eps=1e-12)` before kernel features are
evaluated.

The projection is deliberately only linear. A deep projection MLP could learn
the target while allowing the deployable backbone to remain uninformative.
The linear head can resolve rotations and mild coordinate changes but cannot
memorize individual jets.

### 9.2 Token weights

For the visible tokens of each view, compute
`pT_i = sqrt(px_i^2 + py_i^2)` from the same four-vectors that entered the
model. Let `n` be the visible count:

```text
u_i = 1 / n
p_i = sqrt(pT_i) / sum_j sqrt(pT_j)
w_i = 0.5 * u_i + 0.5 * p_i
```

Weights sum to one separately for student and teacher. The uniform half keeps
soft tracks and neutral particles visible; the softened-pT half prevents a
large number of tiny constituents from dominating. Nonpositive or nonfinite
visible pT fails target construction rather than being silently clipped.

### 9.3 Fixed multi-scale spectral-moment kernel

Use a deterministic finite random-Fourier feature map on unit-normalized
128-dimensional features. The frozen feature map defines the **exact training
kernel**

```text
k_FS(x, y) = phi(x).T phi(y)
```

for this campaign. Its spectral distribution is inspired by the equal mixture
of four RBF kernels below:

```text
k_sigma(x, y) = exp(-||x - y||_2^2 / (2 * sigma^2))
k_mix(x, y)   = (1 / 4) * sum_sigma k_sigma(x, y)
```

```text
sigma in {0.10, 0.25, 0.50, 1.00}
256 Fourier features per sigma
K_token = 1,024 total features
```

For each bandwidth, `omega ~ Normal(0, sigma^-2 I)` and
`b ~ Uniform(0, 2*pi)` are generated once from the versioned representation
kernel seed. The arrays and their logical hashes are part of the recipe. They
are FP32 and frozen. Dynamic batch-median bandwidths are forbidden because
they make the target depend on batch composition and resume layout.

Every scientific SHA derivation in this plan uses one canonical byte rule:
serialize a type-explicit object with Python `json.dumps(payload,
sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False)`, encode that exact string as UTF-8, and hash those bytes.
Hash inputs never use delimiter concatenation, native integer bytes, path
strings, or implicit object stringification. Jet identities enter seed payloads
only through the canonical identity digest already authenticated by the row
manifest. A 64-bit RNG seed is `int.from_bytes(digest[0:8], "big",
signed=False)`. The canonical serialized payload and its SHA-256 are retained
beside every derived resource/map so another language can reproduce the bytes.

Resource generation is exact. For each block hash this payload:

```json
{"bandwidth_index":0,"contract":"HCWDL_REP_RFF/v1","master_seed":20260808,"resource_name":"token_rbf_sigma_0p10"}
```

The four token resource names in index order are
`token_rbf_sigma_0p10`, `token_rbf_sigma_0p25`,
`token_rbf_sigma_0p50`, and `token_rbf_sigma_1p00`; relation resources use
`relation_rbf_sigma_0p05`, `relation_rbf_sigma_0p10`,
`relation_rbf_sigma_0p20`, and `relation_rbf_sigma_0p40`. Initialize
`numpy.random.Generator(numpy.random.PCG64(seed64))`, draw the complete
standard-normal frequency array followed by the complete uniform-phase array
in FP64, divide frequencies by the declared sigma, then cast C-contiguous
arrays to FP32. The producing NumPy version is recorded. Runtime acceptance is
governed by the stored array logical hashes, not merely by a seed or dependency
version.

For normalized token `z`:

```text
phi(z) = concatenated_sigma sqrt(2 / K_token) * cos(omega_sigma z + b_sigma)
mu(Z)  = sum_i w_i phi(z_i)
```

The per-jet set loss is:

```text
L_set = mean_j ||mu(A_tok Z_s,j) - mu(Z_t,j)||_2^2
```

This is the exact weighted MMD for the frozen finite kernel `k_FS`. It is not
claimed to reproduce the local gradient of the infinite RBF mixture: the
audited finite map does not do so compactly, and Section 23.4 makes that
limitation explicit. As a finite nonlinear spectral-moment objective it is
order-invariant, supports unequal counts, has exact gradients through student
token states and `A_tok`, and permits the teacher target to be cached as one
fixed 1,024-vector per jet instead of every teacher token. Gradient calibration
normalizes its achieved training signal; it does not relabel it as an
infinite-kernel gradient.

This interpretation is deliberate, not a relaxed failed gate. A deterministic
pre-implementation sweep found that ideal 128-D RBF-gradient fidelity at the
original threshold required at least 65,536 Fourier features **per bandwidth**
(`262,144` cached values per jet), which is storage-prohibitive for the
three-million-row role. The measured practical point of 256 features per
bandwidth gives ideal-kernel value MAE/p95/correlation
`0.009032/0.026687/0.997016` while remaining compatible with serialized JIT
banks. The plan therefore freezes the 1,024-dimensional finite spectral kernel as its
own estimand and forbids either calling it the
infinite kernel or inflating it into a multi-terabyte target bank. A future
infinite-kernel/coreset/sliced-kernel ascent would be a new registered strategy,
not an in-place change to RSET.

MMD is computed per paired jet and then averaged. Pooling tokens from different
jets into one batch-level cloud is forbidden; doing so could align the global
population while pairing the wrong offline jet with every HLT jet.

### 9.4 M6 charged/neutral treatment

For TOFF, create separate charged and neutral teacher kernel means. Partition
student HLT tokens with a pre-transform training-only family code derived from
canonical raw charge/PID:

1. read the five raw flags in the repository's canonical HLT order
   `(isEl, isMu, isChargedHad, isGamma, isNeutralHad)`. Every flag must be
   finite. A finite binary one-hot row maps exactly to
   `(electron, muon, charged_hadron, photon, neutral_hadron)`; a binary all-zero
   row is `unknown`; a nonbinary or multi-hot row is `malformed`;
2. raw charge must be exactly one of `{-1, 0, +1}` under the parent input
   contract. Nonfinite or out-of-domain charge is an invalid input and fails
   closed rather than becoming a family label. Nonzero proposes `charged` and
   zero proposes `neutral`;
3. a known electron/muon/charged-hadron PID proposes `charged`; a known
   photon/neutral-hadron PID proposes `neutral`. If that proposal disagrees
   with charge, the token is `unclassified_contradiction` rather than resolved
   by precedence;
4. an all-zero unknown PID uses the valid charge proposal and is counted as
   `charge_only`. A malformed PID is `unclassified_malformed` and never falls
   back to charge.

Unclassified tokens are excluded from M6 token and relation KD, retained in
CE/logit KD, and counted by class/reason. The family code and canonical token
ID are propagated through the one trimmer call and stripped before embedding;
neither is a deployable feature.

Use separate identity-initialized student projections `A_tok_charged` and
`A_tok_neutral`, because the two TOFF encoders have different latent bases.
When both families are nonempty on both sides, their set losses receive equal
weight. If exactly one family is jointly nonempty, it receives unit weight. A
family missing on either side supplies no representation gradient and is
reported; an empty target is never fabricated.

This is explicitly a support-overlap estimand. Reports publish teacher-present,
student-present, jointly eligible, charge-only, contradictory, and
unclassified fractions by class and family. Severe missing-family cases can
therefore receive only CE/logit/pooled-jet supervision; the campaign cannot
claim token-level TOFF transfer for those jets.

This family-aware construction is matching-free and count-agnostic. It avoids
the scientifically invalid shortcut of treating charged-encoder and
neutral-encoder coordinates as interchangeable.

## 10. Corrected latent-relational alignment

`RREL` adds one relational component. `RSET` does not compute it and therefore
serves as the nearest equal-budget package control. Because RREL reallocates
the component weights, it is not a strict mathematical submodel comparison.

### 10.1 What is and is not optimized

For token states `z_i` and `z_j` from `particle_block_2`, define the learned
relation:

```text
c_ij = sum_d normalize(z_i, eps=1e-12)[d]
             * normalize(z_j, eps=1e-12)[d]
```

`c_ij` depends on the student encoder, so its loss produces a valid student
gradient. It is invariant to a common orthogonal rotation of a model's latent
space, so student and teacher need not share the same orthogonal orientation.
It is **not** invariant to a general anisotropic, invertible, or nonlinear
reparameterization; independently trained encoders can therefore remain
semantically similar while receiving a nonzero relation/Gram penalty. This is
a declared limitation, not described as full coordinate invariance.

Raw `deltaR`, pair mass, `kT`, pT product, and particle counts are not direct
loss targets. They cannot be changed by the encoder. They may define pair
eligibility, deterministic strata, weights, and report-only closure
diagnostics. A code path that minimizes MMD solely between raw physical pair
clouds must be rejected by tests because its encoder gradient is identically
zero.

### 10.2 Deterministic pair population

Within each student and teacher view independently:

1. retain at most the 32 highest-pT visible tokens;
2. order by descending pT with the canonical pre-trimmer token ID propagated
   in Section 7 as the exact tie breaker;
3. construct every unordered pair `i < j` among those tokens;
4. exclude self-pairs; any nonfinite visible latent relation fails the batch
   and is never converted into an exclusion mask;
5. assign pair weight proportional to the product of the two token weights
   from Section 9.2 and normalize within each candidate stratum.

For normalized stratum weights `a_p`, define in FP64
`ESS = 1 / sum_p a_p^2`. A view/stratum is active only when it contains at
least four pairs and `ESS >= 3.0`. Student and teacher must each satisfy this
rule for the stratum to be jointly eligible. This prevents one unusually
weighted pair from receiving the same stratum weight as a stable relation
population.

The top-32 selection is independent in student and teacher views. It is not a
claim that rank `i` corresponds across views. Its sole purpose is to keep the
quadratic relation population bounded and concentrate on structurally
important particles.

For ordinary M1--M5 teachers, top-32 is selected over the complete visible
token set. At M6, top-32 is selected independently **within each charged and
neutral family** for both student and teacher; one family cannot consume the
other family's pair budget.

Pairs are divided using each view's own raw `deltaR` into fixed physical
strata:

```text
delta_phi = atan2(sin(phi_i - phi_j), cos(phi_i - phi_j))
deltaR     = sqrt((eta_i - eta_j)^2 + delta_phi^2)
```

`eta` and `phi` are deterministically derived from the same post-trimmer
four-vectors used by the model with the repository's canonical
`matching.py::p4_kinematics` NumPy-FP64 implementation. Wrapped `delta_phi`,
`deltaR`, and the comparisons to the two stratum edges are also evaluated in
FP64. Only the resulting fixed stratum IDs enter the FP32 representation-loss
path. Nonfinite visible geometry fails the target/student batch.

```text
local:   0.00 <= deltaR < 0.05
medium:  0.05 <= deltaR < 0.20
wide:    0.20 <= deltaR
```

The edges are fixed before results. Data-dependent quantile bins and
validation-tuned boundaries are forbidden.

### 10.3 Relational kernel sketch

Within each active stratum, use an exact deterministic one-dimensional finite
spectral kernel, constructed by the Section 9.3 resource rule from the stated
four RBF spectral scales, on `c_ij in [-1,1]`:

```text
sigma in {0.05, 0.10, 0.20, 0.40}
64 Fourier features per sigma
K_relation_per_stratum = 256
```

For each bandwidth, draw the frozen one-dimensional frequency and phase arrays
from the relation-specific resource substream and use:

```text
phi_rel(c) = concatenated_sigma sqrt(2 / 256)
             * cos(omega_sigma c + b_sigma)
```

The teacher target is the weighted kernel mean of teacher `c_ij`; the student
quantity is the corresponding weighted mean of student `c_ij`. The stratum
loss is their squared FP32 distance. `L_rel` is the equal-weight mean across
strata that are nonempty in both views.

If a stratum is empty or fails the pair/ESS gate in either view, it contributes
no optimization term and its occupancy/ESS difference is recorded. If no
stratum is jointly eligible for a jet, that jet's relational loss is exactly
zero and the jet is excluded from the `L_rel` denominator. The report defines
`relation_population_adequate = true` only when at least one jointly eligible
stratum exists in at least 90% of the train-only calibration jets. Falling
below that threshold is poor target support, not invalid input: the node still
runs and all descendants remain registered. Section 13.3 determines whether
the component can be calibrated or must remain explicitly inactive; support
is never fabricated and a row is never skipped from CE/logit KD.

Eligibility, pair count, ESS, and loss contribution are reported by class,
family, and stratum. No per-class result is silently generalized to classes
with poor support; class/family support weakness remains a scientific
limitation rather than a reason to skip a node.

At M6, relational sketches are computed separately within the charged and
neutral families defined in Section 9.4 and combined by the same equal-family
rule. Cross-family cosine is forbidden because the native-offline teacher
encoders have distinct bases. Cross-family information can still enter the
student through ordinary ParT attention, the pooled jet target, privileged
logits, and CE.

### 10.4 Interpretation

The relational component asks whether HLT contextual token states have similar
**marginal cosine distributions within fixed physical-separation strata** to
the privileged model. It neither reconstructs pair-graph topology nor makes a
merged HLT object become two tokens. A positive full-ascent `RREL - RSET`
difference supports only that the complete relation-inclusive allocation and
its accumulated predecessors worked better; it is not an isolated causal
estimate of the current relation term. The predeclared M5 no-relation control
provides the narrower, single-rung allocation diagnostic described in Section
23.2.

## 11. Projection regularization

Let the active student-to-teacher projections be the set `A`. Each is
bias-free, square, identity-initialized, and training-only. Define:

```text
L_orth = mean_A ||A.T A - I||_F^2 / 128
```

The coefficient inside the representation objective is fixed to `1e-3`.
Projection parameters are excluded from AdamW weight decay because the
explicit orthogonality term owns their scale/conditioning. They otherwise use
the node's ordinary optimizer and learning-rate schedule.

The orthogonality term discourages a projection from collapsing dimensions or
arbitrarily rescaling the loss. It does not force the deployable backbone to
copy a teacher coordinate system exactly. Singular values, condition number,
and `L_orth` are recorded at every validation boundary. A nonfinite projection
is a numerical failure. A **finite** condition number above `1e4` is a
predeclared poor-conditioning diagnostic, not an execution failure, checkpoint
selector, mid-run reset, or reason to suppress descendants; the registered
row continues and reports the excursion. This preserves the campaign rule
that a finite scientifically poor auxiliary cannot cancel an experiment.

## 12. Exact two representation strategies

Raw component losses have different natural scales. Section 13 defines a
train-only gradient calibration that produces normalized components
`Lhat_jet`, `Lhat_set`, and `Lhat_rel`.

### 12.1 `HCWDL_REP_SET/v1` (`RSET`)

The recommended conservative strategy is:

```text
L_repr_RSET = 0.40 * Lhat_jet
             + 0.60 * Lhat_set
             + 1e-3 * L_orth
```

The set term receives the larger share because unordered particle structure
is the main new information beyond logit KD. The paired jet term anchors the
set objective to the exact paired example and transfers a stable, task-aware
summary.

### 12.2 `HCWDL_REP_REL/v1` (`RREL`)

The potentially stronger relational strategy is:

```text
L_repr_RREL = 0.30 * Lhat_jet
             + 0.45 * Lhat_set
             + 0.25 * Lhat_rel
             + 1e-3 * L_orth
```

The three scientific weights sum to one. `RREL` does not receive a larger
total representation budget merely because it has another component. It
reallocates one quarter of the same budget to latent relations. This makes the
strategy comparison interpretable.

### 12.3 Explicit exclusions from both full ascents

The four full ascents do not include:

- raw index-aligned token MSE;
- matching-assignment-conditioned representation MSE;
- raw pair-physics MMD as a student loss;
- all-layer or layer-by-layer feature matching;
- progressive layer freezing;
- a deep trainable feature adapter;
- attention-map matching that depends on undocumented Weaver internals;
- a frozen teacher-block perceptual forward;
- validation-tuned kernel bandwidths or loss schedules.

A frozen tagger/perceptual lens remains a plausible later single-rung
experiment, but it is not one of these full ascents. Privileged logit KD and
the pooled teacher representation already make both registered strategies
task-aware. Adding a frozen teacher block would substantially increase compute
and introduce a second coordinate-compatibility boundary before the simpler
set and relation hypotheses are established.

## 13. Train-only gradient calibration and total loss budget

### 13.1 Why calibration is required

An MMD value, a cosine loss, and a relation-sketch distance do not share a
meaningful numeric scale. Assigning the same raw coefficient would make the
largest-valued implementation dominate for accidental reasons. HCWDL-RKD
therefore calibrates each component once per node against the already-locked
base-loss gradient.

### 13.2 Calibration population and activation barriers

Select exactly 4,096 train identities by the smallest SHA-256 values of the
Section 9.3 canonical-JSON UTF-8 payload:

```json
{"campaign_sha256":"<hex>","contract":"HCWDL_REP_GRAD_CAL/v1",
 "identity_sha256":"<hex>","parent_logit_counterpart_node_id":"<id>"}
```

The subset remains in natural class proportions up to deterministic hash
variation; it is not class-balanced or validation-selected. Sort the selected
rows by ascending 32-byte selection digest, then by ascending 32-byte canonical
identity digest as the collision tie-break. A repeated canonical identity fails
closed. The ordered list and its SHA-256 are part of the calibration contract;
no filesystem, source-file, loader, or dictionary order may replace it. Divide
that exact order into 16 consecutive batches of 256. If the train role contains
fewer than 4,096 rows, smoke uses every available row in the same order and
records the smaller non-scientific count. RSET and RREL counterparts therefore
use the same calibration identities and batch order; strategy text cannot
perturb either.

Calibration occurs immediately before each component family first becomes
active, not at the fresh initialization:

```text
after pass-2 validation, before pass 3: calibrate L_jet and L_set
after pass-4 validation, before pass 5: calibrate L_rel (RREL only)
```

This timing matters: two base-only passes deliberately move the student away
from initialization, and calibrating against the unused initial state would
misstate the local sensitivity scale when representation KD begins. RREL's relation
scale is likewise measured after its two passes of active jet/set training.

Each barrier uses the exact live deployable backbone and current
training-only heads, cached teacher logits/representations, base-loss recipe,
BF16 forward policy, and FP32 losses. It performs no optimizer/scheduler step
and uses `torch.autograd.grad` on the named captured surfaces, never parameter
`.grad` accumulation. Model mode/buffers, all CPU/CUDA RNG, trimmer state,
sampler position, optimizer state, logging windows, and target cursors are
snapshotted and restored. The next scientific batch is therefore identical to
the no-calibration trajectory except for the newly frozen scalar scales.

Within one calibration batch, execute the stochastic student graph **once**.
That one call produces the logits and every captured student surface used by
the barrier. Derive the base per-row loss and all component losses from those
same tensors, then call `torch.autograd.grad` for `G_base,k` and `G_rep,k` with
the required `retain_graph` setting. Re-running dropout, trimming, attention,
or the model separately for base and representation gradients is forbidden.
Component-specific support changes only the FP32 reduction of already-created
per-row losses; it never creates a second stochastic forward. Teacher targets
are cached/detached. Tests count forward calls and prove identical RNG and
surface tensors across every component norm at the barrier.

The ordering at every validation boundary is exact:

```text
finish scientific pass -> validation -> required calibration barrier
-> representation diagnostic -> boundary/resume commit
```

At ordinary boundaries the calibration step is a no-op. At pass 2, jet/set
scales are published before that boundary's diagnostic; relation is `null`
with `not_yet_calibrated`. At pass 4, RREL relation calibration is published
before the diagnostic; RSET relation remains `null` with
`not_part_of_strategy`. Before pass 2, jet/set and relation scale-dependent
diagnostics are `null` with `not_yet_calibrated`. A diagnostic may never use a
scale that is still being calculated or run before a required barrier.

The pass-two and pass-four barriers are safe checkpoint boundaries. If a job
is preempted before a barrier artifact is atomically published, resume restores
the exact boundary checkpoint and repeats calibration. If the artifact exists,
resume validates its boundary-model and target hashes and never recalibrates.
Each phase is a separate immutable
`HCWDL_REPRESENTATION_GRADIENT_CALIBRATION/v1` artifact. RSET publishes a
one-phase `HCWDL_REPRESENTATION_GRADIENT_CALIBRATION_MANIFEST/v1` after jet/set
calibration; RREL publishes its complete manifest only after both the jet/set
and relation phase artifacts exist.

### 13.3 Scale calculation

Calibration uses one common early-backbone parameter set for every component:

```text
Theta_cal = sorted trainable deployable parameters under
            mod.embed.*, mod.pair_embed.*, mod.blocks.0.*, mod.blocks.1.*
```

The exact names, shapes, count, and total scalar count are bound by the tap and
architecture attestations. Representation projections, classifier-only
parameters, and teacher parameters are excluded. A missing, extra, frozen, or
gradient-disconnected required parameter fails calibration.

For every calibration batch and component `k`, compute FP32 parameter-gradient
RMS on this identical scalar support:

```text
support_k = jets eligible for component k in that calibration batch
G_base,k  = sqrt(sum_{p in Theta_cal} ||d L_base[support_k] / dp||_2^2
                 / sum_{p in Theta_cal} numel(p))
G_rep,k   = sqrt(sum_{p in Theta_cal} ||d L_k[support_k] / dp||_2^2
                 / sum_{p in Theta_cal} numel(p))
```

For jet loss, `support_k` is every jet. For a family-aware set or relation
component it is exactly the jets that enter that component's denominator under
Sections 9 and 10. Both base and representation losses are re-reduced over the
same rows with their declared class weights for calibration only. Padding,
ineligible families, ineligible strata, and zero-filled target storage never
enter either gradient. A batch with no eligible rows supplies no value for
that component and is not replaced by zero.

Component scales are:

```text
s_k = median_batch(G_base,k) / median_batch(G_rep,k)
Lhat_k = s_k * L_k
```

Medians are computed from unrounded FP64 host values. Nonfinite tensors or
gradients remain numerical failures. Valid but weak target support is handled
without cancelling a scientific row: if fewer than 12 batches contain eligible
support, either median is zero/nonpositive, or the finite implied scale lies
outside `[1e-4, 1e4]`, the phase publishes
`component_status = inactive_valid_support`, freezes `s_k = 0`, and records the
exact reason and all observed norms. The component then contributes a literal
zero with zero gradient for the remainder of the node; its nominal coefficient
is not reallocated to another component. Values are never clipped or replaced
with invented targets. Other components, the complete base objective,
validation, checkpoint selection, and all descendants continue.

An active phase publishes `component_status = active` and its scale. Both
statuses are immutable resume state. The calibration report stores every batch
norm, eligible-row/class/family/stratum count, median, scale, hexadecimal
float, target and shuffle-map hash when applicable, exact boundary
deployable-state logical hash, exact boundary training-state logical hash
(including active heads), completed update/pass, and representation recipe
hash. A disconnected implementation on a fixture that should be eligible is
caught by numerical acceptance; runtime sparsity itself is not reclassified as
corruption.

The same jet/set components receive independently calculated scales in RSET
and RREL nodes because warm predecessors and later scientific trajectories can
differ. The algorithm, population, activation boundary, and total budget are
identical. A cold RSET/RREL pair that is still state-identical at pass two must
produce equal scales within the authoritative deterministic tolerance; this is
a paired-execution diagnostic, not a checkpoint selector.

This common-parameter construction makes every `s_k` a comparable local
sensitivity ratio. It still does not make `rho_repr` an exact whole-model
gradient-norm fraction: component gradient vectors can align or cancel, the
jet loss also propagates through early token blocks, and the small
orthogonality penalty is not calibrated. The achieved combined/base gradient
ratio on a fixed diagnostic batch is reported but never used to retune a node.

### 13.4 Base loss remains unchanged

For M1:

```text
L_base = 0.25 * weighted_CE
       + 0.75 * KD(D0, temperature=1)
```

For M2 through M6:

```text
L_base = 0.25 * weighted_CE
       + 0.40 * KD(same-branch HLT predecessor, temperature=1)
       + 0.35 * KD(privileged source, temperature=2)
```

The existing temperature-squared correction, class-weighted CE, unweighted KD
row reduction required by Section 3.3, FP32 CE/KL, and BF16 model execution
remain exact. Representation KD does not renormalize, reduce, or replace any
of these coefficients.

### 13.5 Nominal auxiliary coefficient

The full-strength representation auxiliary coefficient is fixed at:

```text
rho_repr = 0.10
```

The total loss is:

```text
L_total = L_base + rho_repr * scheduled(L_repr)
```

`rho_repr=0.10` means each calibrated component was normalized against base
loss on the common early-backbone parameter set at its activation boundary,
then receives a nominal one-tenth auxiliary coefficient after the declared
component allocation. It is intentionally auxiliary rather than another
probability-mixture coefficient. It is **not** described as a guaranteed 10%
whole-backbone gradient norm for the combined objective.

Changing `rho_repr`, the component allocations, calibration algorithm, or
normalization bounds creates a new representation recipe version and campaign
identity. An environment variable or command-line override is forbidden.

## 14. Exact representation warm-up schedule

Let:

```text
U_pass = ceil(N_train / 256)
e(u)   = (u + 1) / U_pass
```

for zero-based optimizer update `u`. The jet/set ramp is:

```text
r_js(e) = 0                    for e <= 2
          (e - 2) / 4          for 2 < e < 6
          1                    for e >= 6
```

The relational ramp is:

```text
r_rel(e) = 0                   for e <= 4
           (e - 4) / 4         for 4 < e < 8
           1                   for e >= 8
```

Thus both strategies begin with two full passes of the locked CE/logit loss.
Jet/set supervision reaches full strength at the end of pass six. `RREL`
introduces relations only after four complete passes and reaches full relation
strength at the end of pass eight.

For RSET:

```text
scheduled(L_repr) = r_js * (0.40 Lhat_jet + 0.60 Lhat_set
                            + 1e-3 L_orth)
```

For RREL:

```text
w_common(e)       = r_js(e) - 0.25 * r_rel(e)
scheduled(L_repr) = w_common * (0.40 Lhat_jet + 0.60 Lhat_set)
                    + 0.25 * r_rel * Lhat_rel
                    + r_js * 1e-3 L_orth
```

The orthogonality term is active whenever a projection receives gradients;
during a zero representation-ramp interval it is also zero. The base CE/KD
coefficients remain constant for all 60 passes.

Because `0 <= r_rel <= r_js`, `w_common` is nonnegative. The scientific
jet/set/relation coefficients sum to exactly `r_js` at every update in both
RSET and RREL. From passes two through four the strategies have identical
jet/set exposure. From pass four through eight, RREL continuously transfers
up to one quarter of that same scheduled exposure from jet/set to relations;
it never receives less or more total scheduled scientific representation
weight merely because the relation ramp is delayed. At full strength the
coefficients are exactly `0.30/0.45/0.25`.

Smoke uses its two-update execution budget and evaluates the same schedule in
update coordinates. It is expected to remain in the zero-representation
interval, so smoke additionally invokes
`--exercise-full-representation-loss`. This is a two-part, explicitly
non-scientific acceptance path:

1. one bounded real-data batch traverses the complete loader, surface capture,
   target join, family/stratum masking, and base-plus-representation loss; and
2. deterministic post-tap fixtures make **every** registered ordinary-D and
   TOFF jet/set/relation component eligible, including charged and neutral M6
   families and all three relation strata.

The second part intentionally bypasses the production `12 of 16` support gate.
On a temporary deep copy under a forked RNG state it runs the exact Section
13.3 gradient-norm and scale formula on at least one eligible fixture batch,
requires a finite positive scale and finite nonzero student-surface/backbone
gradient for every recipe component, forces both ramps to one, and performs no
optimizer/scheduler step. It does not turn unavailable bounded-population
support into scientific evidence. The probe publishes
`HCWDL_REPRESENTATION_SMOKE_PROBE/v1` with `scientific_authorization=false`,
fixture/resource hashes, component eligibility, scales, and gradient norms;
that artifact is rejected anywhere a pilot/production calibration is required.
The copy is discarded and all RNG/cache counters are restored, so the
two-update trajectory is byte-identical with or without the probe. This keeps a
small smoke from passing vacuously merely because production support logic
marked every component inactive.

## 15. Training duration, validation, and checkpoint selection

Every pilot or production representation node trains for exactly 60 complete
natural-population passes. There is no performance early stopping. Validation
runs after every pass, yielding exactly 60 records.

Checkpoint selection is unchanged from HCWDL:

1. highest validation macro one-vs-rest AUC;
2. lowest validation multiclass CE;
3. highest validation mean log QCD rejection at 50% signal efficiency;
4. earliest optimizer update;
5. lexicographically smallest canonical checkpoint identity.

Representation loss, teacher similarity, adapter condition number, and target
cache diagnostics are report-only and never select a checkpoint. All 60
passes continue after the selected checkpoint is first observed.

## 16. One-pass teacher target construction

### 16.1 Combined target pass

For the declared representation source, run one frozen FP32 teacher pass over
the selected train identities and produce in the same forward:

- 15 raw logits;
- the FP32 pooled jet vector;
- the FP32 token kernel mean or charged/neutral means;
- the FP32 relation kernel means and eligibility masks required by the shared
  RSET/RREL superset, even when the first consumer is RSET;
- canonical label-free jet identity.

The teacher is in evaluation mode and detached. Its logits and representation
surfaces come from the same selected checkpoint and same view. Separate passes
that accidentally use different checkpoint/runtime states are forbidden.

For M2 through M6, the HLT predecessor requires only its ordinary identity-
joined FP32 logit cache. The privileged source supplies both logits and
representation targets. At M1, D0 supplies the combined target bank.

After target construction, all frozen teachers and privileged particle views
leave GPU. The student then trains for all 60 passes from its HLT view cache
and RAM target joins only.

### 16.2 Logical target-bank contract

A physical target generation is authenticated by its
`HCWDL_REPRESENTATION_TARGET_MANIFEST/v1` plus the complete ordered set of
`HCWDL_REPRESENTATION_TARGET_SHARD/v1` children described in Section 20. The
separate logical-bank artifact freezes the scientific target meaning across
screen, confirmation, and recovery generations. A generation manifest binds:

- logical bank ID, physical generation ID/purpose, teacher track/domain, exact
  consumer-registry hash, and complete authorized consumer node/strategy set;
- teacher node/domain/track, selected report, checkpoint, and byte hashes;
- source, split, train row selection, graph, assignment, repair, architecture,
  parent recipe, and representation recipe hashes;
- exact named layer surfaces and installed-Weaver signature;
- exact pre-build target-forward specification and post-build execution
  attestation from Section 16.3;
- RFF resource arrays and hashes;
- identity order/set hashes;
- dtype, shape, logical hash, and finite audit for every target array;
- charged/neutral family and relation-stratum eligibility counts;
- authoritative teacher-forward dtype `float32`;
- `storage_mode = transient_durable_sketch` and
  `durable_particle_dataset_published = false`.

`logical_target_sha256` hashes the ordered logical arrays and scientific
parents only; it deliberately excludes physical generation, compression,
purpose, and consumer registry. The generation hash includes all of those
excluded operational/authentication fields. Thus confirmation can demand the
same scientific target hash while still publishing a distinct immutable
generation for a different consumer set.

Every join requires all requested identities exactly once. Missing, duplicate,
reordered, nonfinite, wrong-layer, wrong-teacher, wrong-strategy, or
wrong-selection targets fail closed.

### 16.3 Canonical target-forward specification and execution attestation

Exact FP32 target bytes are part of the logical scientific artifact. They may
therefore be generated only under an immutable **pre-build**
`HCWDL_REPRESENTATION_TARGET_FORWARD_SPEC/v1`, not merely from the same
checkpoint. The spec contains no output-derived value and is fully validated
before a privileged branch is opened. It binds:

- strict checkpoint byte/logical hashes, model configuration, architecture and
  tap attestations, representation recipe, and kernel-resource hashes;
- the pushed producer commit/source snapshot and installed package inventory,
  including Python, PyTorch, CUDA, cuDNN, NumPy, Awkward, Uproot, and Weaver
  versions or source hashes;
- canonical target-builder request `gpu:gh200:1`, NVIDIA GH200/Hopper
  architecture and compute capability, plus exact driver/runtime/package
  versions measured by the genuine miniature and frozen in the campaign spec;
- evaluation mode, FP32 parameters/inputs/activations, autocast disabled, TF32
  disabled for matmul and cuDNN, and no reduced-precision FP32 reduction;
- `torch.use_deterministic_algorithms(True)`, deterministic cuDNN, benchmark
  disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and all required CPU/CUDA RNG
  states even when the forward is expected to be RNG-free;
- canonical per-source batching: identities sorted by `(source_file_id,
  source_entry)`, batches of exactly 256 that never cross a source partition,
  and one final short batch per partition with no padding or row duplication;
- exact input decoding, feature dtype/layout, trimmer/evaluation behavior,
  family-code construction, surface capture, sketch arithmetic, and output
  array C-order.

After all shards have passed their finite/count/logical audit, but before the
generation manifest is published, create an immutable **post-build**
`HCWDL_REPRESENTATION_TARGET_EXECUTION_ATTESTATION/v1`. It binds the spec hash,
actual producer commit/environment/device model and audit-only GPU UUID, every
enforced backend flag, canonical batch count/partition hash, first/middle/last
sentinel-batch logits/surface/sketch logical hashes, every completed output
array logical hash, measured finite target-construction seconds, and the
resulting `logical_target_sha256`. The generation
manifest binds both the pre-build spec and this post-build attestation. It
cannot validate if either was used in the other's place.

The target-forward batch partition is scientific because kernel selection can
change FP32 activations. It is distinct from compression and file-I/O chunk
sizes, which may vary only after target arrays already exist. A recovery host
that cannot reproduce the complete pre-build spec is **not** allowed to rebuild:
the scheduler must retain or copy already-authenticated shards, wait for a
matching worker, or mark recovery blocked. There is no tolerance-based
substitution for an exact target hash.

For a rebuild, the prior execution attestation supplies the sentinel and full
logical hashes. Before full production the builder dry-replays at least the
first, middle, and last canonical batches and requires exact sentinel
agreement; the completed rebuilt generation must reproduce the prior full
`logical_target_sha256`. Initial construction has no circular sentinel
requirement: it validates only the pre-build spec, then publishes the first
post-build attestation. This is an execution-reproducibility requirement, not
a claim that arbitrary hardware/software stacks produce bit-identical neural
activations.

### 16.4 Consumer RAM materialization

The shared source shards are transient durable campaign artifacts, not
process-local-only data. At student startup, the consumer validates every
shard and manifest parent, allocates an exact-size FP32 RAM bank, loads the
columns its strategy needs, verifies canonical identity coverage, and closes
all shard handles before optimization begins. The consumer never performs a
privileged teacher forward or opens a privileged particle view.

On requeue, the job repeats that deterministic validation/materialization and
requires the logical array hashes to equal those in its resume state. The
durable shards are deleted only by the last-consumer cleanup stage in Section
20. Path existence alone never authorizes reuse, and a scratch copy in
`SLURM_TMPDIR` or `/dev/shm` is never an independently reusable artifact.

### 16.5 Exact target-array memory formulas

For ordinary D teachers, the shared physical superset requires:

```text
15 logits + 128 pooled + 1,024 token sketch + 3 * 256 relation values
= 1,935 FP32 values = 7,740 core bytes per jet
```

The TOFF superset requires:

```text
15 logits + 128 pooled + 2 * 1,024 token sketches
          + 2 * 3 * 256 relation values
= 3,727 FP32 values = 14,908 core bytes per jet
```

Section 29.1 freezes `75` ordinary or `113` TOFF metadata bytes per row, so the
exact uncompressed logical row widths are `7,815` and `15,021` bytes. Thus
300,000 pilot rows require `2,344,500,000` bytes (`2.3445 GB`, `2.183486 GiB`)
for one ordinary bank or `4,506,300,000` bytes (`4.5063 GB`, `4.196819 GiB`)
for TOFF. At exactly 3,000,000 production rows, the corresponding values are
`23,445,000,000` bytes (`23.445 GB`, `21.834858 GiB`) and
`45,063,000,000` bytes (`45.063 GB`, `41.968189 GiB`). Compression and NPZ/JSON
container overhead are excluded from these logical-byte identities and must be
measured separately. The mandatory cold-then-warm lifecycle permits only one
committed bank at a time; no compression saving is assumed for authorization.
The storage estimator adds the exact fixed metadata, staging/recovery reserve,
and filesystem headroom rather than using these core-array values alone. Exact
shape-based bytes from the authenticated row count are computed before
allocation and must remain below both the configured target cap and 75% of
`SLURM_MEM_PER_NODE` after all simultaneously live view, model, optimizer, and
construction buffers are included.

## 17. Numerical precision and autograd boundary

- Teacher target forwards and target sketches are FP32.
- Student ParT execution follows the parent BF16 autocast policy.
- Every captured student representation is recursively converted to FP32
  before LayerNorm, normalization, projections used by the loss, RFF maps,
  Gram products, cosine relations, reductions, and coefficient multiplication.
- RFF frequencies/biases, target banks, gradient-calibration values, and loss
  accumulators are FP32 or FP64 as explicitly declared.
- Teacher arrays are detached and cannot acquire gradients.
- The relational loss must have a nonzero finite gradient with respect to
  student token states on a nondegenerate fixture.
- Raw physical pair features must have no gradient requirement and cannot be
  the only varying coordinate in a representation objective.
- Nonfinite logits, representations, kernel features, gradients, losses, or
  required metrics fail the node; batches are never skipped.

All reported interval losses are means over examples, not one instantaneous
batch. Resume restores partially accumulated logging windows exactly.

## 18. Class weighting and paired stochastic streams

### 18.1 Per-jet representation reduction

`L_jet_direct`, `L_set`, and `L_rel` first produce one value per eligible jet.
They are reduced with the same authenticated square-root inverse-frequency
class weights stored in the parent recipe:

```text
weighted_mean(l_j) = sum_j class_weight[y_j] * l_j
                     / sum_j class_weight[y_j]
```

This prevents the natural QCD majority from owning nearly every
representation gradient while checkpoint selection is macro-AUC first. Labels
are already authorized train-time inputs for CE. The target bank may carry an
audit-only label column for count conservation and the registered shuffled
control, but identity derivation remains label-free and the label column can
never enter a representation target or student forward.

For the batch Gram term, an off-diagonal pair `(i,j)` receives weight
`sqrt(class_weight[y_i] * class_weight[y_j])`, normalized over valid
off-diagonal pairs. Empty-family and ineligible-relation jets are removed from
the relevant denominator rather than assigned a fabricated zero target.

This Gram rule attenuates natural-population majority dominance; it does not
make every observed class-pair cell equal weight. Reports therefore include
the effective Gram weight and squared-error contribution for every observed
unordered class pair. No macro-balanced interpretation is made from the Gram
term itself.

The ordinary logit KD terms retain their parent reduction semantics. This plan
does not retroactively call them class-weighted.

### 18.2 Fair random streams

Each representation node has a canonical counterpart in the parent logit
graph (`RSET_M3c -> M3c`, for example). For a fixed replicate seed:

- cold deployable-backbone initialization uses the counterpart's backbone
  initialization seed;
- sampler, source order, trimming, dropout, augmentation, and validation-order
  seeds use the counterpart's scientific RNG domains;
- RSET and RREL therefore see the same row order and stochastic realizations
  as each other and as the compatible logit control;
- representation-head construction uses a strategy/node-specific auxiliary
  RNG domain (all currently declared square heads still initialize exactly to
  identity); calibration identities and calibration forward RNG use the
  parent-counterpart domains so RSET/RREL common components remain paired;
- the frozen RFF resources are shared across strategies and do not consume a
  training RNG.

Using the full new node ID for every RNG would create unnecessarily unpaired
runs and is forbidden. Warm branches necessarily begin from different
same-branch predecessor weights after M1, but their stochastic input sequence
remains paired by base rung and track.

## 19. Representation overlay recipe and immutable resources

### 19.1 Parent recipe is unchanged

`HCWDL_RECIPE/v3` explicitly defines a logit-only primary graph. It remains
unchanged. HCWDL-RKD introduces `HCWDL_REPRESENTATION_RECIPE/v1`, a
`registered_ablation` overlay that binds the exact authorized parent recipe
hash and may add only the representation semantics in this plan.

The overlay contains no alternate CE/KD coefficient, temperature, batch size,
optimizer, learning rate, class-weight rule, duration, validation cadence, or
checkpoint selector.

### 19.2 Exact representation recipe contents

The overlay binds at minimum:

- strategy IDs `HCWDL_REP_SET/v1` and `HCWDL_REP_REL/v1`;
- all 24 graph nodes and their parent-counterpart mapping;
- `rho_repr=0.10` and every component allocation;
- exact update-coordinate ramps;
- `particle_block_2` and `jet_penultimate` tap names and shapes;
- TOFF charged/neutral and pooled tap semantics;
- token weighting and class-weighted reduction rules;
- top-32 relation population, pT/index ties, `deltaR` strata, and eligibility;
- token and relation RFF dimensions and bandwidths;
- global RFF seed `20260808`, deterministic substream derivation, generated
  FP32 arrays, and logical hashes;
- screening seed `1337`, confirmation seeds `11/22/33/44/55`, shuffled-control
  seed `20260809`, and their disjoint RNG-domain labels;
- identity-initialized bias-free projection policy and `1e-3` orthogonality
  coefficient;
- train-only 4,096-row gradient-calibration population, pass-two jet/set and
  pass-four relation barriers, scale formula, and admissible bounds;
- BF16/FP32 boundary;
- training-only-head reset and deployable-extraction policy;
- target-bank storage/lifecycle policy;
- base graph, recipe, source, split, row-selection, assignment, architecture,
  teacher-import, and producer-source hashes;
- evidence that the zero-coefficient path, exact finite-kernel implementation,
  analytic-gradient, and declared diagnostic-reference tests passed.

The kernel arrays are published once as
`HCWDL_REPRESENTATION_KERNEL_RESOURCES/v1`. Runtime regenerates them from the
declared algorithm/seed and requires exact logical hashes before accepting the
resource. Path existence or a matching seed alone is insufficient.

### 19.3 No mutable overrides

Every scientific value comes from the validated overlay and graph. Slurm
environment variables may point to artifacts but cannot supply coefficients,
layers, kernel dimensions, ramps, token limits, relation strata, teacher
roles, or initialization behavior. A changed value requires a new overlay
contract/version and campaign identity.

## 20. Target-bank sharing and lifecycle

### 20.1 Why a transient durable sketch is justified

Four separate Slurm student jobs cannot share process RAM. Recomputing the
same privileged teacher targets independently for RSET and RREL would waste
offline decoding and could make the variants depend on different teacher
forward realizations. HCWDL-RKD therefore builds one **superset compact target
bank** per selected representation teacher checkpoint/domain and shares it
with all declared consumers.

The bank is not a repaired particle dataset. It contains identities, logits,
pooled vectors, fixed-size kernel means, eligibility bits, and diagnostics
only. It contains no raw HLT/offline particle features, full token activation,
match index, confidence, or alpha.

### 20.2 Ten logical banks

The exact logical banks are:

```text
D0c, D0w,
D25c, D25w,
D50c, D50w,
D75c, D75w,
D100,
TOFF
```

Each ordinary track-specific bank has two consumers: RSET and RREL of the
matching track/rung. The D100 screen generation has eight consumers: four
primary M5 nodes, the two component controls in Section 23.2, and the two
shuffled-pair controls in Section 23.3. The TOFF screen generation has four
primary M6 consumers. A bank stores
the relational superset so RSET and RREL use pooled/token targets from the
exact same teacher forward. RSET does not load or optimize the relation arrays.

### 20.3 Physical format

Introduce:

- `HCWDL_REPRESENTATION_TARGET_LOGICAL_BANK/v1`;
- `HCWDL_REPRESENTATION_TARGET_CONSUMER_REGISTRY/v1`;
- `HCWDL_REPRESENTATION_TARGET_BUILD_INTENT/v1`;
- `HCWDL_REPRESENTATION_TARGET_GENERATION/v1`;
- `HCWDL_REPRESENTATION_TARGET_SHARD/v1`;
- `HCWDL_REPRESENTATION_TARGET_MANIFEST/v1`;
- `HCWDL_REPRESENTATION_TARGET_CLEANUP_AUTHORIZATION/v1`;
- `HCWDL_REPRESENTATION_TARGET_CLEANUP_COMPLETION/v1`.

The logical-bank artifact freezes teacher/target meaning independently of a
physical materialization. A training execution is predeclared from the logical
bank and target purpose, never from a not-yet-created physical generation. Each
screen, confirmation, or recovery materialization has its own immutable
consumer registry and `generation_id` derived from the logical-bank hash,
purpose, consumer-registry hash, and recovery/confirmation parent hash. The
registry can therefore enumerate predeclared execution IDs before the physical
generation exists, and the generation can hash that completed registry without
an identity cycle. Targets are source-sharded,
canonical-identity ordered, contiguous FP32 arrays with explicit
shape/dtype/logical hashes and an immutable generation manifest.
Publication is atomic. Compression may be used only if decompression
reproduces exact FP32 array bytes; compression is operational and recorded,
not part of the scientific identity.

Physical generations never overwrite one another:

```text
targets/<logical_bank>/logical_bank.json
targets/<logical_bank>/staging/<generation_id>/<build_owner_id>/...
targets/<logical_bank>/generations/<generation_id>/
  build_intent.json
  consumer_registry.json
  target_forward_spec.json
  target_execution_attestation.json
  generation.json
  manifest.json
  shards/...
cleanup/<logical_bank>/<generation_id>/
  authorization.json
  completion.json
```

`logical_bank.json` freezes scientific target meaning but contains no mutable
`current` or `latest` pointer. Each consumer receives its exact generation
manifest path through its immutable registry row.

The manifest validates the full selected train-role identity set exactly once
and binds the Section 16 lineage. No target bank is built for validation or
final test.

### 20.4 Just-in-time construction and cleanup

Banks are constructed just before their first eligible rung, not all retained
for the whole campaign. A generation cleanup task uses `afterok` on every
consumer and runs only after every declared consumer has an authenticated
complete report binding that generation. An execution failure is not a
terminal scientific consumer and cannot authorize cleanup.

Cleanup is an exact two-phase state transition. **Before deleting the first
byte**, atomically publish immutable
`HCWDL_REPRESENTATION_TARGET_CLEANUP_AUTHORIZATION/v1`. It binds the generation
manifest, pre-build forward-spec and post-build execution-attestation hashes,
every authenticated consumer report, the complete sorted list of removable
paths with byte/logical hashes and byte totals, the paths that must remain, and
whether exact reconstruction is authorized. Only paths in that authorization
may be deleted. After every removal and retained-file audit succeeds, atomically
publish `HCWDL_REPRESENTATION_TARGET_CLEANUP_COMPLETION/v1`, binding the
authorization plus observed removed/missing/retained sets and byte totals.
Authorization is durable cleanup intent; completion is evidence that the
transition finished. Neither is mutable.

Recovery before authorization reuses only fully validated shards. A missing
shard with no valid authorization is corruption. Once a valid authorization
exists, any listed shard that is missing denotes an interrupted or completed
authorized cleanup, never corruption: recovery first idempotently finishes the
exact authorized deletions and publishes/validates completion. If unfinished
or newly registered consumers require targets, it then publishes a **new**
generation bound to an immutable `HCWDL_REPRESENTATION_RECOVERY_PLAN/v1`,
reconstructs under the same forward spec, validates against the prior execution
attestation and `logical_target_sha256`, runs only that generation's consumers,
and performs a new two-phase cleanup. It never recreates shards inside a
partially deleted old generation. Cleanup never removes model checkpoints,
reports, recipe resources, calibration reports, or scientific metrics. The new
recovery module must schedule `finish old cleanup -> rebuild generation ->
unfinished consumers -> authorize new cleanup -> complete new cleanup`; it
cannot reuse the parent HCWDL transitive retry logic unchanged.

Screen and confirmation are separate physical generations of the same logical
teacher target when cleanup occurs between them. A confirmation rebuild binds
the frozen confirmation registry and its 20 M6 consumers, publishes new shard
byte hashes, and must reproduce the screen generation's
`logical_target_sha256` exactly. It does not create an eleventh scientific
teacher bank or permit a new teacher forward meaning. A second cleanup waits
for all confirmation consumers.

There is no mutable latest-generation index. The immutable campaign spec,
confirmation registry, or recovery plan names the exact generation path each
consumer must load. A separately authorized emergency purge of an abandoned
failed generation is a destructive operational action with its own
authorization/completion audit; it never impersonates a last-consumer
scientific cleanup.

The campaign storage estimator must account for the worst set of banks that
can coexist under the actual DAG. It may serialize bank construction or delay
cleanup for recovery headroom, but it cannot change targets or sample rows.

## 21. Versioned contracts

Implementation introduces distinct contracts rather than broadening the
logit-only HCWDL meanings:

```text
HCWDL_REPRESENTATION_PARENT_IMPORT/v1
HCWDL_REPRESENTATION_PARENT_LOSS_ATTESTATION/v1
HCWDL_REPRESENTATION_ARCHITECTURE_ATTESTATION/v1
HCWDL_REPRESENTATION_ASCENT_GRAPH/v1
HCWDL_REPRESENTATION_RECIPE/v1
HCWDL_REPRESENTATION_KERNEL_RESOURCES/v1
HCWDL_REPRESENTATION_TAP/v1
HCWDL_REPRESENTATION_SURFACE_PARITY/v1
HCWDL_REPRESENTATION_TARGET_FORWARD_SPEC/v1
HCWDL_REPRESENTATION_TARGET_EXECUTION_ATTESTATION/v1
HCWDL_REPRESENTATION_TARGET_LOGICAL_BANK/v1
HCWDL_REPRESENTATION_TARGET_CONSUMER_REGISTRY/v1
HCWDL_REPRESENTATION_TARGET_BUILD_INTENT/v1
HCWDL_REPRESENTATION_TARGET_GENERATION/v1
HCWDL_REPRESENTATION_TARGET_SHARD/v1
HCWDL_REPRESENTATION_TARGET_MANIFEST/v1
HCWDL_REPRESENTATION_TARGET_CLEANUP_AUTHORIZATION/v1
HCWDL_REPRESENTATION_TARGET_CLEANUP_COMPLETION/v1
HCWDL_REPRESENTATION_RECOVERY_PLAN/v1
HCWDL_REPRESENTATION_GRADIENT_CALIBRATION/v1
HCWDL_REP_GRAD_CAL/v1
HCWDL_REPRESENTATION_GRADIENT_CALIBRATION_MANIFEST/v1
HCWDL_REPRESENTATION_DIAGNOSTIC_BATCH/v1
HCWDL_REPRESENTATION_NUMERICAL_ACCEPTANCE/v1
HCWDL_REPRESENTATION_SMOKE_PROBE/v1
HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1
HCWDL_REPRESENTATION_CONTROL_REGISTRY/v1
HCWDL_REPRESENTATION_ZERO_COEFFICIENT_ACCEPTANCE/v1
HCWDL_REPRESENTATION_SHUFFLE_MAP/v1
HCWDL_REPRESENTATION_RESUME_STATE/v1
HCWDL_REPRESENTATION_TRAINING_REPORT/v1
HCWDL_REPRESENTATION_SELECTED_TRAINING_CHECKPOINT/v1
HCWDL_REPRESENTATION_FINAL_TRAINING_CHECKPOINT/v1
HCWDL_REPRESENTATION_CHECKPOINT_SELECTION/v1
HCWDL_REPRESENTATION_DEPLOYABLE_EXTRACTION/v1
HCWDL_REPRESENTATION_SCREEN_AGGREGATE/v1
HCWDL_REPRESENTATION_CONFIRMATION_REGISTRY/v1
HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v1
HCWDL_REPRESENTATION_CONFIRMATION_RUN/v1
HCWDL_REPRESENTATION_VALIDATION_ONLY_AGGREGATE/v1
HCWDL_REPRESENTATION_FINAL_DISPOSITION/v1
HCWDL_REPRESENTATION_PARENT_FINAL_STATE/v1
HCWDL_SHARED_IMMUTABLE_BINARY_ENVELOPE/v1
HCWDL_SHARED_FINAL_POPULATION/v1
HCWDL_SHARED_FINAL_POPULATION_DISJOINTNESS/v1
HCWDL_SHARED_FINAL_EXPOSURE_LEDGER/v1
HCWDL_SHARED_LEGACY_FINAL_EXPOSURE/v1
HCWDL_SHARED_FINAL_POPULATION_REGISTRATION/v1
HCWDL_SHARED_FINAL_RESERVATION/v1
HCWDL_SHARED_FINAL_LEGACY_CANCELLATION/v1
HCWDL_REPRESENTATION_FINAL_ASSIGNMENT_SPEC/v1
HCWDL_REPRESENTATION_PRETRAINING_FINALIST_POLICY/v1
HCWDL_REPRESENTATION_FINALIST_LOCK/v1
HCWDL_SHARED_FINAL_TASK_REGISTRY/v1
HCWDL_SHARED_FINAL_EXECUTION_CLAIM/v1
HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1
HCWDL_SHARED_FINAL_RECOVERY_PLAN/v1
HCWDL_SHARED_FINAL_ROW_SELECTION/v1
HCWDL_SHARED_FINAL_LABEL_ESCROW/v1
HCWDL_SHARED_FINAL_ASSIGNMENT_SHARD/v1
HCWDL_SHARED_FINAL_BRANCH_ACCESS/v1
HCWDL_SHARED_FINAL_ASSIGNMENT_AUDIT/v1
HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1
HCWDL_REPRESENTATION_EXECUTION_LOCK/v1
HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v1
HCWDL_REPRESENTATION_FINAL_EVALUATION/v1
HCWDL_REPRESENTATION_PREDICTION_SHARD/v1
HCWDL_REPRESENTATION_PREDICTION_MANIFEST/v1
HCWDL_REPRESENTATION_METRIC_JOIN/v1
HCWDL_REPRESENTATION_FINAL_AGGREGATE/v1
HCWDL_REPRESENTATION_CAMPAIGN_SPEC/v1
HCWDL_REPRESENTATION_COMMAND_PLAN/v1
HCWDL_REPRESENTATION_RUNTIME_BINDING/v1
HCWDL_REPRESENTATION_RUNTIME_PREREQUISITES/v1
HCWDL_REPRESENTATION_RUNTIME_DRY_RUN_AUDIT/v1
HCWDL_REPRESENTATION_WORKER_RUNTIME_MEASUREMENT/v1
HCWDL_REPRESENTATION_EXECUTABLE_CANDIDATE_AUDIT/v1
HCWDL_REPRESENTATION_SUBMISSION_EVENT/v1
HCWDL_REPRESENTATION_SUBMISSION_LEDGER/v1
HCWDL_REPRESENTATION_RECOVERY_SUBMISSION_LEDGER/v1
HCWDL_REPRESENTATION_MONITOR_REPORT/v1
HCWDL_REPRESENTATION_RESOURCE_PROFILE/v1
HCWDL_REPRESENTATION_STORAGE_ESTIMATE/v1
HCWDL_REPRESENTATION_FIXED_SIZE_INVENTORY/v1
HCWDL_REPRESENTATION_SCHEDULER_EVIDENCE/v1
HCWDL_REPRESENTATION_MINIATURE_EVIDENCE/v1
HCWDL_REPRESENTATION_TIGRIS_EVIDENCE_BUNDLE/v1
HCWDL_REPRESENTATION_TIGRIS_ACTION_PROOF/v1
HCWDL_REPRESENTATION_USR1_EXACT_RESUME_PROOF/v1
HCWDL_REPRESENTATION_VALIDATION_PROXY_PROOF/v1
HCWDL_REPRESENTATION_PRODUCTION_WORKER_SMOKE_PROOF/v1
HCWDL_REPRESENTATION_LOCAL_SMOKE_REPORT/v1
HCWDL_REPRESENTATION_CACHE_MINIATURE/v1
HCWDL_REPRESENTATION_CACHE_MINIATURE_BANK/v1
HCWDL_REPRESENTATION_TIGRIS_ACCEPTANCE/v1
HCWDL_REPRESENTATION_SUBMISSION_AUTHORIZATION/v1
HCWDL_REPRESENTATION_ACCEPTANCE_BOOTSTRAP/v1
```

`HCWDL_REPRESENTATION_ACCEPTANCE_BOOTSTRAP/v1` is a deliberately bounded
operational authority for collecting the first nonfinal acceptance evidence.
It is not a campaign spec, executable-candidate audit, submission
authorization, or pilot authorization. It binds the exact planning spec,
runtime binding, clean source commit, bootstrap-worker bytes, conservative
smoke resources, bounded role counts, and a dependency-closed prefix of
nonfinal acceptance tasks. It cannot name a final-role route, reserve or claim
a shared-final population, train a ladder node, register a scheduler mutation,
or satisfy the executable-pilot gate by itself.

The worker-runtime measurement contract and its embedded live-runtime and
row-signature hash domains close the review-time/runtime boundary. An allocated
worker measures its exact clean checkout bytes and commit, active Conda/Python
isolation, installed-Weaver
source bytes, package stack, device, and backend before registered scientific
I/O. The reusable row signature excludes only the audit-only physical GPU UUID;
target execution attestations record that freshly measured UUID. A measurement
artifact is immutable but explicitly nonauthorizing and scheduler-neutral.

Every reusable JSON artifact has a canonical content hash and explicit parent
hashes. Binary objects record byte and logical hashes. Old PMARD representation
arms, `HCWDL_GRAPH/v1`, and `HCWDL_RECIPE/v3` remain valid under their existing
consumers and are never accepted as if they already implemented this plan.

Every non-resume binary-plus-JSON artifact uses the common
`HCWDL_SHARED_IMMUTABLE_BINARY_ENVELOPE/v1` publication protocol. Validate all
immutable parents first; derive deterministic `envelope_id` as the SHA-256 of
Section 9.3 canonical JSON over `{contract, producer_task_id, schema,
immutable_parent_hashes, registered_output_row}` and `envelope_owner_id` over
that payload plus the campaign/recovery owner. Neither identity depends on
not-yet-computed output bytes; the committed sidecar supplies those hashes. Write
payloads, sidecar, branch-access record when applicable, and `commit.json` into
`staging/<envelope_id>/<envelope_owner_id>/`; fsync every file and the complete
tree; then atomically rename that nonempty directory to
`committed/<envelope_id>/` and fsync the committed parent. The committed
directory appearance is the sole publication point. `commit.json` binds every
member's relative path, byte hash, logical hash, dtype/shape where applicable,
and parent hash. A final-path payload outside a committed envelope is never an
artifact.

Restart with the same owner/payload validates and continues only missing
staged members or validates an already committed envelope; a different owner
cannot adopt staging. A conflicting committed envelope or invalid committed
member is corruption and is never overwritten. Orphan staging may be removed
only under the exact-owner recovery plan. This protocol is mandatory for the
kernel-resource pair, shuffle-map pair, final label escrow, final assignment
shards, final prediction shards, and paired-bootstrap binary sidecars. Target
generations use the stronger whole-generation directory commit in Section 29;
resume states use the sequence protocol below. Tests inject death before and
after every member write/fsync, before and after `commit.json`, and before and
after the directory rename.

The three binary-bearing control/state families are explicit rather than
implicit Python conventions:

- `HCWDL_REPRESENTATION_RESUME_STATE/v1` is a crash-atomic immutable generation,
  never a replace-in-place PT/JSON pair. Generation `q` contains
  `state_q.pt`, `state_q.json`, and `commit_q.json`; the PT and sidecar are
  atomically renamed first, and the commit record binding both hashes is
  atomically published last. Resume scans committed sequence numbers and loads
  the highest fully valid generation. Orphan PT/JSON files without a commit are
  ignored and audited. Only after a new commit validates may generations older
  than the immediately previous committed state be deleted, so at least two
  recoverable states survive cleanup. The sidecar enumerates every required
  state namespace, tensor key/shape/dtype, logical-state hash, serialized byte
  hash, completed pass/update, next canonical batch, sampler/trimmer/RNG state,
  optimizer and scheduler state, partial interval aggregates, active
  projections, calibration artifacts, exact target generation/logical hashes,
  and producer runtime signature. Resume rejects missing or unexpected state;
  a report path or uncommitted file pair is never a resume contract.
- `HCWDL_REPRESENTATION_CONTROL_REGISTRY/v1` contains the exact four M5 control
  node definitions, their parent-counterpart mappings, target consumers, loss
  allocations, and validation-only disposition. The 24-node primary graph does
  not silently absorb them.
- `HCWDL_REPRESENTATION_SHUFFLE_MAP/v1` is a JSON sidecar plus compact integer
  mapping array. It binds seed `20260809`, train identity order/hash, label and
  row-selection hashes, class offsets, source/target identity hashes, exact
  cyclic-derangement algorithm, and proves no fixed point or cross-class edge.
  Both shuffled strategies consume the same authenticated mapping.

Likewise, prediction shards use the shared committed binary/JSON envelope:
canonical
identity digest plus finite FP32 `[rows,15]` logits only--no probabilities or
labels--with exact finalist and checkpoint hashes, row/order/set hashes, and
producer runtime signature. The
prediction manifest proves complete disjoint final-role coverage before the
single locked label/metric join.

## 22. Campaign DAG

### 22.1 High-level graph

```text
implement named taps + installed-Weaver parity
  -> representation architecture attestation
  -> validate parent HCWDL prefix/loss semantics
  -> representation parent-import lock
  -> kernel resources + representation overlay recipe
  -> representation local/real-worker miniature
  -> rung-serialized D0c/D0w target banks
       -> RSET_M1c -> RSET_M2c -> ... -> RSET_M6c
       -> RREL_M1c -> RREL_M2c -> ... -> RREL_M6c
       -> RSET_M1w -> RSET_M2w -> ... -> RSET_M6w
       -> RREL_M1w -> RREL_M2w -> ... -> RREL_M6w
  -> four registered M5-cold mechanism controls after their M4 parents/D100
  -> screen aggregate
  -> predeclared confirmation controls/replicates
  -> combined finalist/execution boundary, if test remains sealed
  -> HLT-only final inference
  -> final aggregate
```

The abbreviated graph hides mandatory just-in-time storage barriers. Within a
rung, track-specific banks run in deterministic `cold -> cleanup -> warm ->
cleanup` order after their selected teachers exist; they are not simultaneously
durable. Every bank consumer depends on its exact committed generation;
cleanup depends on all registered consumers. The **next rung's bank-build
readiness task depends on completion attestations for every bank in the prior
rung**, as well as its selected teacher. Thus the exact chain is
`build D0c -> cold M1 consumers -> clean -> build D0w -> warm M1 consumers ->
clean -> build D25c -> ... -> clean D100 -> build TOFF`. TOFF confirmation is a later physical generation
that depends on screen-generation cleanup. No D25-or-later screen bank may
build merely because its downward teacher already exists. Student branches
remain sequential internally because predecessor logits and warm weights come
from their own prior rung. For an ordinary cold or warm bank, its two same-bank
RSET/RREL consumers may run concurrently. Cold and warm consumers cannot run
concurrently because warm-bank readiness requires cold-bank cleanup. The shared
D100 bank may fan out to its eight registered M5 primary/control consumers, and
the shared TOFF bank to its four registered M6 consumers; those are the only
larger same-generation waves. Bank lifecycles may not cross the frozen storage
barrier. This serialization is
intentional because the evidence-backed 1,024/256-dimensional sketches favor a
stronger finite kernel while respecting the user's limited durable storage.

### 22.2 Exact 24-node registry

`HCWDL_REPRESENTATION_ASCENT_GRAPH/v1` contains exactly 24 training nodes. It
stores strategy, track, rung, student domain, initialization parent,
predecessor-logit teacher, representation/logit teacher, parent logit-node
counterpart, target-bank identity, and deployability explicitly. Runtime never
infers these fields from a node-name suffix.

Graph validation requires:

- M1 has D0 only and never M0;
- M2–M6 use their own strategy/track predecessor;
- the privileged source follows D25/D50/D75/D100/TOFF exactly;
- no representation target comes from the predecessor at M2–M6;
- every student domain is HLT;
- cold nodes have no initialization parent;
- warm nodes load same-strategy immediate predecessors, with warm M1 loading
  track-matched D0;
- no cycle, cross-track edge, cross-strategy predecessor, or missing rung;
- every target bank has exactly its declared consumers and cleanup dependency.

### 22.3 Modes

- Smoke executes all distinct target/tap/loss/extraction paths on bounded rows
  and two optimizer updates, plus the explicit full-loss backward probe.
- Pilot executes all 24 primary nodes plus the four M5 controls on 300k/100k
  with one screening seed.
- Production executes all four complete ascents and the same four controls on
  the complete parent roles.
  A later decision to run only a validation-selected subset is a different,
  newly versioned campaign and cannot claim to be this four-ascent plan or be
  represented by an unsatisfiable dependency.

No mode opens final-test particles while building representation targets.

## 23. Controls and causal checks

The four full ascents answer the system-level question. The following compact
controls prevent a misleading implementation-level interpretation without
creating more complete ladders.

### 23.1 Zero-coefficient wrapper parity

On deterministic synthetic and real miniature batches, an ordinary HLT ParT
and the representation wrapper with `rho_repr=0` and identical backbone must
produce:

- identical logits within the authoritative forward tolerance;
- identical CE/KD values;
- identical gradients for every shared backbone/classifier parameter;
- no representation-head contribution to logits;
- identical trimmer/RNG progression and optimizer update.

This is a required implementation control, not a selectable scientific model.

### 23.2 Jet-only and no-relation current-rung controls

Register two 60-pass, validation-only M5-cold controls:

```text
RSET_M5c_JET_ONLY_REP
RREL_M5c_NO_REL_REP
```

`RSET_M5c_JET_ONLY_REP` uses the exact `RSET_M4c` predecessor logits, D100
privileged logits/jet target, M5c cold backbone initialization, base loss,
row/stochastic stream, pass-two calibration barrier, ramp, `rho_repr`, and
selector of `RSET_M5c`. Its representation objective is:

```text
scheduled(L_repr) = r_js * (1.00 Lhat_jet + 1e-3 L_orth)
```

It instantiates no token projection or set/relation target in RAM. The paired
`RSET_M5c - JET_ONLY` delta asks whether allocating the current rung's
representation coefficient to the unordered token set is better than using
only the paired pooled-jet target. It does not decompose accumulated earlier
RSET rungs.

`RREL_M5c_NO_REL_REP` uses the exact `RREL_M4c` predecessor and every other
base/init/stream/teacher setting of `RREL_M5c`, but retains the entire
scheduled representation allocation in jet/set:

```text
scheduled(L_repr) = r_js * (0.40 Lhat_jet + 0.60 Lhat_set
                            + 1e-3 L_orth)
```

It has no relation calculation or pass-four relation calibration. Comparing
`RREL_M5c` with this control isolates whether transferring up to 25% of the
same current-rung scheduled coefficient from jet/set to latent relations is
better under the already-accumulated RREL predecessor. Neither compact control
has descendants, enters the primary 24-node count, or becomes a final-test
candidate.

### 23.3 Within-class shuffled-pair control

Register two validation-only terminal controls:

```text
RSET_M5c_WITHIN_CLASS_SHUFFLED_REP
RREL_M5c_WITHIN_CLASS_SHUFFLED_REP
```

They use the correct labels, correct predecessor logits, correct privileged
logits, exact cold initialization/stochastic streams of their unshuffled M5
counterpart, and the same representation loss. Only representation target
identities are replaced by a deterministic no-fixed-point permutation within
the same true class. Within each class, sort train identities by the digest of
the Section 9.3 canonical-JSON UTF-8 payload
`{"contract":"HCWDL_REP_SHUFFLE/v1","identity_sha256":"<hex>",
"master_seed":20260809}` and give zero-based sorted position `i` the
jet/set/relation representation targets from position `(i + 1) mod n`.
Every class must contain at least two selected train rows; otherwise the
control fails structurally. The privileged logits stay joined to the correct
identity, and validation targets remain unused.

Each shuffled control performs its **own** Section 13 calibration after the
shuffle map has been applied, at the same pass-two/pass-four barriers and on
the same canonical calibration identities as its paired primary node. Its
calibration artifacts bind the shuffle-map hash and shuffled target joins;
they never reuse the unshuffled primary scales. Recalibration is the
deterministic normalization consequence of the sole exogenous shuffle
intervention and keeps the nominal local auxiliary coefficient comparable. No
validation value enters it. If shuffled support is weak, the exact
inactive-component rule in Section 13.3 applies.

This preserves coarse class information while destroying exact jet pairing.
If shuffled targets perform as well as paired targets, any gain is generic
class-conditioned regularization rather than evidence for paired privileged
structure.

M5 cold is chosen before results because it is the strongest Shell-Exact
teacher rung, avoids TOFF's separate charged/neutral architecture as an extra
mechanism change, and permits exact pairing with the ordinary HLT-token
surface. Running the control only at M5 keeps it diagnostic rather than adding
another pair of six-rung cascades. It does not establish that every earlier or
warm rung has the same mechanism.

### 23.4 Finite-kernel and invariance controls

Required noncampaign controls include:

- exact frozen finite-kernel MMD in its slow pairwise form versus squared
  feature-mean distance in its cached form on 512 deterministic token fixtures
  from numerical seed `991`: independently draw
  set sizes in `[2,32]`, unit-normalized 128-vectors from a random unit center
  plus standard-normal noise at a log-uniform scale in `[0.02,0.8]`, and
  `Uniform(0,1)` weights normalized to sum one. Require FP64 maximum absolute
  disagreement `<=1e-10` and FP32 runtime disagreement `<=1e-5`;
- the analogous 512 relation fixtures from numerical seed `992`, with set
  sizes in `[2,496]`, center `Uniform(-0.8,0.8)`, standard-normal noise at a
  log-uniform scale in `[0.01,0.5]`, values clipped to `[-1,1]`, and normalized
  `Uniform(0,1)` weights. Require the same finite-kernel pairwise/cached
  equivalence tolerances;
- analytic finite-kernel **student gradients** versus PyTorch autograd on 64
  additional small token fixtures from seed `993` and 64 relation fixtures
  from seed `994`, each with set sizes in `[2,16]`, including the runtime
  `F.normalize` Jacobian. For gradient norm at least `1e-8`, require flattened
  cosine `>=0.99999`, norm ratio in `[0.999,1.001]`, and maximum absolute entry
  error `<=1e-5`; a near-zero analytic fixture requires autograd norm `<=1e-6`;
- ideal infinite-four-RBF **value** quality on the 512 seed-`991` token and
  512 seed-`992` relation value fixtures defined in the first two bullets--not
  on the smaller seed-`993`/`994` gradient fixtures. Token
  MAE/p95/correlation must be `<=0.015/<=0.040/>=0.990` (audited
  `0.009032/0.026687/0.997016`); relation must be
  `<=0.030/<=0.100/>=0.985` (audited `0.019179/0.076771/0.992085`). Ideal-RBF
  gradient comparison is always report-only, so the finite loss cannot be
  described as infinite-RBF gradient fidelity. Relation's
  stronger ideal-gradient agreement is also diagnostic, not a different
  acceptance meaning;
- finite-kernel joint-rotation sensitivity on 64 token fixtures from seed `995`.
  Generate a deterministic Haar orthogonal 128-by-128 matrix by FP64 Gaussian
  QR with positive diagonal convention, rotate both student and teacher token
  clouds, and compare with the unrotated fixed-resource result. Exact RBF MMD
  must be invariant to numerical tolerance; the finite kernel is expected to
  change because its resource coordinates are fixed. Require finite-kernel loss
  change mean `<=0.020` and p95 `<=0.065` (audited
  `0.013436/0.047950`). Publish the transformed-gradient cosine/norm
  report-only--the audited median cosine is `0.0318`--and make no gradient
  rotation-invariance claim. This bounds value sensitivity while exposing
  rather than denying the finite-map basis artifact; the learned `A_tok`, fixed
  teacher checkpoint, and predeclared resource seed define the actual estimand;
- token permutation invariance;
- exact Gram/relation invariance under a shared orthogonal rotation and a
  report-only synthetic demonstration that anisotropic/nonlinear transforms
  are not invariant, preventing an overbroad coordinate-free claim;
- padding invariance;
- unequal-count and independently reordered sets;
- one-token and empty-family behavior;
- deterministic top-32/tie selection;
- nonzero relational gradient to student states;
- zero encoder gradient from a physics-only pair loss, proving that prohibited
  formulation cannot masquerade as representation KD.

Fixture generation is executable, not prose-dependent. Every stream uses
`numpy.random.Generator(PCG64(seed))`, and NumPy's exclusive-high integer
semantics. For each token fixture, in order: draw student and teacher counts
independently; draw one 128-dimensional standard-normal center and unit
normalize it; draw independent student/teacher log-scales uniformly between
the written log endpoints; draw the complete student then teacher
standard-normal noise arrays; form `center + scale * noise` and unit-normalize
each row; draw the complete student then teacher `Uniform(0,1)` weight arrays
and normalize each separately. For each relation fixture, use the same field
order with one scalar uniform center, independent log-scales, student then
teacher scalar noise arrays, clipping, and student then teacher weights. Counts
are drawn with `integers(2, 33)` for token-value fixtures,
`integers(2, 497)` for relation-value fixtures, and `integers(2, 17)` for both
gradient families. Analytic/autograd and ideal-reference comparisons treat the
completed student cloud as the FP64 leaf requiring gradient; the teacher cloud,
weights, and frozen resources are detached and byte-identical across compared
computations.

For seed `995`, draw the full 128-by-128 Gaussian matrix first, apply FP64 QR,
and multiply each Q column by the sign of the corresponding R diagonal
(`zero -> +1`); then draw the 64 token fixtures in the token field order above.
The same Q is applied jointly to both clouds for every fixture. No draw is
discarded or retried.

The exact fixture generator, distribution parameterization, serialized seed
payloads, selected-resource hashes, unrounded error/gradient vectors, and
pass/fail values are published as
`HCWDL_REPRESENTATION_NUMERICAL_ACCEPTANCE/v1`. The resource seed is never
searched for one that passes. Failure requires a new predeclared recipe; it
cannot be repaired after viewing model metrics. All draws, finite-kernel
references, and ideal-RBF diagnostics are FP64.

### 23.5 What is not a control

The logit-only cold/warm ascents are full scientific controls. `RSET` is the
equal-budget full-system control for `RREL`. A bad result, a zero incremental gain,
or an adapter close to identity is not an execution failure.

## 24. Reporting and comparisons

### 24.1 Required classification metrics

Every node reports the parent HCWDL metric set: CE, overall and balanced
accuracy, macro/per-class OVR AUC, mean/per-class log QCD rejection at 50%
signal efficiency, explicit Xbb/QCD and Xcc/QCD FPR/rejection, ECE, Brier,
confusion matrix, class counts, and always-QCD accuracy context.

Checkpoint selection remains macro-AUC first. Reports preserve exact FP64 and
hexadecimal selector inputs.

### 24.2 Representation diagnostics

Every validation boundary additionally records report-only train-interval
means for:

- unscaled and calibrated jet, set, relation, and orthogonality loss;
- active representation ramp values and effective coefficients;
- gradient-calibration scales and current component gradient norms on a fixed
  diagnostic microbatch without changing training RNG, including achieved
  combined-representation/base gradient norm ratio and cosine on the common
  `Theta_cal` support;
- projection singular-value minimum/maximum and condition number;
- token counts and scalar-pT represented in each set target;
- M6 charged/neutral/unclassified fractions by class and the exact
  PID/charge reason counts for every unclassified token;
- set-loss eligibility, denominator weight, and effective sample size by
  class/family;
- pair counts, active/empty relation strata, and relation effective sample
  size by class/family/stratum;
- effective Gram weight and squared-error contribution by unordered class
  pair;
- teacher-target join count, cache bytes, construction/read time, and hashes;
- BF16-forward/FP32-loss finite checks.

The diagnostic microbatch is exact. It is the first canonical 256-identity
batch of the Section 13 digest-ordered calibration population, materialized
once from train-only inputs and the node's exact target generation. At every
post-validation safe boundary, first complete any calibration barrier required
at that pass under Section 13.2, then run the live student/heads in `eval()`
mode with the node's BF16-forward/FP32-loss policy and one shared student
forward for base and all components. Use the Section 13 matched-support
reductions and `Theta_cal`; an ineligible or not-yet-calibrated component
reports `null` plus the exact status/counts, never an invented zero or a scale
from the future. Teacher values are cached/detached. The diagnostic precedes
the boundary/resume commit.

Before the diagnostic, snapshot model mode and buffers, CPU/CUDA RNG, trimmer,
sampler, optimizer/scheduler, data/target cursors, and logging state. Run inside
`torch.random.fork_rng` seeded from the Section 9.3 canonical payload
`{contract: HCWDL_REP_DIAGNOSTIC/v1, execution_id, completed_update}`; use
`autograd.grad` with no `.grad` accumulation and no optimizer step. Restore the
snapshot and require byte/logical equality of every state and the identity of
the next scientific training batch. The diagnostic input has its own immutable
manifest and cursor, so it never advances a training/cache iterator. A
with-diagnostic versus without-diagnostic exact-resume fixture must produce
identical subsequent checkpoints. These report-only values cannot fail a
finite row or select a checkpoint.

These values never select a checkpoint or fail a finite scientifically weak
model. Only violated structural/numerical contracts fail execution.

### 24.3 Ordered comparisons

For every rung and track, report paired validation deltas:

```text
RSET - logit-only
RREL - logit-only
RREL - RSET
RSET_M5c - RSET_M5c_JET_ONLY_REP
RREL_M5c - RREL_M5c_NO_REL_REP
paired M5c - within-class-shuffled M5c
warm - cold within each strategy
child - immediate same-branch predecessor
M6 - M1 and M0
```

At later rungs, `RREL - RSET` is a **complete-system** difference: it includes
the accumulated effect of different earlier representation-trained
predecessors, not only the current relation term. The M5 shuffled controls and
per-rung histories help interpret mechanism but do not turn the cascade into a
single-loss causal estimate.

Gap-recovery tables use M0/D0 or the corresponding logit model as lower anchor
and D100/TOFF as declared upper oracle, with the parent sign convention for CE.
Undefined or reversed oracle denominators remain explicitly undefined.

For a higher-is-better metric `h` and lower-is-better CE `c`:

```text
recovery_h = (h_student - h_lower) / (h_upper - h_lower)
recovery_c = (c_lower - c_student) / (c_lower - c_upper)
```

A nonpositive denominator under the declared direction yields `null` plus a
reason, never infinity, clipping, or a reversed percentage. Values below zero
or above one remain un-clipped scientific results.

### 24.4 Statistical confirmation

The pilot screen uses seed `1337` for all four full ascents. After all 24
primary nodes and four M5 controls complete, a frozen confirmation registry
reruns the four M6 terminal
objectives for seeds `11`, `22`, `33`, `44`, and `55` while holding each
branch's selected M5
teacher/initialization parent and the privileged teacher fixed. Cold M6 uses a
fresh seed-specific backbone; warm M6 reloads its branch's selected M5
deployable backbone and varies the new-run stochastic streams.

Every training run, including the seed-1337 screen, has an immutable
`execution_id = SHA256(Section 9.3 canonical JSON of {campaign, strategy,
node_id, purpose, seed, initialization_parent, teacher, logical_target_bank,
target_purpose, recipe})`. `logical_target_bank` is the authenticated logical
bank hash; `target_purpose` is exactly `screen` or `confirmation`. A physical
`target_generation` is deliberately excluded because it is operationally
reconstructible and its consumer registry already contains these execution
IDs. Calibration phases/manifests, resume generations, checkpoints, reports,
and selection artifacts are all namespaced by execution ID and separately bind
the realized physical generation ID, its consumer-registry hash, and its stable
`logical_target_sha256`. Once an execution publishes its first calibration or
resume artifact, that execution cannot switch physical generations; a rebuild
may serve that same predeclared execution ID only when it has no committed
training state. A partially trained execution necessarily prevents its target
generation's last-consumer cleanup and therefore resumes against the original
generation. `node_id` alone is never an output path. Thus the five confirmation
executions of one M6 node cannot
collide with each other or with its screening execution. The confirmation
registry enumerates the 20 exact execution IDs before its consumer registry and
physical generation are constructed.

This estimates conditional terminal-rung training variance, not full-ladder
seed variance, and reports that limitation. A future full-system replicate
requires four additional complete ascent graphs and a new authorization.

`HCWDL_REPRESENTATION_CONFIRMATION_AGGREGATE/v1` has one exact conditional
estimand per branch and classification metric: the arithmetic mean across the
five seed-specific point estimates, sample standard deviation with `ddof=1`,
standard error `s/sqrt(5)`, and two-sided 95% Student-t interval
`mean +/- t_(0.975,4) * s/sqrt(5)`. It also reports all five raw values and the
five paired seed-wise differences for each RREL-minus-RSET and warm-minus-cold
contrast, summarized by the same formula. Seeds pair only when their integer
seed is equal. There is no pass/fail improvement criterion, multiple-testing
claim, best-seed choice, or finalist selection from this aggregate; it is
conditional terminal-M6 training variance around fixed screening-seed M5
parents, not full-ladder uncertainty.

Paired bootstrap uses a new explicit
`HCWDL_REPRESENTATION_PAIRED_BOOTSTRAP/v1` surface; neither an existing
10-class stratified helper nor an unstratified scalar helper is compatible.
The input join must contain the same identity-ordered prediction row for every
model and at least one observed row in each of the 15 frozen classes. Create
exactly 2,000 replicates from
`numpy.random.Generator(PCG64(8041))`. For each replicate and each class in
canonical class-code order, draw `n_c` integer offsets with replacement from
that class's `n_c` observed rows using `integers(0, n_c, size=n_c)`. Concatenate
those class blocks; the resulting identity index vector is used unchanged for
every paired model and delta in that replicate. No cross-class resampling,
row-level independent model draw, retry, or metric-specific bootstrap is
allowed.

Store replicate values and paired deltas for CE, overall and balanced
accuracy, macro OVR AUC, macro mean log-QCD-rejection at 50% signal efficiency,
top-label ECE-15, multiclass Brier score, and every class's OVR AUC plus
50%-efficiency QCD FPR/rejection. Report the original-population point estimate,
bootstrap median, and 2.5%/97.5% intervals using NumPy's `method="linear"` on
unrounded FP64 replicate values. The rejection policy is bound explicitly to
the current `evaluation.py` convention: store raw `qcd_pass`; empirical FPR is
`qcd_pass / N_qcd`; rejection is `N_qcd / max(1, qcd_pass)`; and log rejection
is `log(max(rejection, 1.0))`. Because within-class resampling preserves a
positive count for every class, every OVR AUC and `N_qcd` denominator is
defined. Any other nonfinite metric is a contract failure for that prediction
join, not a dropped bootstrap replicate or redraw. The artifact stores the exact joined-identity hash, prediction/label
parent hashes, RNG metadata, replicate-index logical hash, full replicate
arrays or losslessly equivalent immutable binary sidecar, and every registered
comparison sign.

## 25. Final-test boundary

### 25.1 Population-scoped disposition and pre-training reservation

The original logit-only campaign and HCWDL-RKD use different campaign roots,
so a root-local `execution_claim.json` cannot prevent a race. First create
`HCWDL_SHARED_FINAL_POPULATION/v1` without opening any ROOT branch. It hashes
the complete ordered set of canonical `(source_file_sha256, source_entry)`
identities in the parent split's **entire mapped final-test role**, not the
later selected subset. The common secrecy namespace is:

```text
<checkpoint_namespace>/final_claims/<final_population_sha256>/
```

Selection rules, selected rows, and label contracts are bound *inside* this
namespace and cannot create a new namespace over the same population. A
proposed new holdout additionally publishes
`HCWDL_SHARED_FINAL_POPULATION_DISJOINTNESS/v1` against every population in the
common exposure ledger; any identity overlap fails closed. A changed split
name or label policy is never proof of a new holdout.

Population registration and overlap checking are one serialized transaction,
not a check-then-write pair. The proposal has deterministic
`registration_owner_id = SHA256(canonical JSON of the population, proposed
campaign identity, disposition request, and selection/label/assignment rule
commitments)`. A neutral shared registrar takes an exclusive OS-level lock on
the one fixed exposure-ledger lock file and executes this crash-recoverable
order:

1. validate `HEAD` and the full immutable
   `HCWDL_SHARED_FINAL_EXPOSURE_LEDGER/v1` chain, then check the proposed
   identity set against every entry;
2. write population, disjointness, and registration payloads into an
   owner-scoped proposal directory, fsync its files/tree, and atomically rename
   the whole directory into the population namespace;
3. publish/fsync the next immutable ledger generation binding that directory;
4. atomically replace/fsync `HEAD` to point to that generation.

The registrar does not treat an intermediate same-owner state as a second
registration. On restart under the lock, it recognizes the exact owner/payload
at every boundary and idempotently completes only the missing rename, ledger
generation, or `HEAD` swap. A different owner/payload, conflicting immutable
file, or ledger entry without its exact bundle fails closed. After the campaign
spec/disposition audit, the same owner publishes the reservation as a separate
atomic file under this lock; absent reservation is resumable, exact existing
reservation is idempotent, and any difference fails. This ownership claim
prevents a second campaign from reserving the already registered population
while never stranding it after a coordinator crash.

No campaign-root code may register directly. The real-filesystem miniature
must prove the selected lock/directory-rename/file-rename/fsync primitives on
Tigris; if the filesystem cannot provide them, final execution is blocked
rather than falling back to an unlocked scan. Tests inject death before/after
every step and race both identical populations and different population hashes
with at least one overlapping identity; exactly one owner may complete
registration/reservation. Here `source snapshot` always means the authenticated
**data** source snapshot; producer source commit is a separate bound field.

Before the representation campaign spec is created, audit the parent
submission ledger, all legacy final-task Slurm IDs, final outputs, root-local
claims, the population namespace, and exposure ledger. Freeze exactly one
campaign-spec disposition:

```text
combined_confirmatory
validation_only_parent_claim_consumed
```

`combined_confirmatory` is legal only when no prediction, model-derived final
output, or execution claim exists for any overlapping population. Every
pending/running legacy parent final job is cancelled by exact ledger-bound job
ID; `HCWDL_SHARED_FINAL_LEGACY_CANCELLATION/v1` records the IDs, scheduler
states after cancellation, output audit, and proof that no worker can race.
Before training starts, atomically publish
`HCWDL_SHARED_FINAL_RESERVATION/v1` in the population namespace. It binds source,
split, full population, exact class-stratified parent selection rule, label
contract, Shell-Exact assignment specification, combined campaign spec, and a
pre-training finalist-registry commitment: complete parent-union policy, four
representation endpoint IDs, screening seed, and selector rules, but not
unknown checkpoint hashes.

`validation_only_parent_claim_consumed` is mandatory if a prior claim/output,
un-cancellable worker, overlap, or ambiguous external state exists. Its
immutable task registry contains no population claim, selection, assignment,
prediction, metric join, or final aggregate task. Disposition is decided at
spec creation, never by a late DAG `if`.

### 25.2 Exact combined finalist registry

After validation/confirmation, the combined registry is the union of the
complete parent finalist registry and the new representation endpoints. It
cannot drop parent-selected intermediates or null controls merely to shorten
this plan. It contains:

- every deployable checkpoint and nondeployable oracle/control named by the
  parent confirmation/finalist policy, including M0, logit M6 cold/warm,
  validation-selected parent intermediates, required self/predecessor/warm
  null controls, D100, and TOFF;
- the screening-seed (`1337`) selected checkpoints for RSET M6 cold/warm and
  RREL M6 cold/warm.

The 20 seed-`11/22/33/44/55` representation confirmations estimate conditional
terminal variance and are not finalists. No best-seed selection is performed.
The four M5 controls are validation-only. A different endpoint/seed policy
requires a new reservation on a genuinely untouched disjoint population.

`HCWDL_REPRESENTATION_FINALIST_LOCK/v1` binds both campaign hashes, full union
registry, recipes, loss/architecture attestations, source/split/full-population
hashes, selection-rule and assignment-spec commitments, every checkpoint and
extraction report, shared reservation, and legacy-cancellation proof. It does
**not** pretend that selected final rows or final assignments exist yet. Those
are authenticated after the shared claim in Section 25.3. All six deployable
M6 endpoints are predeclared estimands inside the larger parent-union wave.

### 25.3 Shared claim, exact task registry, and recovery

Freeze `HCWDL_SHARED_FINAL_TASK_REGISTRY/v1` and its submission ledger before
any final branch opens. It enumerates every logical/array row, purpose,
authorized branch family, source partition, finalist/checkpoint, output path,
resource signature, and dependency in this exact graph:

```text
shared_claim_gate
  -> final_row_selection_and_label_escrow
  -> final_assignment_shards[]
  -> final_assignment_manifest_and_audit
  -> final_data_attestation
  -> combined_execution_lock
  -> label_free_prediction_shards[finalist, source]
  -> prediction_manifest[finalist]
  -> locked_metric_join
  -> final_aggregate
```

The selection task preserves the parent's frozen class-stratified rule (100k
rows in pilot); assignment uses the frozen Shell-Exact matcher and publishes
the complete selected-role shard/manifest/audit artifacts required by D100.
`HCWDL_SHARED_FINAL_DATA_ATTESTATION/v1` binds realized selection and assignment
hashes to the population, claim, precommitted rules, exact row/class/source
counts, and final task registry. The subsequent
`HCWDL_REPRESENTATION_EXECUTION_LOCK/v1` binds that attestation, the finalist
lock, shared claim, and exact prediction/metric registry before any prediction
begins; it does not retroactively authorize selection or assignment.

The coordinator atomically creates
`HCWDL_SHARED_FINAL_EXECUTION_CLAIM/v1` after the finalist lock/task registry
exist and before selection. The immutable claim contains population,
reservation, finalist-lock, task-registry, submission-ledger, source commit,
and deterministic `claim_owner_id = SHA256(canonical_json({campaign_spec,
task_registry, finalist_lock}))`. Exclusive creation has exact idempotence: if
the file already exists with the same owner and every parent hash, the same
coordinator/recovery validates and reuses it; any difference fails before a
branch read. Fan-out workers never create claims. They present
`HCWDL_SHARED_FINAL_ROLE_CAPABILITY/v1` binding the claim, registry, exact task
row, purpose, and allowed branches to the common final-role reader. Prediction
and metric rows additionally bind the combined execution-lock hash.

Preemption or task failure does not create another exposure. An immutable
`HCWDL_SHARED_FINAL_RECOVERY_PLAN/v1` binds the existing claim and registry,
audits every atomic output, and resubmits only rows whose final output is absent
(an orphan temporary path is first audited and removed under the ordinary
atomic-publication contract) under the same owner. A published artifact that
fails byte/content/parent validation is corruption and fails closed: recovery
never overwrites it or silently creates a new unregistered attempt. Completed
shards/manifests are reused only after hash validation. Recovery cannot add
finalists, source rows, branch families, output paths, attempts, or metric
computations. A two-process claim race, coordinator crash immediately
before/after publication, partial prediction wave, same-owner recovery, and a
corrupt-published-output rejection are mandatory tests.

All final data entry points use one versioned capability gate in a shared
module; updating only `hcwdl_final.py` is insufficient. The old root-local
parent evaluator and old `require_role_access(..., {finalist, execution})`
capability are rejected after reservation. Both parent and representation
workers must validate the population-scoped claim and exact task row before
opening any final branch.

### 25.4 Label access, prediction artifacts, and metric join

The parent final-row selection is label-dependent. HCWDL-RKD does not hide
that access: `final_row_selection_and_label_escrow` is the one registered ROOT
label read, after the shared claim. It projects only canonical identity,
baseline-selection fields, and label branches; selects the frozen natural
class-stratified population; publishes a public identity/count manifest and a
separately capability-protected `HCWDL_SHARED_FINAL_LABEL_ESCROW/v1` mapping
selected identity digests to uint8 labels. It loads no particle branches and
computes no model output. The escrow payload/sidecar/commit are published by the
Section 21 shared directory-envelope protocol; an NPZ without its committed
envelope is invisible, not a recoverable published artifact.

Assignment tasks read only the required HLT/offline particle and identity
branches. Prediction tasks use a new reusable label-free final streamer that
projects identity plus the exact model-input branches only. It exposes three
explicit, audited paths: `iterate_final_hlt_inputs`,
`iterate_final_shell_exact_inputs`, and `iterate_final_native_offline_inputs`.
The HLT/native-offline paths do not call current
`dataset.py::iterate_model_batches`, because that function unconditionally
adds `LABEL_BRANCHES` and constructs labels. The Shell-Exact path does not call
current `pmard_stream.py::iterate_pmard_batches`, because that training stream
also carries labels; it uses a label-free repair streamer over the already
claimed identities and authenticated final assignment manifest. Each new path
has a frozen projected-branch allow-list, resolves only identity digests in the
claimed row-selection manifest, and emits `(identity_digest, model_inputs)`
without a label field. A branch-access audit records the ordered ROOT files,
trees, projected branch names, entries, and capability hash for every shard.
Instrumented tests make any label-branch request in these paths raise before
I/O. The existing `iterate_model_batches`, `iterate_pmard_batches`, and
`evaluate_model` label-joining paths are forbidden for final prediction.
Deployable finalists load only extracted HLT backbones and the common HLT
view. D100/TOFF diagnostics use only their predeclared input domains under the
same claim. No final task builds or loads a representation target bank.

Each `HCWDL_REPRESENTATION_PREDICTION_SHARD/v1` contains only sorted identity
digests and finite FP32 `[rows,15]` logits--no labels or probabilities. Its
manifest proves exact disjoint selected-row coverage. The single
`locked_metric_join` validates all manifests, opens the label escrow (not ROOT
particle/label branches), joins every identity exactly once, computes the
frozen softmax/metric set, and publishes one evaluation per registered
finalist. The prediction execution/metric signatures and resources are frozen
in Section 27.

### 25.5 If the parent claim has already executed

When disposition is `validation_only_parent_claim_consumed`, the
representation ascents are validation-only follow-up experiments on this
split. They cannot share or reopen the old confirmatory claim. A confirmatory
representation result requires a genuinely untouched, identity-disjoint
holdout with a new population/disjointness artifact and reservation. Code
reports this status rather than issuing a second claim over exposed rows.

## 26. Failure semantics

Finite poor AUC, CE, rejection, calibration, representation similarity, or
negative rung gain is a completed scientific result. It never cancels a
descendant or suppresses a row.

The following fail closed:

- stale or mismatched parent imports;
- wrong teacher/track/domain or target-bank lineage;
- missing/duplicate identities or incomplete target coverage;
- invalid RFF resources, named taps, shapes, masks, or TOFF family semantics;
- target construction from validation or final test;
- nonfinite required tensors, losses, gradients, or metrics;
- nonfinite calibration inputs/gradients, a stale phase artifact, or a
  component-status/scale pair inconsistent with Section 13.3;
- warm loading of training-only heads or cross-strategy backbones;
- deployable export containing a representation target input or offline
  parameter path;
- final-test access without the new combined locks;
- a resume state with different representation recipe, targets, projections,
  calibration, or stochastic lineage.

An execution failure is recovered with the same immutable semantics. It is not
fixed by editing a target bank, coefficient, layer, teacher, or graph in place.

## 27. Resource, storage, and Slurm plan

Introduce a measured `gpu_representation` resource class separate from
ordinary logit-only dual-teacher training. It includes:

- HLT train/validation view caches;
- one loaded target bank;
- ParT/optimizer/resume state;
- representation activations and projections;
- RFF and top-32 relation workspace;
- construction transients and operating-system headroom.

Target-builder jobs have a separate measured GPU class because they perform
one frozen teacher forward and source-sharded publication but no optimizer
steps. They request exactly one GH200 and set the Section 16.3 deterministic
environment before Python imports Torch. Cleanup/merge/lock/report tasks are
CPU jobs.

Final inference has a separate measured `gpu_final_prediction` class and
`HCWDL_REPRESENTATION_FINAL_PREDICTION_SPEC/v1`. It freezes one GH200, strict
checkpoint/architecture/input-domain hashes, model evaluation mode, FP32
parameters/inputs/forward, autocast disabled, TF32 disabled, deterministic
algorithms/backend flags, canonical 256-row batches that never cross a source
partition, final-short-batch behavior, feature/identity streamer hash, and
FP32 C-order logit output. Softmax is not computed or stored by GPU workers.
The CPU metric join freezes the source hashes and versions of
`evaluation.py::softmax`/`classification_metrics`, converts FP32 logits to
FP64, uses SciPy `logsumexp`, and binds label-escrow/order hashes.

Every allocation is shape-estimated before large arrays are created and must
stay below 75% of the Slurm memory request after accounting for all concurrent
arrays. Pilot may request more than 192 GiB when the real miniature justifies
it; production is expected to use the parent 384-GiB scale or a separately
measured value. Scientific batching remains effective/microbatch 256/256.

The storage estimator covers target generations, two retained resume states,
selected/final checkpoints, final selection/label escrow, final assignment
shards, and every prediction shard simultaneously required by the task DAG.
Prediction payload bytes are exactly
`N_selected * N_finalists * (32 + 15 * 4)` before container metadata and
compression: 32 identity-digest bytes plus 15 FP32 logits per row. Probabilities
are not duplicated. The estimate also reserves one interrupted-shard
generation and 25% filesystem headroom.

Arrays are uncapped by default. A measured filesystem or site limit may add a
recorded operational cap without changing targets or row selection. Training
jobs are checkpointable, request `--signal=B:USR1@120`, and run Python as the
batch-shell PID through `exec python -s`.

Walltimes come from a genuine Tigris miniature that measures separately:

- ordinary and TOFF target construction;
- RSET cold and warm two-update/full-loss probes;
- RREL cold and warm two-update/full-loss probes;
- target loading and cleanup;
- validation, checkpoint publication, and USR1 resume;
- label-dependent final selection on a validation-role proxy, Shell-Exact
  assignment, label-free prediction for every input-domain family, manifest
  merge, label-escrow metric join, and partial-wave recovery.

No full pilot submission is authorized until those measurements, a storage
estimate, complete dry run, clean pushed source, and explicit user approval
exist.

## 28. Planned implementation architecture and file map

### 28.1 Separation from the parent implementation

HCWDL-RKD is an additive campaign package. The implementation must not edit
the meaning of `HCWDL_GRAPH/v1`, `HCWDL_RECIPE/v3`, any existing node ID, or
any completed parent artifact. New code may reuse general-purpose readers,
metrics, checkpoint utilities, and HLT view caches, but every representation
artifact is namespaced under the contracts in Section 21.

In particular, the legacy PMARD representation arms in:

```text
src/hlt_classification/models/scouting_particle_transformer.py
src/hlt_classification/scouting/engine.py
src/hlt_classification/scouting/training.py
```

are reference implementations only. Their current late-layer surfaces,
same-mask assumptions, live privileged forwards, and `R1`--`R5` loss meanings
do **not** implement this plan. HCWDL-RKD must never obtain its semantics by
passing a new string through the old PMARD arm switch.

### 28.2 Model-surface changes

Extend
`src/hlt_classification/models/scouting_particle_transformer.py` without
changing either public classifier forward:

1. Add a frozen dataclass for the ordinary HLT/D-teacher surfaces:

   ```text
   HCWDLScoutingSurfaces
     logits:              [B, 15]
     particle_block_2:    [B, N, 128]
     jet_penultimate:     [B, 128]
     particle_mask:       [B, N] bool
     vectors:             [B, 4, N]
     visible_indices:     [B, N] integer
     family_codes:        [B, N] int8
   ```

2. Add a distinct method named `forward_hcwdl_surfaces(...)`. It accepts the
   training-only canonical token IDs and pre-transform family codes, propagates
   them through the same trimmer operation as auxiliary channels, strips them
   before embedding, performs
   the installed-Weaver forward once, captures the output of particle block
   index one, captures the exact 128-vector that enters the final 15-class
   linear map, and returns logits from that same execution. Tests prove the
   auxiliary channel consumes no model RNG and has no logit/gradient path.
3. Add `HCWDLNativeOfflineSurfaces` and
   `NativeOfflineParticleTransformer.forward_hcwdl_surfaces(...)` with
   separate charged and neutral block-two tensors/masks/vectors plus the
   128-dimensional classifier hidden state and logits.
4. Do not change `forward_representations(...)` or
   `RepresentationScoutingParticleTransformer`; old PMARD reports must remain
   reproducible.
5. Treat the installed-Weaver callable/signature, block count, hidden width,
   trimmer behavior, aggregator path, and classifier path as an authenticated
   runtime signature. A missing or changed surface fails before training.

The representation method may share a private manual-forward helper with the
existing representation method only after forward/logit/gradient parity is
proven for both call paths. Public `forward(...)` remains the only deployable
interface.

### 28.3 New reusable modules

Add the following focused modules under
`src/hlt_classification/scouting/`:

| Module | Owned responsibility |
|---|---|
| `hcwdl_representation_contracts.py` | contract names, canonical payload validation, artifact parent checks, role access |
| `hcwdl_representation_graph.py` | 24-node registry, teacher mapping, strategy/track/init invariants, DAG validation |
| `hcwdl_representation_recipe.py` | immutable overlay recipe, kernel resources, ramps, coefficients, inherited-parent validation |
| `hcwdl_representation_kernels.py` | fixed spectral-resource generation, exact finite-kernel/analytic-gradient checks, ideal-RBF diagnostics, weighted token/relation sketches |
| `hcwdl_representation_losses.py` | jet, set, relation, orthogonality, family-aware reductions, gradient calibration |
| `hcwdl_representation_targets.py` | target shard construction, manifests, exact joins, RAM materialization, cleanup |
| `hcwdl_representation_training.py` | model/head initialization, one-time predecessor logits, 60-pass engine adapter, resume, extraction |
| `hcwdl_representation_reporting.py` | per-node reports, paired deltas, screen/confirmation/final aggregate |
| `hcwdl_paired_bootstrap.py` | exact 15-class paired resampling, metric/delta vectors, interval artifact |
| `hcwdl_shared_final.py` | neutral population registrar/exposure ledger, reservation, claim, task capabilities, idempotent recovery; mandatory for parent and representation evaluators |
| `hcwdl_final_stream.py` | label-only selection/escrow, three label-free inference loaders, projected-branch audits, locked label join |
| `hcwdl_representation_final.py` | complete parent-union finalist registry, final assignment/data attestation, prediction manifests, combined aggregate |
| `hcwdl_representation_locks.py` | parent import, screen, finalist, representation execution and cleanup validation; delegates shared final access to `hcwdl_shared_final.py` |
| `hcwdl_representation_workflow.py` | task dispatcher and prerequisite/output validation |
| `hcwdl_representation_campaign.py` | immutable campaign spec, command plan, Slurm resources and dependency graph |
| `hcwdl_representation_recovery.py` | monitor classification, exact task resumption, target-bank reconstruction rules |

The modules may be consolidated during implementation only when ownership and
tests remain equally clear. Scientific meanings, contract names, and graph
entries may not be consolidated into generic unversioned dictionaries.

### 28.4 Existing modules that may be reused or narrowly extended

The implementation should reuse:

- `hcwdl_ladder.py` for the authenticated parent nodes/domains;
- `hcwdl_recipe.py` for exact parent weights, temperatures, LR, batching, and
  schedule;
- `dataset.py::iterate_model_batches` for train/validation HLT/native-offline
  decoding only; final prediction must use the separate label-free functions in
  `hcwdl_final_stream.py` and may reuse lower-level feature decoders only after
  their projected-branch allow-list is explicit;
- `pmard_stream.py::iterate_pmard_batches` for train/validation repaired-D
  stream construction, with the new training-only token-ID/family auxiliary
  channel propagated through the same trim/permutation; the D100 final oracle
  must use the separate label-free Shell-Exact streamer;
- `view_cache.py::EphemeralPmardViewCache` for the production process-local RAM
  replay pattern, extended or replaced under a new contract because its current
  payload does not preserve the required auxiliary token metadata;
- `highcov_cache.py::DenseAssignmentStore` for operational assignment joins;
- `hcwdl_assignment.py` only for construction/finalization and lineage-audit
  patterns--it does not own the runtime assignment store;
- `hcwdl_views.py` only as the parent RAM-bank semantic prototype; current
  production execution is assembled in `hcwdl_runner.py` from the concrete
  loaders/streams/cache above, so importing this prototype alone is not a
  complete data path;
- `hcwdl_training.py` checkpoint selection and warm checkpoint
  authentication;
- `targets.py` identity and finite-audit patterns, not its limited logit-only
  shape contract;
- `engine.py` metrics, validation, LR, checkpoint, and preemption helpers;
- `loaders.py` strict checkpoint/model-factory validation patterns;
- `hcwdl_runner.py` as a map of the current concrete HCWDL assembly, not as a
  representation-training entry point;
- `hcwdl_final.py` only after versioning it to delegate **all** final-role access
  to `hcwdl_shared_final.py` and its label-free stream/join surfaces; its current
  root-local `execution_claim.json` and label-joining evaluator are explicitly
  incompatible;
- `hcwdl_authorization.py` and `hcwdl_resources.py` authorization/resource
  patterns.

`training.py::pmard_loss` is **not** reusable for the locked base objective
until the Section 3.3 parent-loss attestation issue is resolved: today it
multiplies both KL rows by class weights, while the authoritative recipe in
this plan requires unweighted KL row means. The implementation must add a
tested locked-loss function or version the shared helper and rerun any parent
artifacts whose declared semantics disagree. Reusing the current helper and
merely relabeling its report is forbidden.

If a reusable parent helper is extended, the old call signature and default
behavior must remain byte/gradient equivalent. A contract is bumped if an old
artifact consumer would otherwise accept a new meaning.

### 28.5 Training engine boundary

Implement a dedicated `train_hcwdl_representation_node(...)` entry point. It
may factor common optimizer/validation/checkpoint mechanics out of
`train_pmard`, but it must not enter the old PMARD representation-arm branch.
Its optimizer step is exactly:

1. load one HLT-only student batch and canonical identities;
2. run `forward_hcwdl_surfaces(...)` once under the parent BF16 policy;
3. convert logits and captured surfaces to FP32 at the loss boundary;
4. join predecessor logits when the rung has a predecessor;
5. join privileged logits and compact representation targets;
6. compute the locked base loss;
7. compute the active scheduled representation loss;
8. backpropagate the sum once;
9. step the shared optimizer/scheduler once;
10. update interval-mean reporting accumulators without per-update host
    synchronization.

The student never receives teacher tokens. It receives only compact detached
teacher summaries from the bank. The only token-sized live representation is
its own current HLT activation.

At M2--M6, the same-branch HLT predecessor logits are built exactly once at
student-job startup from the authenticated predecessor checkpoint and the
HLT train cache. They are held in process RAM, logically hashed, and the
predecessor model is released before optimization. They are not recomputed
each pass and are not published as a durable campaign dataset. At M1, the D0
bank supplies the sole teacher logits and representation targets.

### 28.6 Head ownership and deployable extraction

Store the student backbone/classifier and training-only projections in
separate state namespaces:

```text
deployable_model.*
representation_heads.jet.*
representation_heads.token.*
representation_heads.token_charged.*
representation_heads.token_neutral.*
```

Only heads required by a node exist. `RSET` cannot instantiate a relation
head. Ordinary M1--M5 nodes cannot instantiate charged/neutral-specific heads.
The complete rolling and selected training checkpoints contain heads,
optimizer slots, calibration, RNG, interval accumulators, and target hashes
for exact resume.

`HCWDL_REPRESENTATION_DEPLOYABLE_EXTRACTION/v1` publishes a separate HLT-only
checkpoint containing only the canonical `ScoutingParticleTransformer`
state. Strict load into a fresh public model, HLT-only inference parity, key
allow-listing, and absence of every representation/offline key are mandatory.
Warm children load this extracted deployable checkpoint, then initialize new
identity representation heads. They never inherit the parent's old head.

## 29. Target-bank physical design

### 29.1 Shard contents

One physical shard corresponds to one authenticated source-file partition and
contains sorted arrays for the selected train rows in that source. The packed
payload contains only the columns appropriate to its bank:

```text
source_file_id                 uint32 [J]
source_entry                   uint64 [J]
identity_digest                uint8  [J, 32]
label                         uint8  [J]       audit/join only
logits                        float32 [J, 15]
jet_penultimate               float32 [J, 128]
token_kernel_mean             float32 [J, 1024]      ordinary D
token_kernel_mean_charged     float32 [J, 1024]      TOFF
token_kernel_mean_neutral     float32 [J, 1024]      TOFF
token_family_eligibility      uint8  [J, F_bank]
relation_kernel_mean          float32 [J, 3, 256]    ordinary D
relation_kernel_mean_charged  float32 [J, 3, 256]    TOFF
relation_kernel_mean_neutral  float32 [J, 3, 256]    TOFF
relation_eligibility          uint8  [J, F_bank, 3]
token_count                   uint16 [J, F_bank]
token_scalar_pt_sum           float32 [J, F_bank]
relation_pair_count           uint16 [J, F_bank, 3]
relation_effective_sample     float32 [J, F_bank, 3]
family_reason_counts          uint16 [J, R_bank]
```

The axes are not symbolic runtime choices. For every ordinary D bank,
`F_bank=1` with ordered family axis `("all",)` and `R_bank=1` with reason axis
`("ordinary_all",)`. For TOFF, `F_bank=2` with ordered axis
`("charged","neutral")` and `R_bank=6` with this exact reason-code order:

```text
known_pid_charge_agree_charged
known_pid_charge_agree_neutral
charge_only_charged
charge_only_neutral
unclassified_contradiction
unclassified_malformed
```

`token_count`, `token_scalar_pt_sum`, pair count, and relation ESS describe the
post-trimmer **teacher target** clouds. TOFF `family_reason_counts` describe the
companion HLT bookkeeping tokens after the target-forward spec's exact trim and
Section 9.4 classifier; ordinary `ordinary_all` counts that bank's visible
target tokens. Live stochastic student counts remain training diagnostics, not
cached target metadata. Invalid/nonfinite charge or PID inputs fail before
counting. The first four TOFF reasons sum to classified charged/neutral HLT
tokens; the last two sum to HLT tokens excluded from M6 token/relation KD. The
manifest records these ordered vocabularies verbatim and validates the declared
teacher and companion-HLT conservation equations separately.

Fixed metadata bytes per row are exactly
`4 + 8 + 32 + 1 + F_bank + 3*F_bank + 2*F_bank + 4*F_bank +
6*F_bank + 12*F_bank + 2*R_bank = 45 + 28*F_bank + 2*R_bank`.
They are therefore `75` bytes for an ordinary bank and `113` bytes for TOFF,
before container overhead or compression. Together with the core floats this
gives the exact `7,815`/`15,021` logical bytes per row used by Section 16.5.

`label` cannot participate in a representation target or model forward. It is
stored solely to audit class counts and implement the predeclared within-class
shuffle control. The canonical identity remains label-free.

The concrete container is one deterministically keyed compressed NumPy archive
per source partition plus an immutable JSON sidecar. Array names are sorted;
dtype, shape, C-order logical hash, compressed byte hash, row count, first/last
identity, and parent hashes are recorded. Compression may change physical
bytes across a reconstructed artifact, so recovery requires identical logical
array hashes and publishes new physical-byte hashes under explicit recovery
lineage. Consumers always authenticate the actual bytes named by their
manifest.

Every manifest has a physical `generation_id`, an authorized consumer set,
and a stable `logical_target_sha256` over ordered array logical hashes and
scientific parents. Reconstruction changes the generation and physical byte
hashes but is valid only when the stable logical target hash is unchanged.

### 29.2 Exact bank construction algorithm

For each logical bank:

1. validate the parent import, overlay recipe, kernel resources, teacher
   selection, assignment/repair lineage, train row selection, and canonical
   target-forward specification and any prior execution attestation required
   for a rebuild; no staging directory or build-intent contract may exist yet;
2. calculate exact output shapes, logical metadata bytes, peak bytes, target
   paths, and consumer set before loading the teacher; reject an impossible
   storage/resource plan during this preflight;
3. create only the owner-scoped staging directory and atomically publish/fsync
   `HCWDL_REPRESENTATION_TARGET_BUILD_INTENT/v1` inside it, binding the logical
   bank, generation/consumer registry, forward spec, exact sorted source
   partitions/output names, expected rows/shapes/bytes, and deterministic
   `build_owner_id`;
4. load the selected teacher checkpoint strictly and set evaluation mode;
5. open only projected branches needed for that teacher view;
6. stream each selected train identity exactly once in canonical source/entry
   order using the pre-build forward spec's per-source 256-row batch
   partition;
7. perform one deterministic FP32 teacher surface forward per canonical batch
   with every backend/precision flag in the forward spec enforced;
8. calculate token and relation sketches immediately, then discard token
   activations and particle inputs;
9. publish each audited shard atomically **inside staging**. A staged shard is
   resumable work product, not a reusable target-generation artifact;
10. after exact total-row, class-count, identity-set, and shard-parent
    conservation succeeds, publish the post-build execution attestation,
    generation record, and manifest inside staging; fsync every file and the
    directory tree;
11. atomically rename the complete owner staging directory to
    `generations/<generation_id>`, fsync its parent, and treat that directory
    appearance as the sole generation commit point. If the destination exists,
    validate it byte-for-byte instead of overwriting it;
12. release teacher, view buffers, and construction scratch.

Before a recovery publication, step 6 first replays the prior execution
attestation's sentinel batches under the unchanged forward spec and must
reproduce their prior logical hashes. The finished generation
must then reproduce the complete prior `logical_target_sha256`; sentinel
agreement alone is necessary but not sufficient.

An interrupted staging tree has exact-owner idempotence. Recovery validates the
build intent, owner, forward signature, and every completed staged shard,
continues only missing partitions, and quarantines/recomputes a corrupt staged
file because it was never committed as an artifact. A different owner or
payload cannot adopt it. A crash before the directory rename exposes no
generation; a crash after it exposes a complete generation. Atomic nonempty
directory rename/fsync behavior is a required real-Tigris miniature gate. A
complete manifest cannot reference a partial shard. Reuse validates both the
immutable manifest and every current shard byte; a manifest is never trusted
because its directory exists.

### 29.3 Target coverage invariants

For every logical bank:

```text
manifest train rows == parent selected train rows
union(shard identities) == parent train identity set
intersection(any two shard identities) == empty
sum(shard class counts) == parent selected train class counts
every identity has exactly one finite logit and jet target
every eligibility bit agrees with the presence of its target sketch
no validation or final-test identity appears
```

Token/relation ineligibility is legal only under Sections 9 and 10 and is
represented by an explicit bit plus zero-filled storage that cannot enter a
loss denominator. It is never represented by NaN, a missing row, or an
invented empty target.

### 29.4 Just-in-time lifecycle and maximum storage

Banks are built rung-by-rung:

```text
D0c/D0w   -> four M1 consumers  -> cleanup
D25c/D25w -> four M2 consumers  -> cleanup
D50c/D50w -> four M3 consumers  -> cleanup
D75c/D75w -> four M4 consumers  -> cleanup
D100      -> eight M5 primary/control consumers -> cleanup
TOFF      -> four M6 consumers  -> cleanup
```

Within each of the first four lines, warm-bank readiness depends on cold-bank
cleanup **completion**; the next line's cold bank depends on warm-bank cleanup
completion. Shared D100/TOFF each have one completion. These are actual DAG
edges, not documentation shorthand or optional optimization. The screen `TOFF`
cleanup precedes construction of the separate 20-consumer confirmation
generation. Dry-run graph tests compute the maximum live generation set and
prove it is at most one committed bank plus explicitly reserved staging/recovery
headroom.

For track-specific banks each individual bank has two consumers. Cleanup waits
for an authenticated complete training report from every registered consumer
and for those reports/resume checkpoints to bind the exact generation and
target logical hash. Failed, cancelled, timed-out, preempted, or merely
terminally recorded jobs never satisfy a cleanup dependency.
Cleanup publishes hashes and byte totals of removed target shards. Reports,
manifests, logical hashes, and cleanup records remain durable; compact target
payloads do not.

Recovery is a generation-aware state machine with these exhaustive states:

```text
STAGING_BUILD_IN_PROGRESS       -> same-owner validate/continue, then commit
COMPLETE_SHARDS_NO_AUTH        -> validate and resume registered consumers
CLEANUP_AUTHORIZED_IN_PROGRESS -> idempotently finish exact deletion, attest
CLEANUP_COMPLETED              -> rebuild new generation if consumers remain
INCOMPLETE_NO_AUTHORIZATION    -> corruption; fail closed
ABANDONED_AUTHORIZED           -> finish only its separately authorized purge
```

State precedence is content-based: a valid committed generation ignores any
orphan staging tree; otherwise a valid same-owner build intent defines staged
construction and no committed-generation corruption exists. For a committed
generation, a valid cleanup completion wins; otherwise a valid authorization
plus manifest defines in-progress cleanup; otherwise every committed shard
must validate or the generation is corrupt. The recovery plan names incomplete
or newly authorized consumer rows explicitly and creates a new consumer
registry. It cannot transitively retry already completed descendants, cannot
write into the old generation directory, and cannot infer state from Slurm job
names. The recovery generation is cleaned only after every consumer in its
smaller registry publishes an authenticated complete report and its own cleanup
authorization exists.

The campaign-spec builder computes pilot and production peak durable bytes
from exact selected rows and schemas. It refuses a configured target-storage
cap smaller than the next single bank plus its declared staging/recovery and
filesystem headroom. It never keeps two committed banks, much less all ten, at
once.

## 30. Command-line and worker surface

### 30.1 Thin Python entry points

Add thin scripts with no scientific logic:

```text
scripts/build_hcwdl_representation_parent_import.py
scripts/build_hcwdl_representation_recipe.py
scripts/audit_hcwdl_representation_parent_final_state.py
scripts/build_hcwdl_representation_final_disposition.py
scripts/build_hcwdl_representation_fixed_size_inventory.py
scripts/build_hcwdl_representation_storage_estimate.py
scripts/build_hcwdl_representation_scheduler_evidence.py
scripts/build_hcwdl_representation_miniature_evidence.py
scripts/build_hcwdl_representation_resource_profile.py
scripts/build_hcwdl_representation_usr1_exact_resume_proof.py
scripts/build_hcwdl_representation_validation_proxy_proof.py
scripts/build_hcwdl_representation_production_worker_smoke_proof.py
scripts/build_hcwdl_representation_tigris_action_proof.py
scripts/build_hcwdl_representation_tigris_evidence_bundle.py
scripts/build_hcwdl_representation_tigris_acceptance.py
scripts/measure_hcwdl_representation_worker_runtime.py
scripts/build_hcwdl_representation_targets.py
scripts/train_hcwdl_representation_node.py
scripts/select_hcwdl_representation_checkpoint.py
scripts/extract_hcwdl_representation_model.py
scripts/authorize_hcwdl_representation_cleanup.py
scripts/complete_hcwdl_representation_cleanup.py
scripts/aggregate_hcwdl_representation_screen.py
scripts/build_hcwdl_representation_confirmation_registry.py
scripts/register_hcwdl_shared_final_population.py
scripts/claim_hcwdl_shared_final_execution.py
scripts/build_hcwdl_shared_final_selection.py
scripts/build_hcwdl_shared_final_assignment_shard.py
scripts/finalize_hcwdl_shared_final_assignments.py
scripts/build_hcwdl_shared_final_data_attestation.py
scripts/build_hcwdl_shared_final_legacy_cancellation.py
scripts/build_hcwdl_representation_execution_lock.py
scripts/predict_hcwdl_shared_final_shard.py
scripts/finalize_hcwdl_shared_final_predictions.py
scripts/join_hcwdl_shared_final_metrics.py
scripts/recover_hcwdl_shared_final.py
scripts/create_hcwdl_representation_campaign.py
scripts/build_hcwdl_representation_runtime_binding.py
scripts/build_hcwdl_representation_runtime_prerequisites.py
scripts/build_hcwdl_representation_runtime_rows.py
scripts/dry_run_hcwdl_representation_campaign.py
scripts/audit_hcwdl_representation_executable_candidate.py
scripts/build_hcwdl_representation_submission_authorization.py
scripts/build_hcwdl_representation_acceptance_bootstrap.py
scripts/run_hcwdl_representation_acceptance_bootstrap.py
scripts/submit_hcwdl_representation_campaign.py
scripts/run_hcwdl_representation_task.py
scripts/monitor_hcwdl_representation_campaign.py
scripts/resume_hcwdl_representation_campaign.py
scripts/cancel_hcwdl_representation_campaign.py
scripts/run_hcwdl_representation_local_smoke.py
```

Every scientific argument comes from the immutable campaign spec and recipe.
CLIs may select a registered task/array index, device, output path, and dry-run
mode; they may not override coefficients, layers, bandwidths, ramps,
temperatures, rows, seeds, or teachers.

### 30.2 Slurm worker

Add two thin workers:

```text
sbatch/run_hcwdl_representation_task.sh
sbatch/run_hcwdl_representation_deterministic_task.sh
```

The first nonfinal evidence prefix additionally has two isolated bootstrap
workers:

```text
sbatch/run_hcwdl_representation_acceptance_bootstrap.sh
sbatch/run_hcwdl_representation_acceptance_bootstrap_deterministic.sh
```

They require the exact phrase-authorized bootstrap artifact and can dispatch
only its frozen dependency-closed scalar prefix. The bootstrap contract binds
their bytes and the clean source commit. They cannot run arrays, ladder
training, reservations, shared-final/final-role work, or campaign submission,
and the bootstrap builder never mutates the scheduler. The present prefix ends
at zero-coefficient acceptance; its full-loss smoke performs no optimizer
step, so separate two-update/USR1 authority is still required before the
complete Tigris acceptance bundle can exist. The bootstrap planning spec lives
at the canonical `campaign_spec.json` of a dedicated smoke root; that root is
never overwritten or promoted into an executable pilot root.

The action-specific proof contracts make these evidence boundaries explicit,
but do not authorize or execute the missing actions. Until separately
authorized actions produce and validate every required proof,
`HCWDL_REPRESENTATION_TIGRIS_ACCEPTANCE/v1` remains nonauthorizing and fails
closed. Scheduler completion plus a caller-authored success boolean is not
evidence of exact USR1 resume, production-worker semantics, or final-role
isolation. Each of the seven frozen Tigris check rows binds all three of its
scheduler evidence, miniature evidence, and action proof; no two-record row is
accepted.

Authorizing scheduler evidence is derived from immutable raw `sacct`
allocation and step records captured by a separate Tigris Slurm collector,
not from CLI state/RSS/elapsed arguments. Its pre-submission comment binds the
exact source, recipe, task, resource request, worker role, and worker bytes.
The corresponding miniature record binds the exact immutable semantic result
into that execution lineage. Local scheduler/action fixtures remain explicitly
nonauthorizing, and the final acceptance validator requires the raw-capture
provenance class for all resource measurements and all seven action proofs.

Both validate `PROJECT_DIR`, activate the declared Conda environment, set
`PYTHONNOUSERSITE=1`, and prepend `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`.
The ordinary worker handles training/report/lock tasks and deliberately leaves
the parent training CUDA environment unchanged. The deterministic worker is
registered only for target-generation and final-prediction task kinds, sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python/CUDA initialization, and refuses
any other task. The Python dispatcher validates the corresponding immutable
task kind and remaining precision/backend flags. Both end with:

```bash
exec python -s "${PROJECT_DIR}/scripts/run_hcwdl_representation_task.py" ...
```

Neither contains relative repository imports or scientific defaults. Shell
regressions verify the final `exec`, environment isolation, absolute helper
paths, quoted arguments, deterministic-task allow list, and absence of the
CUBLAS override in paired training jobs.

### 30.3 Dry run and submission rules

The dry run publishes a versioned symbolic command plan without mutation. Each
row contains exact invariant `sbatch` argv, resources, arrays, source commit,
worktree-clean proof, campaign root/spec hash, a stable `task_key`, and
dependencies as ordered upstream **task keys**, represented in argv by typed
tokens such as `${afterok:target_D0c}`. Scheduler job IDs cannot appear in a
pre-submission dry run and are never guessed from job names.

Submission topologically consumes that template. After each successful
`sbatch`, it records the returned exact job/array ID, substitutes only the
declared dependency tokens of downstream rows, and publishes the fully
materialized argv and key-to-ID map in
`HCWDL_REPRESENTATION_SUBMISSION_LEDGER/v1`. Removing, adding, reordering, or
changing any nondependency token fails. A deterministic validator rematerializes
each argv from the immutable template plus earlier ledger IDs and requires
byte-for-byte equality. Thus the reviewed plan is exact while acknowledging
that Slurm assigns dependencies only during submission.

Array indices map through immutable registry rows. Operational concurrency is
uncapped unless a measured resource profile explicitly sets a limit. Changing
only a concurrency limit does not change scientific identity, but it is
recorded in the command plan and ledger.

Every recovery submission publishes
`HCWDL_REPRESENTATION_RECOVERY_SUBMISSION_LEDGER/v1`, binding the recovery plan,
campaign spec, original ledger, complete ordered chain of prior recovery
ledgers, symbolic recovery templates, materialized argv, and newly returned
exact job IDs. Monitoring and cancellation first validate the original ledger
plus this hash chain and operate on the authenticated union of exact live IDs.
They never infer replacement jobs from names or omit them because they were not
in the original submission. Broad `scancel --name` operations are not part of
the interface.

Monitoring is append-only. Invocation sequence `q` publishes immutable
`monitoring/reports/<q>_<content_hash>.json` under
`HCWDL_REPRESENTATION_MONITOR_REPORT/v1`, binding the complete authenticated
submission-ledger chain and the preceding monitor-report hash (or `null` at
`q=0`). Repeated monitoring never overwrites `monitor_report.json`. An optional
operational `monitoring/HEAD` is an atomically replaced, fsynced pointer
containing only sequence, relative path, and content hash; it is not a
scientific artifact, and readers validate the referenced immutable report. A
missing, stale, or corrupt HEAD can be reconstructed from the valid report
chain without changing any report.

## 31. Exact artifact layout

The representation campaign uses a new root; it never writes into the parent
campaign root:

```text
<representation_root>/
  campaign_spec.json                  # campaign_spec_parent = "."
  planning/
    campaign_spec.json                # campaign_spec_parent = "planning"
  command_plan.json
  runtime/
    runtime_binding.json
    runtime_prerequisites.json
    dry_run_audit.json
    measurements/
      ${resource_class}.json
  submission_ledger.json
  submission_events/
    <sequence>_<content_hash>.json
  recovery_submission_ledgers/
    <sequence>_<content_hash>.json
  graph/
    ascent_graph.json
  import/
    parent_import.json
    parent_loss_attestation.json
    architecture_attestation.json
    parent_final_state.json
    final_disposition.json
  architecture/
    tap.json
    surface_parity.json
  recipes/
    representation_recipe.json
    kernel_resources/
      staging/<envelope_id>/<envelope_owner_id>/...
      committed/<envelope_id>/
        kernel_resources.npz
        sidecar.json
        commit.json
  resources/
    measured_profile.json
    storage_estimate.json
    fixed_size_inventory.json
  acceptance/
    numerical.json
    smoke_probe.json
    local_smoke_report.json
    cache_miniature.json
    cache_miniature_D100.json
    cache_miniature_TOFF.json
    tigris_acceptance.json
    tigris_evidence_bundle.json
    executable_candidate_audit.json
    evidence/
      scheduler/<evidence_id>.json
      miniature/<evidence_id>.json
    proofs/
      <evidence_kind>.json
      usr1_exact_resume_result.json
      final_role_validation_proxy_result.json
      production_worker_smoke_result.json
    bootstrap/
      spec.json
  recovery/
    targets/<recovery_id>.json
    training/<recovery_id>.json
  targets/
    <logical_bank>/
      logical_bank.json
      staging/
        <generation_id>/<build_owner_id>/...
      generations/
        <generation_id>/
          build_intent.json
          consumer_registry.json
          target_forward_spec.json
          target_execution_attestation.json
          generation.json
          manifest.json
          shards/
            <source_partition>.npz
            <source_partition>.json
  training/
    <strategy>/<node_id>/
      <execution_id>/
        calibration/
          selection.json
          diagnostic_batch.json
          jet_set.json
          relation.json               # RREL only
          manifest.json
        resume/
          state_<sequence>.pt
          state_<sequence>.json
          commit_<sequence>.json
        checkpoints/
          selected/
            staging/<envelope_id>/<envelope_owner_id>/...
            committed/<envelope_id>/
              training_state.pt
              deployable_state.pt
              sidecar.json
              commit.json
          final/
            staging/<envelope_id>/<envelope_owner_id>/...
            committed/<envelope_id>/
              training_state.pt
              sidecar.json
              commit.json
        deployable_extraction.json
        training_report.json
        checkpoint_selection.json
    controls/<control_node_id>/<execution_id>/
      ...                             # identical execution-scoped schema above
  controls/
    registry.json
    zero_coefficient/
      acceptance.json
    shuffled_representation/
      staging/<envelope_id>/<envelope_owner_id>/...
      committed/<envelope_id>/
        shuffle_map.npz
        sidecar.json
        commit.json
  confirmation/
    registry.json
    runs/<execution_id>.json
    aggregate.json
  cleanup/
    <logical_bank>/
      <generation_id>/
        authorization.json
        completion.json
  locks/
    00_submission_authorization.json
    05_finalists.json
    06_final_data_attestation.json
    07_execution.json
  reports/
    screen_aggregate.json
    validation_only_aggregate.json
    final_aggregate.json
    paired_bootstrap/<comparison_id>/
      staging/<envelope_id>/<envelope_owner_id>/...
      committed/<envelope_id>/
        bootstrap_arrays.npz
        sidecar.json
        commit.json
  final/
    task_registry.json
    pretraining_finalist_policy.json
    prediction_spec.json
    capabilities/<task_id>.json
    recovery/
      <recovery_id>.json
    selection/
      row_selection.json
      branch_access.json
      label_escrow/
        staging/<envelope_id>/<envelope_owner_id>/...
        committed/<envelope_id>/
          labels.npz
          sidecar.json
          commit.json
    assignment/
      specification.json
      shards/
        <source_partition>/
          staging/<envelope_id>/<envelope_owner_id>/...
          committed/<envelope_id>/
            assignment.json
            assignment.npz
            sidecar.json
            commit.json
      manifest.json
      audit.json
    predictions/
      <finalist_id>/
        shards/<source_partition>/
          staging/<envelope_id>/<envelope_owner_id>/...
          committed/<envelope_id>/
            logits.npz
            sidecar.json
            branch_access.json
            commit.json
        manifest.json
    evaluations/
      <finalist_id>.json
    metric_join.json
  monitoring/
    reports/<sequence>_<content_hash>.json
    HEAD                              # optional operational atomic pointer
```

The campaign-spec contract has one parameterized route template,
`${campaign_spec_parent}/campaign_spec.json`. The closed parent vocabulary is
`.` for the live specification (and for a dedicated bounded-bootstrap smoke
root that is never promoted) or `planning` for the nonauthorizing reviewed
specification. No other parent value or alias is valid.

The gradient-calibration route's closed `${phase}` vocabulary is `jet_set` or
`relation`; `selection.json`, `diagnostic_batch.json`, and `manifest.json` are
distinct contracts and cannot be selected as phase aliases.

The canonical confirmation aggregate exists only at
`confirmation/aggregate.json`; there is no second copied artifact under
`reports/`. Every `staging/committed` pair in this layout follows the common
Section 21 envelope protocol. An ellipsis denotes the already-enumerated
execution-scoped training schema, not an unregistered artifact family. Every
contract in Section 21 therefore has one canonical root-relative publication
route template; no producer may invent an unregistered parameter value,
timestamped alias, or second authoritative copy.

The population-scoped shared reservation and execution claim are deliberately
outside either campaign root:

```text
<checkpoint_namespace>/final_claims/<final_population_sha256>/
  population.json
  population_disjointness.json
  population_registration.json
  reservation.json
  legacy_cancellation.json        # required when legacy jobs existed
  execution_claim.json            # exclusive, exact-owner idempotent

<checkpoint_namespace>/final_claims/exposure_ledger/
  registrar.lock                  # operational exclusive-lock inode
  legacy_final_exposure.json      # immutable marker for pre-shared exposure
  proposals/<registration_owner_id>/...
  generations/<sequence>_<content_hash>.json
  HEAD.json                       # atomic validated pointer, not identity
```

The common exposure ledger is also outside campaign roots and indexes every
population reservation/claim by identity-set hash. Under the registrar lock,
`HEAD.json` is atomically replaced only after the next immutable generation and
registration have been fsynced; readers validate the full hash/sequence chain.
It contains no model prediction or label payload.

Both the versioned parent final worker and the representation worker resolve
this same path from the complete authenticated final-role population, then
validate the exact task/capability row. A campaign-local lookalike does not
satisfy the shared-claim contract.

The exact filenames may be constants in the implementation, but they cannot
depend on timestamps. Campaign identity is a canonical content hash, while
the human-readable directory may include mode and pushed commit for operator
clarity.

## 32. Required test matrix

### 32.1 Contract and graph tests

Add focused tests for:

- exact four branches and 24 unique node IDs;
- M1--M6 teacher mappings and cold/warm predecessor rules;
- no cross-track, cross-strategy, or training-head warm parent;
- graph acyclicity and complete finite-result descendants;
- schema freeze -> tap implementation/parity -> architecture attestation ->
  parent import ordering, with pre-parity attestation/import rejected;
- parent-import rejection for every stale/missing hash class;
- rejection of the current class-weighted-KD runtime under the required
  unweighted-KD parent-loss attestation, plus acceptance only after corrected
  loss/report/checkpoint lineage agrees;
- exact four-row control registry distinct from the 24-node primary graph;
- execution IDs derive from logical-bank/purpose identity, confirmation
  registries enumerate those IDs before generation construction, and physical
  generations separately bind the registry without a hash cycle;
- overlay recipe exactness and rejection of CLI/environment overrides;
- unchanged validation selector order: macro AUC, CE, logR50, earliest update;
- role access: train targets only, validation selection only, sealed final
  test;
- every contract version and canonical content hash.

### 32.2 Model-surface parity tests

With an installed Weaver, test both ordinary and native-offline models for:

- public-forward versus surface-forward logits in FP32;
- public-forward versus surface-forward input and parameter gradients;
- exact named shapes/dtypes/masks at block two and penultimate layers;
- correct trim/permutation propagation into masks, vectors, canonical token
  IDs, and pre-transform family codes, with no auxiliary feature reaching the
  embedding;
- charged/neutral TOFF separation;
- one forward call producing logits and all target surfaces;
- no state mutation in evaluation target construction;
- strict deployable extraction/load and absence of training-only keys.

Synthetic Weaver doubles may test failure topology locally, but they cannot
replace the installed-Weaver acceptance.

### 32.3 Mathematical loss tests

For `RSET` and `RREL`, require:

- token permutation and padding invariance;
- unequal particle counts and independent orders;
- one-token and empty-family behavior;
- exact token-weight normalization and softened-pT formula;
- deterministic top-32 selection and tie breaking;
- relation strata boundary behavior at exactly `0.05` and `0.20`;
- canonical-JSON UTF-8 seed payload/hash fixtures for token/relation RFF,
  calibration-row ranking, and shuffle ranking, including first-eight-byte
  big-endian seed conversion;
- pair-count/ESS eligibility at exactly 3/4 pairs and immediately below/at
  `ESS=3.0`;
- independent student/teacher pair populations;
- no cross-family TOFF relation;
- RFF resource determinism and exact finite-kernel identity from frozen arrays;
- slow pairwise finite-kernel MMD equals the cached feature-mean loss under the
  frozen FP64/FP32 tolerances across all fixtures;
- analytic finite-kernel gradients equal autograd, including normalization,
  under the frozen seed-993/994 tolerances;
- ideal infinite-RBF value-quality and joint-rotation loss gates meet the
  frozen evidence-backed thresholds, while ideal-gradient and
  rotation-gradient values remain report-only and are never mislabeled as
  fidelity/invariance acceptance;
- finite, nonzero gradients to student jet and token states;
- detached/no-gradient teacher targets;
- zero encoder gradient for a deliberately raw-physics-only relation loss;
- FP32 normalization, Gram construction, RFF evaluation, and reductions under
  BF16 model execution;
- class-weighted per-jet and geometric-pair reductions;
- one-row Gram behavior;
- projection identity initialization, no weight decay, orthogonality,
  nonfinite-projection failure, and finite `condition_number > 1e4`
  report-only continuation.

### 32.4 Calibration and schedule tests

Require exact tests for:

- smallest-hash 4,096-row selection, ascending selection-digest/canonical-
  identity tie-break order, ordered-list hash, and no validation identities;
- 16 consecutive canonical calibration batches at pilot/production size;
- jet/set calibration occurs exactly after pass two and before pass three;
- relation calibration occurs exactly after pass four and before pass five;
- RNG/trimmer/optimizer state unchanged by calibration;
- exactly one stochastic student forward per calibration batch supplies base
  and every component gradient norm, with component support changing only the
  reduction over those same tensors;
- preemption immediately before and after either calibration barrier resumes
  without skipping or duplicating calibration;
- exact `validation -> required calibration -> diagnostic -> boundary commit`
  sequencing at passes 2 and 4, with explicit `not_yet_calibrated`/strategy-null
  fields and no future scale visible to a diagnostic;
- screen/confirmation execution IDs scope every calibration/resume/report path;
  20 concurrent M6 confirmations cannot collide;
- FP64 median calculation and hexadecimal persistence;
- finite out-of-range scale, insufficient valid support, zero gradient, or low
  relation eligibility produces the exact inactive-component state without
  suppressing the node, while a nonfinite calibration fails closed;
- jet/set ramp values immediately before, at, and after passes 2 and 6;
- relation ramp values immediately before, at, and after passes 4 and 8;
- `rho_repr=0` equality of logits, base loss, shared gradients, optimizer
  update, RNG state, and checkpoint bytes where serialization permits;
- smoke's explicit real-data path probe plus all-component eligible post-tap
  fixtures despite the two-update zero ramp; every component must have a finite
  positive smoke-only scale and nonzero gradient even when production support
  would mark it inactive;
- exact 60 validation records and no performance early stopping;
- fixed diagnostic identities/order/mode, one-forward matched-support
  reductions, and byte-identical subsequent training with diagnostics on/off.

### 32.5 Target-bank tests

Require:

- exact selected-row, identity-set, class-count, and source-shard conservation;
- exact `K_token=1024`/`K_relation=256` ordinary/TOFF shapes and
  `1,935`/`3,727` FP32-value core byte formulas at authenticated row counts;
- exact ordinary `F=1,R=1` and TOFF `F=2,R=6` metadata shapes, ordered family
  and reason vocabularies, conservation checks, `75`/`113` metadata bytes, and
  `7,815`/`15,021` total logical bytes per row;
- duplicate, missing, unexpected, reordered, cross-role, nonfinite, wrong-tap,
  wrong-teacher, wrong-track, and wrong-kernel rejection;
- atomic publication and orphan temporary-file rejection;
- invalid preflight leaves no staging directory or build intent; initial-build
  process death immediately after durable build intent but before shard 1,
  after every staged shard, after attestation, after manifest, and immediately
  before/after directory rename has exact same-owner continuation and can never
  expose a partial committed generation;
- shard byte-hash and array logical-hash validation;
- same teacher execution supplies logits and representation summaries;
- RSET ignores but authenticates the shared relation superset;
- strategy consumers obtain identical common target bytes;
- one-time predecessor-logit construction and model release;
- resume rematerialization with equal logical hashes;
- exact target-forward batch partition/backend/runtime-signature enforcement,
  sentinel replay, and full logical-hash equality on authorized reconstruction;
- rejection of a rebuild with changed batch size, TF32 flag, deterministic
  backend flag, device signature, software signature, or teacher checkpoint;
- immutable per-generation paths, consumer registries, and no mutable-latest
  resolution;
- the D100 screen generation has exactly eight primary/control consumers and
  cannot clean up after only the four primary nodes;
- rung `r+1` build-readiness depends on every rung `r` cleanup completion, and
  within-rung warm readiness depends on cold cleanup; dry-run peak-liveness
  analysis cannot schedule two committed banks or all preexisting-teacher banks;
- last-consumer cleanup cannot run early and records exact removed bytes;
- failed/cancelled/timed-out consumers cannot authorize cleanup;
- cleanup authorization is atomically durable before the first removal;
  injected process death immediately after authorization but before deletion
  1, after every individual deletion, and immediately before/after completion
  publication resumes the same authorized deletion set and never classifies it
  as corruption;
- missing shard without cleanup authorization is corruption, while completed
  authorized-cleanup recovery creates a new generation for unfinished
  consumers only;
- screen and confirmation generations do not collide, and reconstruction after
  preemption produces identical logical targets.

### 32.6 Training, stochastic-pairing, and reporting tests

Require:

- parent logit coefficients/temperatures/LR remain exact in every node;
- representation coefficients and ramps are reportable and cannot affect
  checkpoint selection;
- paired stochastic stream equivalence to each logit-only counterpart;
- auxiliary RFF randomness cannot perturb data/model RNG streams;
- cold fresh initialization and warm same-branch deployable loading;
- M2--M6 predecessor logits come from the exact selected prior rung;
- M6 uses TOFF logits/pooled/charged/neutral targets and HLT student inputs;
- interval-mean loss logging survives exact resume;
- the versioned resume envelope rejects a missing/extra state namespace,
  changed target generation, changed calibration, altered RNG/sampler cursor,
  or mismatched binary logical/byte hash;
- resume crash injection after PT publication, after JSON-sidecar publication,
  immediately before/after commit publication, and during old-generation
  pruning always selects the highest fully valid commit and retains the prior
  valid generation;
- selected checkpoint retains training heads while extraction removes them;
- within-class shuffle is a deterministic derangement, changes only
  representation identity joins, is shared by RSET/RREL, validates under its
  binary/JSON shuffle-map contract, and leaves validation untouched;
- every registered finite poor result is complete and cannot suppress a
  descendant;
- ordered pairwise delta tables and undefined gap denominators;
- five-seed terminal confirmation registry is frozen before execution;
- confirmation aggregation uses exactly five raw values, `ddof=1`, the frozen
  Student-t interval, and equal-seed paired contrasts, and never selects a seed;
- mandatory parent report/finalist imports cannot be silently omitted;
- the new 15-class paired bootstrap produces the frozen 2,000 replicate index
  hash, uses identical indices across paired models, preserves every class
  count, exercises raw zero-QCD-pass capping exactly, and is not routed through
  incompatible 10-class/unstratified helpers;
- screening-seed representation endpoints, not confirmation seeds, enter the
  combined finalist registry;
- prediction shards are label-free, disjoint, complete, and bound before the
  locked label join, contain exactly FP32 logits, and reject probability arrays;
- every kernel, shuffle, escrow, assignment, prediction, checkpoint, and
  paired-bootstrap envelope survives death between each payload/sidecar/commit
  write and before/after its directory rename; no partial final-path envelope
  is visible and a corrupt committed envelope is never overwritten;
- HLT, Shell-Exact D100, and native-offline final streamers request only their
  frozen identity/input branch allow-lists; an instrumented label-branch read
  fails before I/O, while the selection task is the only ROOT-label reader;
- a root-local legacy claim cannot satisfy the shared claim, two racing
  evaluators yield exactly one winner, and a validation-only disposition has no
  final tasks;
- two distinct population hashes with overlapping identities raced through the
  global registrar yield exactly one exposure-ledger registration/reservation;
- registrar death before/after bundle rename, ledger-generation publication,
  `HEAD` swap, and reservation publication is same-owner idempotent and cannot
  strand or double-register a population;
- symbolic dependency templates materialize only scheduler-ID tokens, and the
  ledger validator reconstructs the exact submitted argv;
- original plus chained recovery submission ledgers own every replacement job
  ID for monitoring and exact-ID cancellation;
- repeated monitors publish a validated immutable sequence/hash chain; an
  operational HEAD may be lost or stale without overwriting a report;
- every Section 21 contract resolves to exactly one canonical Section 31 route
  template,
  including recovery plans, acceptance artifacts, storage estimate, paired
  bootstrap, and submission authorization;
- absent final outputs may resume under the same owner/path, whereas a
  published corrupt output fails closed and is never overwritten;
- final aggregate cannot read final test without combined new locks.

### 32.7 End-to-end acceptance ladder

Run, in order:

1. pure contract/unit tests without Weaver;
2. installed-Weaver CPU forward/backward parity;
3. local synthetic smoke covering every task and array row;
4. genuine Tigris smoke using production workers, bounded rows, two optimizer
   updates, and explicit full-strength representation backward probes;
5. cache miniature covering ordinary D and TOFF target build/load/cleanup;
6. USR1 preemption and exact-resume miniature;
7. full dry run with exact resources/dependencies and no mutation;
8. complete repository test suite and `git diff --check`;
9. clean pushed-source verification on Tigris.

Production readiness requires all nine. Local mocks do not authorize the
pilot.

## 33. Implementation sequence and acceptance gates

### Block A — freeze contracts and graph

Freeze the parent-loss correction, architecture/tap **schemas**, representation
recipe/resource schemas, node/control graphs, binary resume/control envelopes,
and validation-only access rules. Implement and test the parent-loss runtime
correction, but do not publish architecture or parent-import artifacts before
Block B parity. Add contract/graph tests.

**Gate:** the authoritative base-loss runtime and frozen graph schemas agree,
all 24 primary nodes plus four controls serialize to their frozen graph and
registry hashes, and old HCWDL tests remain unchanged except for an explicitly
versioned loss/final-claim correction.

### Block B — expose authenticated model surfaces

Implement ordinary and TOFF surface forwards plus deployable extraction. Run
installed-Weaver parity, then publish the architecture attestation, validate all
parent checkpoints against it, and only then publish the parent-import artifact.

**Gate:** installed-Weaver FP32 logit/gradient parity, exact shapes/masks,
strict HLT-only extraction, architecture attestation, and parent import all
validate in that order.

### Block C — implement kernels and losses

Implement fixed resources, weighted sketches, family logic, latent relations,
jet/Gram loss, projections, calibration, and schedules.

**Gate:** all mathematical invariance, finite-kernel analytic-gradient, and
control tests pass; ideal-RBF comparisons remain diagnostics; raw
physics cannot masquerade as representation KD.

### Block D — implement target banks

Implement the canonical target-forward spec/attestation pair, one-forward
source-sharded
construction, logical-bank/per-generation manifests, RAM joins,
predecessor-logit RAM caches, resume validation, cleanup, and generation-aware
recovery.

**Gate:** exact coverage/lineage/signature/reconstruction tests pass; screen,
confirmation, and recovery generations cannot collide; and a bounded
ordinary/TOFF cache miniature stays within its measured memory/storage plan.

### Block E — implement node training

Wire the locked base loss plus scheduled representation auxiliary, exact
initialization, validation, selection, reporting, preemption, and extraction.

**Gate:** zero-coefficient parity, two-update smoke, full-loss probe,
nontrivial resume, and one-pass validation fixture all pass.

### Block F — implement campaign orchestration

Implement the just-in-time bank DAG, four ascents, controls, cleanup,
confirmation registry, resources, dry run, submission, monitoring, recovery,
and exact cancellation.

**Gate:** every task in a dry-run command plan has declared validated inputs,
outputs, resources, and dependencies; no scientific argument is supplied by
an array-index accident or environment override.

### Block G — reporting and final boundary

Implement paired aggregates, representation diagnostics, immutable final
disposition, shared population-scoped reservation/legacy cancellation/execution
claim honored by both evaluator families, complete parent-union finalist
registry, label-free prediction shards, HLT-only final evaluation, and final
aggregate.

**Gate:** a forged/stale/cross-campaign lock fails; a poor but finite model
completes; final-test reads are impossible before the combined lock.

### Block H — real acceptance and pilot authorization

Run the full acceptance ladder in Section 32.7, record measured resources,
update the handoff, and obtain explicit user authorization for the exact
clean pushed campaign.

**Gate:** genuine Tigris miniature evidence exists. This plan alone never
satisfies the gate.

After every block, update `docs/HANDOFF.md` with files, contracts, tests, real
Weaver/Tigris evidence, and the next blocker. If donor code is copied from an
outside repository, record the exact donor path and commit in
`docs/LEGACY_SOURCE_MAP.md`; the PDF concept itself is not donor source code.

## 34. Frozen values and deliberately operational values

### 34.1 Frozen scientific values

The following are frozen by this plan and require a new contract/campaign to
change:

- four full ascents and 24 node identities;
- D0/D25/D50/D75/D100/TOFF representation-source ladder;
- cold versus same-branch warm initialization;
- HLT/D and TOFF named tap semantics;
- RSET and RREL component definitions and weights;
- `rho_repr=0.10`;
- token and relation finite spectral-kernel bandwidths/RFF widths and the
  explicit non-equivalence to infinite-RBF gradients;
- top-32 relation population and fixed deltaR strata;
- M6 charged/neutral separation;
- projection structure, identity initialization, regularizer, and reset rule;
- gradient-calibration algorithm/population/bounds;
- canonical target-forward 256-row per-source batch partition, precision,
  backend determinism, software/device signature, and exact logical-target
  reconstruction rule;
- pass-based ramps;
- 60 passes, validation every pass, no performance early stopping;
- macro-AUC-first checkpoint selection;
- paired stochastic streams and confirmation seed policy;
- train-only target construction and HLT-only deployment;
- just-in-time shared superset banks and cleanup semantics;
- ordered logit-only/RSET/RREL comparisons and shuffled M5 controls;
- combined final-test boundary.

The exact parent base recipe is also frozen by authenticated import rather
than copied approximately: peak LR `3e-4`, M1 CE/teacher weights `0.25/0.75`,
M2--M6 CE/predecessor/privileged weights `0.25/0.40/0.35`, predecessor
temperature `1`, privileged temperature `2`, effective/microbatch `256/256`,
AdamW/schedule/class-weight details, and BF16 execution come from the exact
authorized `HCWDL_RECIPE/v3` hash. If that parent recipe differs, this plan's
campaign cannot silently adapt to it.

### 34.2 Operational values measured before submission

The following are not scientific tuning knobs and are fixed in the immutable
campaign spec only after measurement:

- CPU count, RAM request, training-job GPU type/count, walltime, and array
  concurrency, provided the relevant scientific runtime signatures still
  validate; the target-builder GPU/runtime is frozen separately above;
- target compression level and post-forward file-I/O chunk size, provided
  decompressed arrays remain exact; neither may alter the canonical
  target-forward batch partition;
- maximum simultaneously open source files;
- validation worker count;
- retry/resume job structure;
- human-readable campaign-root suffix.

Operational choices may not change row order, batches, RNG, target values,
loss arithmetic, checkpoint selection, or model initialization. If a resource
change would alter those, it is a new scientific execution.

## 35. Definition of done

Implementation is complete only when all of the following are true:

1. all contracts in Section 21 exist and reject stale/cross-lineage inputs;
2. the parent-loss and architecture attestations validate against actual
   runtime/checkpoints rather than documentation alone;
3. the exact 24-node graph, four-control registry, and all four ascents are
   executable;
4. ordinary and TOFF surface forwards pass installed-Weaver parity;
5. RSET and RREL losses pass the complete mathematical test matrix;
6. teacher representations are constructed once, compactly shared, loaded
   into RAM, and never recomputed per epoch;
7. every physical target generation is immutable, signature-bound,
   generation-aware in cleanup/recovery, and exactly reconstructible or
   explicitly non-rebuildable;
8. predecessor logits are computed once per node and never per epoch;
9. students train only from HLT inputs and deploy through the unchanged
   HLT-only public model;
10. cold/warm initialization, exact resume, 60 validations, AUC-first
   selection, and extraction are verified;
11. every finite poor result remains a completed row and descendants run;
12. both component controls, both shuffled controls, and zero-coefficient
    parity execute under their versioned registries;
13. reporting produces every ordered comparison and diagnostic in Section 24;
14. population-scoped shared final locks prevent racing, premature, or retrospective
    test access by either evaluator family;
15. local focused and full tests pass;
16. a genuine Tigris production-worker miniature, cache miniature, and USR1
     resume test pass under measured resources;
17. `docs/HANDOFF.md`, runbook documentation, and donor map are current;
18. the exact source is committed, pushed, clean, dry-run reviewed, and only
     then explicitly authorized by the user.

Until item 18, the correct project status is **implemented locally but not
authorized for pilot submission**. Local source and tests do not substitute
for installed-Weaver parity, measured Tigris resources, or the genuine Tigris
miniature.

## 36. Compact scientific summary

The primary `RSET` experiment distills three complementary signals into an
HLT-only ParT: class logits, an exact paired jet representation, and a
matching-free weighted distribution of privileged token representations. A
small identity-initialized linear projection handles coordinate rotation, a
strictly orthogonal-basis-invariant jet Gram term guards against
projection-only fitting, and a
fixed per-jet finite-spectral-kernel MMD supports different particle counts and
orderings without claiming infinite-RBF gradient equivalence.

The equal-budget `RREL` alternative adds one corrected notion of structure: the
distribution of **learned latent cosine relations**, stratified by fixed raw
angular regions. The raw geometry chooses which relations to summarize but is
not itself optimized, so the term genuinely trains the encoder. It aligns
per-jet marginal relation distributions, not graph topology; different pair
graphs can share the same summaries. TOFF charged and neutral spaces remain
separate throughout.

Running both strategies as complete cold and warm ascents provides the
predeclared screening comparisons for whether representation KD helps at one
rung, compounds, is stored better by warm continuation, and justifies the
relational package. Those full-ladder and M5 mechanism comparisons remain
descriptive single-seed screens. Only terminal M6 variance is conditionally
replicated; the confirmation seeds are not finalist candidates. The logit-only
ascents, zero-coefficient parity, component controls, and within-class
shuffled-pair controls make the screening evidence interpretable without
overstating full-system uncertainty.

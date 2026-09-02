# HCWDL Offline+HLT Concatenation, Fusion, and Withdrawal Plan

Status: implementation-authoritative scientific plan. No implementation or
live submission is authorized merely by the presence of this document.

## 1. Scientific motivation

The full-cardinality persistent-support anchor provides an informative new
observation. On the complete validation role, the pure-offline `U000` oracle
and the persistent-support anchor have nearly identical macro AUC, while the
persistent anchor has materially stronger macro background rejection:

```text
U000             accuracy 0.818691   AUC 0.957972   R50 4884.3
SP4P_U000        accuracy 0.818101   AUC 0.957987   R50 5218.4
```

`SP4MT20_U000` exactly reproduces `SP4P_U000`, as intended, because the two
anchors use the same view, initialization, seed, schedule, and CE-only loss.
MT20 supervision begins only after the anchor. This equality is a
reproducibility result, not an independent replicate.

The comparison nevertheless suggests that native HLT particles can contain
useful information not represented by the pure-offline collection. This plan
asks three ordered questions:

1. Does a single Particle Transformer benefit from seeing the complete
   offline and HLT token collections together?
2. Does an explicit two-stream architecture exploit their complementary
   information better than simple concatenation, and which fusion topology is
   the best bridge to a deployable HLT-only model?
3. If unrestricted fusion is stronger, can its prediction be retained while
   continuously removing an explicitly gated offline residual from an
   otherwise ordinary HLT Particle Transformer?

The first two questions are privileged oracle studies. They do not attempt to
remove offline information. The third question is a distinct training-time
privilege-withdrawal study. Scientific performance in one study never causes
an already registered row to fail or disappear.

### 1.1 Selected best-version design

The primary implementation decision is frozen as follows:

```text
oracle ceiling:
    symmetric two-stream local processing followed by shared fusion

transferable primary model:
    canonical HLT ParT prediction branch
    + one-way offline-context cross-attention after HLT blocks 2, 4, 6, 8
    + an explicit scalar offline-residual gate alpha

unrestricted primary oracle:
    alpha = 1

deployable endpoint:
    alpha = 0
    offline encoder and every cross-attention module removed
    one ordinary HLT-only ParT remains

withdrawal:
    frozen alpha=1 teacher
    trainable cosine alpha path
    exact alpha=0 HLT path trained on every update
```

This is the preferred compromise between oracle strength and transferability.
The symmetric model can discover whether unrestricted bidirectional fusion has
a higher ceiling, but the HLT-anchored model keeps the classifier attached to
the representation that must survive deployment. Offline information enters
only through named residual adapters with one common removable control. The
plan does not use `F(H,H)` as deployment, does not replace offline particles
with HLT particles inside an offline-role branch, and does not rely on random
offline-token dropout to define the endpoint.

## 2. Non-negotiable boundaries

- Deployable inference consumes HLT data only.
- Offline particles are authorized only for explicitly labeled oracle,
  teacher, or training-time privileged roles.
- Train and validation use the authenticated full mapped populations. Final
  test remains sealed.
- All comparisons use the same authenticated jet identities, labels, split,
  HLT view, and full-cardinality offline/HLT source lineage.
- Construction indices, assignment indices, degradation coordinates, file
  order, and row order are forbidden model inputs.
- Source collection identity may enter only through an explicit versioned
  domain embedding. It is not a substitute for particle identity.
- No view may be silently truncated. Required sequence capacities and memory
  are established by an all-row audit before model training.
- Poor accuracy, AUC, rejection, calibration, or transfer is a valid result.
  Invalid lineage, stale inputs, forbidden access, corrupt artifacts, or
  nonfinite required quantities fail closed.

## 3. Common authenticated inputs

The studies reuse read-only:

- the full-cardinality lexicographic bottleneck assignment;
- canonical HLT particles and the collection-aware raw offline endpoint;
- the complete train and validation row selections;
- the existing label map and 15-class output order;
- `M0CE60`, pure-offline `U000`, and persistent-support `SP4P_U000` as
  reporting controls;
- the established recovery convention `M0CE60 = 0%`, pure-offline
  `U000 = 100%`.

The new studies create separate contract families, source locks, roots,
worktrees, ledgers, job namespaces, reports, checkpoints, monitoring, and
restart-from-zero recovery. They have no write path or Slurm dependency into
any running four-spine, persistent-support, MT20, or attention campaign.

## 4. Canonical token collections

For each jet, define:

```text
O = all visible authenticated offline particles in native offline order
H = all visible authenticated HLT particles in native HLT order
```

`O` uses the same HLT-compatible projected endpoint schema already authorized
for the pure-offline and persistent-support controls. `H` uses the canonical
HLT schema. The existing matching remains available for audits and optional
diagnostics, but match indices are never model inputs.

The concatenated sequence is exactly:

```text
C = O followed by H
```

This deliberately retains both representations of a matched physical
particle. Offline-only and HLT-only particles are also retained. The model is
being asked whether both reconstructions jointly contain useful information;
this stage must not deduplicate them.

The all-row capacity audit records, separately for train and validation:

- `n_offline`, `n_hlt`, and `n_offline + n_hlt` distributions;
- maximum and high quantiles of the combined length;
- projected feature, vector, mask, and attention-memory use;
- the number of rows that would exceed each candidate capacity;
- confirmation that the selected capacity loses zero authenticated tokens.

The present endpoint cap of 200 does not imply that a concatenated cap of 200
is valid. If each source can reach 200, the concatenated implementation must
support up to 400 tokens or fail before training. It may not keep the first
200 tokens, select by transverse momentum, or otherwise hide truncation.

## 5. Study A: single-stream concatenation oracle

### 5.1 Question

Can ordinary Particle Transformer self-attention exploit the complete `O + H`
set, and does an explicit source label help it distinguish reconstruction
domains?

### 5.2 Registered fresh arms

The matched screen contains two fresh CE-only fits:

```text
CONCAT_UNTAGGED    one ParT over O + H, no source-domain embedding
CONCAT_TAGGED      one ParT over O + H, with a learned source-domain embedding
```

Before that paired screen, one explicitly authorized feasibility pilot may run
`CONCAT_TAGGED` alone.  The pilot is a single fresh fit with the exact frozen
60-pass, global-batch-256, matched-seed recipe below.  It exists to answer the
first-order question requested by the campaign owner: whether tagged `O + H`
is competitive with `M0CE60`, pure-offline `U000`, and persistent-support
`SP4P_U000`.  It does not replace the later tagged-versus-untagged comparison,
does not establish an architectural advantage from one seed, and cannot be
used as a deployable endpoint.  Its source locks, jobs, reports, checkpoints,
monitoring, and restart-zero recovery are isolated from every running ladder.

The tagged model uses two learned content-source embeddings:

```text
offline token -> OFFLINE_CONTENT
HLT token     -> HLT_CONTENT
```

The embedding is added after the appropriate numerical input transform and
before particle-attention blocks. It is not appended as an unnormalized
physics feature. Padding has no source identity and remains fully masked.

Both arms otherwise use the same standard ParT trunk, class-attention head,
initialization domain, label supervision, optimizer, batch semantics, and
checkpoint-selection rule. The only within-study variable is the source
embedding.

### 5.3 Frozen training comparison

The first screen uses one matched seed and the same CE-only 60-pass recipe as
the persistent-support anchor. Global batch size remains 256. Any change
required solely because a 400-token sequence exceeds measured memory must be
registered as an operational batch-accumulation transformation that preserves
the same global batch and update sequence; it cannot silently change the
scientific batch.

The aggregate compares both arms with:

- `M0CE60`;
- pure-offline `U000`;
- persistent-support `SP4P_U000`;
- identical-seed fresh HLT-only and offline-only controls if the imported
  controls do not share the exact selected training recipe.

If a later confirmation is authorized, all registered oracle arms are rerun
under at least three paired seeds. A single favorable seed is not reported as
an established architecture advantage.

## 6. Study B: unrestricted fusion oracle and transferable fusion

### 6.1 Question and frozen design decision

Does explicit domain-local processing recover more complementary information
than a flat concatenated sequence, and can that information enter the HLT
network through a connection that can later be removed exactly?

The oracle screen registers two deliberately different fusion topologies:

1. a symmetric concatenate-after-local-processing model is retained as the
   high-capacity oracle-ceiling control;
2. an asymmetric HLT-anchored model is the primary architecture for eventual
   privilege withdrawal.

Both are unrestricted CE-only oracles in Study B. Offline information remains
available throughout every training and validation forward. Study B contains
no gate annealing, modality dropout, invariance loss, KD, or representation
alignment. This separation is essential: the best attainable joint-input
teacher must be measured before any pressure is applied to make it
deployable.

### 6.2 Symmetric fusion ceiling

The symmetric ceiling retains the original architecture proposal:

```text
O -> offline adapter -> two offline-local blocks --+
                                                   |-> concatenate
H -> HLT adapter     -> two HLT-local blocks ------+      -> six shared blocks
                                                          -> class head
```

The two streams have distinct adapters, local-attention parameters, branch-
role embeddings, and content-source embeddings. Their encoded token sets are
concatenated only after the two local blocks. Six shared particle-attention
blocks then allow unrestricted bidirectional interaction. The depth on each
token path remains the established eight particle blocks, followed by the
standard class-attention blocks and 15-class head.

This topology is intentionally symmetric and expressive. It is not the
withdrawal student: once tokens enter the shared blocks, offline influence is
distributed throughout the common representation and has no exact removable
boundary.

The registered matched-seed controls are:

```text
SYMMETRIC_FUSION_OO    O in both branches
SYMMETRIC_FUSION_HH    H in both branches
SYMMETRIC_FUSION_OH    O in the offline-role branch, H in the HLT-role branch
```

All three instantiate every parameter. Branch-role embeddings stay distinct;
content-source embeddings describe the actual supplied content. The arms have
identical parameter counts, although `2*n_offline`, `2*n_hlt`, and
`n_offline+n_hlt` may yield different measured attention costs.

### 6.3 Primary HLT-anchored architecture

The primary fusion model is one-way and HLT-centered. Its HLT branch is the
only path connected to the class-attention head. The offline branch is a
context encoder whose information can enter the HLT tokens only through four
explicit residual cross-attention adapters:

```text
O -> offline adapter -> eight offline context blocks -> K,V context
                                                        |
                                                        |  O -> H residuals
                                                        v
H -> HLT adapter -> eight canonical HLT blocks -> class attention -> logits
                     ^       ^       ^       ^
                     |       |       |       |
                  after 2  after 4  after 6  after 8
```

At injection block `l` in `{2, 4, 6, 8}`, the update is exactly:

```text
h_l_base = HLT_BLOCK_l(h_(l-1))
r_l      = CROSS_ATTN_l(
               Q = LN_Q_l(h_l_base),
               K = LN_K_l(o_l),
               V = LN_V_l(o_l),
               mask = H_mask x O_mask,
               bias = CROSS_PAIR_BIAS(H_four_vector, O_four_vector),
           )
h_l      = h_l_base + alpha * sigmoid(g_l) * W_l(r_l)
```

`CROSS_PAIR_BIAS` uses the same registered standard-four geometric variables
computed directly for every active HLT/offline token pair. It uses no matching
index, assignment score, source row, or degradation coordinate. Padding is
masked before softmax and cannot affect an active token.

`alpha` is an externally supplied scalar privilege gate. In Study B it is
fixed to one. `g_l` is a learned per-block residual gate. Each output
projection `W_l` is zero-initialized, so the initial HLT prediction is exactly
the imported HLT checkpoint even though the offline encoder is present. The
output projections, gates, cross-attention modules, both encoders, and HLT
head are then trainable in the unrestricted oracle fit.

Information flow is only `O -> H`: offline tokens never receive HLT updates,
and there is no class head on the offline stream. This asymmetry is the main
design choice. It gives the oracle enough capacity to exploit offline context
while preserving an ordinary HLT backbone whose offline residuals can be
bypassed without replacing its tokens, duplicating HLT into a second branch,
or changing the classifier head.

### 6.4 Initialization and exact alpha-zero path

The HLT adapter, eight HLT blocks, class-attention blocks, and head are loaded
from the selected matched-seed `M0CE60` exact-HLT checkpoint. The offline
adapter and eight context blocks are loaded from the selected matched-seed
pure-offline `U000` checkpoint. Optimizer and scheduler state are never
imported. Newly introduced cross-attention, pair-bias, gate, and output-
projection parameters use a graph-bound initialization seed, with the output
projection fixed to exact zero as described above.

When `alpha == 0`, dispatch does not load `O`, execute the offline encoder,
construct cross-pair features, or execute a cross-attention adapter. The
forward is exactly:

```text
H -> imported-shape HLT adapter -> eight HLT blocks
  -> standard class-attention blocks -> standard 15-class head
```

The deployable extractor therefore emits only those ordinary HLT modules.
Offline-encoder and cross-attention parameters are absent, rather than merely
multiplied by zero after evaluation. A state-dictionary projection test must
prove that the extracted model and `alpha=0` wrapper are bitwise identical in
FP32 and within the registered BF16 envelope. The endpoint is denoted
`A(H, empty; alpha=0)`, shortened to `A_0(H)`. The earlier `F(H,H)` proposal is
explicitly superseded; duplicating HLT into an offline-role branch is not the
deployment endpoint.

### 6.5 HLT-anchored oracle controls

Three matched rows isolate the primary model's claims:

```text
HLT_WARM_CONTINUE       M0CE60 continued with CE, no context branch
ANCHORED_FUSION_HH      HLT primary plus H context, alpha=1
ANCHORED_FUSION_OH      HLT primary plus O context, alpha=1
```

`ANCHORED_FUSION_HH` and `ANCHORED_FUSION_OH` instantiate the identical
architecture, parameter registry, optimizer, schedule, and seed domains. For
both rows the context branch starts from the same U000 weights; only actual
context content and its content-source embedding differ. This deliberately
awkward but strict control keeps initialization fixed while asking whether O
contains value beyond a second H copy. `HLT_WARM_CONTINUE` controls the gains
available from simply continuing the imported HLT checkpoint for the same
number of updates.

The primary comparisons are:

```text
SYMMETRIC_FUSION_OH - CONCAT_TAGGED
    value of explicit domain-local symmetric processing

SYMMETRIC_FUSION_OH - {SYMMETRIC_FUSION_OO, SYMMETRIC_FUSION_HH}
    complementary dual-domain value under one fixed architecture

ANCHORED_FUSION_OH - ANCHORED_FUSION_HH
    value of offline rather than duplicate-HLT context

ANCHORED_FUSION_OH - HLT_WARM_CONTINUE
    complete gain from the removable offline-context architecture

SYMMETRIC_FUSION_OH - ANCHORED_FUSION_OH
    oracle ceiling versus withdrawal-friendly topology
```

The symmetric and anchored differences answer different questions and are not
collapsed into one architecture claim. Every row completes regardless of its
metric.

## 7. Oracle-screen campaign and reporting

Studies A and B share one authenticated oracle-screen campaign. Its eight
fresh fits are independent siblings after the common gates:

```text
authenticate
  -> all-row dual-source capacity/resource audit
  -> genuine single-GH200 production miniature
  -> {
       CONCAT_UNTAGGED, CONCAT_TAGGED,
       SYMMETRIC_FUSION_OO, SYMMETRIC_FUSION_HH,
       SYMMETRIC_FUSION_OH,
       HLT_WARM_CONTINUE,
       ANCHORED_FUSION_HH, ANCHORED_FUSION_OH
     }
  -> validation aggregate
  -> completion
```

The first screen uses one matched seed and the same CE-only 60-pass budget,
global batch 256, optimizer family, validation cadence, and checkpoint rule
across comparable rows. The two concatenation arms and three symmetric arms
are trained from scratch. The three anchored-family rows use the declared
warm starts in Section 6.4. Warm-start parents and newly initialized parameter
sets are recorded explicitly. The report must not describe a warm-started
anchored row and a from-scratch symmetric row as differing only in topology.

The aggregate reports:

- accuracy and macro one-vs-rest AUC;
- linear macro background rejection at 50% signal efficiency;
- Hbb, Hcc, Hqq, Hgg, top, and every remaining available per-class rejection;
- recovery from `M0CE60` to pure-offline `U000` in AUC and linear R50 space;
- all registered paired differences above;
- parameter count, active parameters, tokens, attention elements, peak CPU
  RAM, peak GPU memory, rows/second, and wall time;
- selected pass and complete validation history;
- matching diagnostics only as non-input explanatory metadata;
- `final_test_accessed: false`.

Because R50 is threshold-sensitive, any confirmation report includes paired
bootstrap intervals over validation jets. If an oracle advantage is large
enough to motivate Study C, the eight-arm screen is rerun under at least three
paired seeds before making a general architecture claim. One seed may be used
to decide whether further study is worth the compute, but not to establish a
new production method.

The primary withdrawal teacher is `ANCHORED_FUSION_OH`, not whichever row
happens to rank first by one noisy validation metric. It proceeds only if it
beats `HLT_WARM_CONTINUE` and shows useful absolute or per-class improvement.
`SYMMETRIC_FUSION_OH` remains the joint-input ceiling. If only the symmetric
model improves, its frozen probabilities may inform the direct-KD control,
but the anchored withdrawal curriculum is not claimed to have a strong
starting oracle. Failure to improve is a valid completed result.

Study C requires separate human authorization after the complete oracle
screen is reviewed.

## 8. Study C: gated offline-residual withdrawal

### 8.1 Goal and frozen teacher

Study C starts from the selected `ANCHORED_FUSION_OH` checkpoint at
`alpha=1`. One immutable copy is frozen as teacher `T`; a second copy is the
trainable withdrawal student `S`. Teacher parameters never change and never
receive gradients. The durable teacher bank contains only identity digests
and 15-class probabilities on train and validation; hidden states and token
views are never banked.

The three relevant predictions are:

```text
p_T       = softmax(A_T(O, H; alpha=1) / T)   frozen privileged teacher
z_alpha   = A_S(O, H; alpha=a_p)              trainable curriculum path
z_0       = A_S(H, empty; alpha=0)             exact deployable HLT path
```

The student always evaluates `z_0`. It therefore learns the deployable path
from the first update instead of hoping that a privileged network survives a
late input replacement.

### 8.2 Primary cosine withdrawal schedule

For pass `p`, the maximum active offline residual is:

```text
a_p = 1                                              for 1 <= p <= 10
a_p = 0.5 * (1 + cos(pi * (p - 10) / 50))           for 11 <= p <= 60
a_p = 0                                              for 61 <= p <= 100
```

Every batch is the deterministic three-level sandwich
`teacher alpha=1`, `student alpha=a_p`, and `student alpha=0`. When `a_p=0`,
the two student routes are evaluated once and their algebraically combined
loss is applied; duplicate execution is forbidden. The teacher probabilities
are identity-joined from the compact bank, so the teacher model need not run
inside every training job.

The optimization horizon is at most 100 passes, with minimum pass 60,
patience 15 beginning only after pass 60, and exact-best restoration.
Checkpoint selection uses macro AUC from `z_0` only. Privileged validation
cannot select a deployable checkpoint.

### 8.3 Primary withdrawal loss

At temperature `T=2`, define:

```text
KD_T(z) = T^2 * KL(p_T || softmax(z / T))

J_logit = KL(stopgrad(softmax(z_alpha)) || softmax(z_0))

J_repr  = mean over active HLT tokens and l in {2,4,6,8} of
          SmoothL1(
              LN(h_0_l),
              stopgrad(LN(h_alpha_l)),
          )

L = 0.25 * CE(z_0, y)
  + 0.30 * KD_T(z_0)
  + 0.15 * CE(z_alpha, y)
  + 0.20 * KD_T(z_alpha)
  + 0.05 * J_logit
  + 0.05 * J_repr
```

`J_repr` is unusually clean here because both routes contain the exact same
ordered HLT tokens; no HLT/offline assignment is needed to compare them.
Layer normalization is feature-wise with fixed epsilon, padded tokens are
excluded, and the curriculum representation is stop-gradient so the weaker
endpoint is pulled toward the currently richer path rather than the reverse.
The logit term follows the same direction. When `a_p=0`, both consistency
terms are exactly zero and the surviving CE/KD coefficients combine onto the
single ordinary HLT forward.

These coefficients, stop-gradient directions, normalization, and gate
schedule are part of the versioned recipe. Changing one creates a new arm; it
is not an operational retry.

### 8.4 Mandatory controls

Study C registers three independent rows initialized from matched parent
weights:

```text
FUSION_DIRECT_KD_WARM
    M0CE60 HLT model -> ordinary HLT ParT
    frozen anchored-teacher logits, C25/P75, T=2

FUSION_WITHDRAW_COS
    copied ANCHORED_FUSION_OH student
    primary cosine residual withdrawal and full loss above

FUSION_WITHDRAW_STEP
    copied ANCHORED_FUSION_OH student
    a_p=1 through pass 60, then a_p=0
    otherwise the same endpoint-forward and loss machinery
```

The direct control is warm-started because the deployable branch inside the
withdrawal model is also warm-started from `M0CE60`; this keeps the comparison
from becoming a warm-start ablation. A separately reported from-scratch KD
row may be added later, but it does not replace the matched warm control.

The step arm asks whether any gain comes merely from a block of privileged
training followed by HLT-only refinement. The cosine arm asks whether keeping
the exact endpoint trained while smoothly reducing residual offline influence
is better. All three rows complete independent of performance.

### 8.5 Endpoint capability proof

`A_0(H)` is accepted as deployable only if all of the following hold:

- the extracted artifact contains only the standard HLT adapter, eight HLT
  particle blocks, class-attention blocks, and 15-class head;
- offline files, assignments, content embeddings, pair features, and context
  modules are absent from its input signature and state dictionary;
- wrapper `alpha=0` and extracted-model logits reproduce on mixed-length HLT
  batches under the registered precision envelopes;
- perturbing, deleting, or denying every offline artifact leaves extracted
  inference byte-identical;
- construction/source indices remain absent from model inputs;
- the extracted endpoint passes the existing label-free HLT inference audit.

Passing this proof is stronger and cheaper than the superseded `F(H,H)`
endpoint: deployment is one ordinary HLT branch, not a duplicated two-branch
network.

## 9. Study C evaluation

Validation reporting includes:

- frozen `ANCHORED_FUSION_OH` teacher at `alpha=1`;
- its same-checkpoint `A_0(H)` result before withdrawal;
- `FUSION_DIRECT_KD_WARM`;
- cosine- and step-withdrawal extracted `A_0(H)` endpoints;
- `M0CE60`, pure-offline `U000`, `SP4P_U000`, concatenation, the symmetric
  ceiling, and the strongest authenticated HLT-only ladder endpoints.

Every withdrawal pass records `a_p`, privileged and HLT-only validation
metrics, teacher KL, endpoint gap, the six loss components, and all four
learned residual gates. The aggregate reports AUC/R50 recovery, complete
per-class rejection and calibration, selected pass, wall time, and peak
resources. It also plots privileged and deployable trajectories against
offline-residual strength.

A successful result requires improvement of the extracted HLT-only endpoint.
A strong `alpha=1` teacher, a small within-wrapper gap, or a favorable
privileged metric is not deployable performance and cannot be reported as
such.

## 10. Storage and execution policy

- Particle views, local branch activations, attention maps, and representation
  targets are RAM-only and die with the job.
- No concatenated or fusion token cache is written to research-compute storage.
- Durable outputs are limited to compact specifications, locks, reports,
  selected/final model checkpoints, and—only when required for frozen-teacher
  reuse—15-class probability banks plus identity digests.
- No optimizer-state rolling resume is kept. Recovery restarts an incomplete
  fit from update zero in a new source-pinned worktree.
- The selected anchored teacher's probability banks are built once and reused
  by exact identity join. They contain only uint8 identity digests and float32
  15-class probabilities; fused hidden states are never banked.
- The `alpha=0` path does not allocate the offline collection, offline encoder,
  cross-pair tensor, or cross-attention state. The `alpha>0` path constructs
  them batch-locally and releases them before the next batch.
- Resource requests are derived from a genuine Tigris miniature using the
  production worker and the maximum realized token lengths. A full campaign
  is not submitted from synthetic estimates.
- CPU preprocessing must use the established bounded process-parallel source
  path. GPU jobs use one GH200 initially; multi-GPU execution is not introduced
  into the scientific comparison without a separately validated equivalence
  contract.

## 11. Reproducibility and artifacts

Each study versions and authenticates:

- input-view and domain-embedding contracts;
- the all-row capacity/resource audit;
- graph, recipe, seed registry, and architecture registry;
- source and comparison locks;
- genuine installed-Weaver execution acceptance;
- training, checkpoint, probability, and endpoint-capability reports;
- validation aggregate and bootstrap report;
- exact Slurm command plan and submission ledger;
- task attestations, monitor, restart-zero recovery, and completion marker.

The architecture registry records parameter names, shapes, active parameter
counts, branch allocation, source/role embeddings, mask semantics,
cross-attention injection blocks, pair-bias semantics, gate initialization,
warm-start parameter maps, and the exact state-dictionary projection from the
anchored wrapper to `A_0(H)`. Path existence alone never authorizes reuse.

## 12. Required tests before queue readiness

### 12.1 View tests

- concatenation contains every authenticated `O` and `H` token exactly once;
- tagged and untagged views differ only by the registered source embedding;
- masks and four-vectors remain aligned;
- zero-hot/multi-hot raw endpoint identities remain preserved;
- combined lengths are exact and overflow fails rather than truncates;
- row ordering and matching indices cannot enter model features.

### 12.2 Fusion tests

- symmetric `OO`, `HH`, and `OH` arms have identical architecture and
  parameter counts;
- anchored `HH` and `OH` arms have identical architecture, parameter counts,
  initialization parents, and seed domains;
- `OO`, `HH`, and `OH` branch-role and content-source identities are exact;
- fully masked padding cannot affect outputs;
- symmetric branch-local blocks do not cross domains before concatenation;
- symmetric shared blocks permit gradients from both active streams;
- anchored information flow is one-way `O -> H`, and the offline stream has no
  classifier head or HLT-conditioned update;
- cross-attention occurs only after blocks 2, 4, 6, and 8 and uses only direct
  standard-four cross-pair geometry;
- zero-initialized residual projections exactly reproduce the imported HLT
  checkpoint before optimization;
- `alpha=0` skips all offline and cross-attention execution, while `alpha=1`
  supplies nonzero gradients to every active context module;
- label-free inference works for every oracle arm.

### 12.3 Withdrawal tests

- the frozen teacher receives `O,H` and no gradient;
- `a_p` follows the exact registered cosine or step schedule and is zero after
  pass 60 in the primary arm;
- the `alpha=1`, `alpha=a_p`, `alpha=0` sandwich is exact and avoids a duplicate
  student forward when `a_p=0`;
- `A_0(H)` reads no offline, assignment, matching, or cross-pair artifact;
- teacher probabilities join by exact canonical identity and never by row
  position alone;
- all six registered loss terms, stop-gradient directions, masks,
  normalization, and T-squared scaling are exact;
- representation consistency compares the same ordered HLT tokens without
  using particle matching;
- checkpoint selection uses only HLT-only validation output;
- wrapper `alpha=0` and the extracted standard HLT model reproduce under the
  registered FP32/BF16 envelopes;
- the final endpoint remains unchanged when offline files are unavailable or
  deliberately perturbed;
- recovery cannot inherit partial optimizer or hidden-state artifacts.

### 12.4 Production acceptance

- focused and neighboring repository tests pass;
- installed-Weaver model construction and backward parity pass;
- a real Tigris miniature traverses maximum-length concatenation, all eight
  oracle arms, nonzero cross-attention gradients, `alpha=0` extraction, and one
  withdrawal transition;
- measured peak RAM/GPU memory and storage remain below locked limits;
- the full dry run contains only the intended new job IDs and paths;
- final test remains inaccessible.

## 13. Implementation order

1. Implement and test canonical dual-source views and the all-row capacity
   audit.
2. Implement source-domain embeddings and the two concatenation arms.
3. Implement the symmetric two-local-plus-six-shared oracle ceiling and its
   exact `OO`, `HH`, and `OH` controls.
4. Implement the HLT-anchored one-way cross-attention model, warm-start maps,
   zero-residual parity, `HH`/`OH` controls, and exact `A_0(H)` extractor.
5. Build the independent eight-fit oracle-screen campaign, aggregate,
   monitoring, and restart-zero recovery.
6. Run local tests, installed-Weaver parity, a genuine Tigris miniature, and a
   nonmutating dry run.
7. Execute and review the complete oracle screen; obtain separate human
   authorization before starting withdrawal work.
8. Freeze the anchored teacher and implement the identity-joined probability
   bank, matched warm direct-KD control, cosine withdrawal, and step control.
9. Prove exact HLT-only extraction, run the Study C miniature, and then submit
   its independent full campaign.

This order preserves the core causal story: first establish whether joint
information exists, then whether explicit fusion extracts it, and only then
ask how much of it can survive removal of offline inference capability.

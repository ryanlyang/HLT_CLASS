# Luka FullSim: aligned particle-attention distillation down the D ladder

Date: 2026-10-17. Status: **implemented in the isolated `luka_attention`
namespace; local verification is documented in HANDOFF. Ready to prepare a
reviewed SPORC acceptance submission after committing/pushing; real A100
acceptance and separate science authorization remain required.**

See the [implementation plan](plans/LUKA_D_ATTENTION_PLAN.md),
[v1 contract](contracts/LUKA_D_ATTENTION.md), and
[commit/queue runbook](LUKA_D_ATTENTION_RUNBOOK.md).

The user requested the complete `U100 -> D066 -> D033 -> D000` experiment,
explicitly skipping the previously suggested one-step scientific pilot. This
document records that decision and a concrete recommended implementation design.
It does not submit jobs, change the running fixed30 campaign, or authorize
deleting or replacing any existing artifact.

## 1. The idea in plain language

Teach an ordinary single-view Particle Transformer (ParT) both:

1. **What to predict:** the richer-view teacher's class probabilities.
2. **How to relate particles:** the richer-view teacher's attention between
   corresponding particle slots.

Move this supervision down three successive steps until the student consumes
only native HLT particles:

```text
Frozen regular U100 teacher
       |
       | class-probability KD + aligned particle-attention KD
       v
Single-view D066 student, then freeze its selected checkpoint
       |
       | class-probability KD + aligned particle-attention KD
       v
Single-view D033 student, then freeze its selected checkpoint
       |
       | class-probability KD + aligned particle-attention KD
       v
Single-view D000 student: ordinary HLT-only ParT at deployment
```

Every arrow trains a new student from scratch. Each selected student becomes
the immediate teacher of the next rung. There is no live two-encoder fusion
model, concatenated particle input, context branch, or teacher-weight warm start.

The motivation is that ordinary logit KD supervises attention only indirectly
through classification. An explicit attention loss can ask the HLT-side model to
learn particle relationships expressed by richer measurements. This is a
hypothesis, not a guarantee that teacher accuracy or attention can be reproduced
from poorer inputs.

## 2. Why the D ladder permits this comparison

Equal particle counts alone would not be enough. The important property is
**persistent HLT-slot identity across the constructed views**.

For a jet with `N` native HLT particles, the current view builder starts from the
same ordered HLT shell. It replaces or interpolates fields in those slots using
the frozen HLT-to-offline assignment; it does not independently reorder each
D view. At `U100`, the extra offline-only tail has already been removed.

| View | Number of real tokens | Information in persistent slots |
|---|---:|---|
| U100 | N | Offline features in assigned slots; native HLT in unassigned slots |
| D066 | N | Registered transition retaining exactly 2/3 offline fraction |
| D033 | N | Registered transition retaining exactly 1/3 offline fraction |
| D000 | N | Exact native HLT; no offline capability needed |

The fractions describe the registered view transformation, not a claim that
exactly that fraction of particles is offline in every jet. Four-vectors are
interpolated; identity and measurement-validity changes use coherent registered
switches. Existing conventions must be reused without modification.

Thus teacher attention entry `(i, j)` and student attention entry `(i, j)` refer
to the same two persistent slots, even though their measured features differ.
This also holds for initially unassigned slots: they remain native HLT, while
their contextual representations can change because other particles change.

Important limits:

- The assignments are algorithmic full-cardinality salience matches, **not
  truth-level certificates that two reconstructed particles are the same
  physical particle**. Forced or imperfect matches can make targets less useful.
- Slot alignment makes the loss well-defined; it does not establish that every
  teacher relationship is recoverable from HLT.
- Earlier disappointing KD results do not prove that missing alignment caused
  them. The existing matched D ladders already had persistent slots; their
  standard loss did not explicitly use those slots for attention supervision.
- Unequal-count U views are not mathematically impossible to distill, but they
  require an explicit transport/restriction and attention-renormalization policy.
  That extension is outside this first experiment.

Implementation references: [view construction](../src/hlt_classification/cms_proxy_ladder/views.py),
[Luka preparation](../src/hlt_classification/luka_fullsim/regular_cache.py), and
[stage-2 conventions](contracts/LUKA_FULLSIM_STAGE2.md).

## 3. Dataset, split, and the existing U100 anchor

Use the **Luka trial ALLJETS population**, not the separate newer FullStat/Oscar
dataset and not either synthetic JetClass2 proxy:

- 299,234 training jets.
- 74,895 validation jets for checkpoint selection and primary reporting.
- The original 50,000 validation identities for secondary reporting only, using
  the same full-validation-selected checkpoints.
- Natural per-role class composition; no rebalancing, new cuts, or split search.
- Same authenticated views, assignments, units, uncertainty conventions, model
  inputs, and jet identities as the existing matched batch128/fixed30 study.

The historical test was already evaluated and explicitly retired into TRAIN.
**There is no independent test set in this experiment.** File disjointness does
not establish cross-file event independence, which remains unverified. See the
[ALLJETS population contract](contracts/LUKA_FULLSIM_ALLJETS.md).

The proposed common anchor is the validation-selected regular `COARSE_U100`
checkpoint from the fixed30 campaign, not `FUSION_U100`, a technical preflight
model, or the older batch256/long-schedule study. Import it only after its fit
and required artifacts have completed and been authenticated; this document
does not establish its current queue/completion status. Its upstream U-side
training remains exactly as registered; this proposal does not retrain or
change the U ladder.

Known parent locator from this conversation:

```text
Source commit:
74aa58c582ff6f497a2c6d91c503f5489fac3e9a

Campaign container:
/home/ryreu/atlas/HLT_Classification/checkpoints/luka_fixed30_74aa58c5_r1

Science specification:
<campaign container>/science/campaign_spec.json
```

These are locators, not proof of completion or permission to trust file contents.
Before reuse, resolve the anchor through that specification and authenticate its
selected checkpoint, task receipt, training report, exact recipe, original
source/environment, and population/view fingerprints. Its exact checkpoint hash
is **not yet recorded by this document**. If it is unavailable or incompatible,
stop and register an explicit replacement; do not silently choose another model.

## 4. Experimental graph and the matched control

Run the complete attention ladder directly. Also include a matched complete
logit-only D ladder from the **same frozen U100 checkpoint**:

```text
                              common frozen U100
                             /                  \
                     ordinary KD          ordinary KD + attention KD
                         /                          \
                  LOGIT_D066                      ATTN_D066
                       |                              |
                  LOGIT_D033                      ATTN_D033
                       |                              |
                  LOGIT_D000                      ATTN_D000
```

These are proposed node names, not existing runnable task IDs. There are six
new scientific fits: three per arm. Each arm uses **its own immediate parent's
selected checkpoint** for both logit targets and, where applicable, attention
targets. After D066, never mix probabilities from one arm with attention from
the other. No all-prior teacher mixture is introduced.

Corresponding fits have identical architecture, initialization seed, epoch
permutations, data, optimizer, schedule, and checkpoint selector. Use the same
instrumented model adapter in both arms; the control has attention coefficient
zero. Reuse the original per-coordinate initialization/sampler seed integers
from the authenticated fixed30 node records, rather than deriving different
seeds from the new arm names.

Refitting the three logit-only controls gives a clean comparison under the new
adapter. Existing regular/fusion results are useful secondary references, but
must not silently substitute for this matched arm. The primary comparison is
`ATTN_D000` versus `LOGIT_D000`, both single native-HLT models with equal capacity.

Skipping the one-step pilot means **no D033-to-D000 scientific screening stage
and no performance gate before the full ladder**. Technical acceptance remains
necessary for correctness, resource safety, and valid submission. A finite but
weak intermediate result must still advance to the next registered rung.

## 5. Exactly what attention is distilled

### 5.1 Chosen surface

For the initial design, capture the final particle self-attention block:
block 8, zero-based index 7, in the current eight-block ParT. This is **before
the class-attention aggregator**, and is not a class-token saliency vector.

For head `h`, query slot `i`, and key slot `j`, the intended surface is the
actual pre-dropout attention probability:

```text
scores[h, i, j] = installed-Weaver Q/K attention score
                 + that head's native geometric pair bias
                 + the native attention/padding mask

A[h, i, :] = softmax(scores[h, i, :])
```

The installed Weaver implementation is authoritative for Q/K normalization,
scaling, layouts, and masks. This schematic is not permission to substitute an
approximate `QK^T` implementation. Capture probabilities after pair bias and
masking, before attention dropout. Do not distill raw unnormalized scores, omit
the pair bias, or confuse the pair-embedding output with the attention matrix.
Teacher and student each compute their bias from **their own supplied view**;
the student's forward must never receive the teacher's offline geometry.

### 5.2 Head averaging

Average the probabilities across heads, not the scores before softmax:

```text
A_bar[i, j] = mean_h A[h, i, j]
```

Independently trained head numbers have no guaranteed semantic correspondence.
Head averaging avoids imposing arbitrary head-1-to-head-1 matches and stores one
matrix rather than eight. It deliberately sacrifices head-specific information;
head alignment, per-head losses, value-relation losses, and multi-layer targets
are later ablations, not silently included in this first run.

### 5.3 Masks, order, and numerical definition

- Join teacher targets and student inputs by authenticated jet identity.
- Bind the native HLT-slot order, counts, assignments, and per-view fingerprints.
  The current adapters preserve that order; implementation must test it end to
  end through cache loading and attention extraction.
- No independent pT sorting, truncation, token dropping, or random permutation
  without transporting both attention axes and all corresponding tensors.
- Valid keys are the jet's real particle slots; padded keys carry no mass.
  Padded query rows are excluded from the loss.
- Include self-attention on the diagonal as permitted by the native model;
  do not add a new diagonal mask for the auxiliary loss.
- Tracking-unavailable particles remain valid particles. Measurement-validity
  masks must not be confused with particle-presence masks.
- Use all persistent slots, including slots with no offline assignment. There
  is no new matching-confidence cut, pair-selection threshold, or particle crop.

Compute the auxiliary probabilities and loss in FP32. For the `n` valid keys of
each valid query, use the same small smoothing/renormalization on both sides:

```text
epsilon = 1e-8
smooth(P)[j] = (P[j] + epsilon) / (sum_valid_keys P + n * epsilon)
```

Require finite, nonnegative probabilities with positive valid-row mass before
smoothing. Epsilon must not conceal an invalid all-zero or nonfinite target.
Invalid padded rows are excluded before arithmetic; do not compute NaNs and
then multiply them by zero. A one-particle jet has a valid zero attention KL;
empty jets violate this population's input contract and must fail, not be skipped.

## 6. Training objective and schedule

Let `T_logit = 2`, and let `q_T` and `p_T` be the teacher and student class
probabilities at that temperature. The existing classification objective is:

```text
L_base = 0.25 * CE(true_label, student_logits)
       + 0.75 * T_logit^2 * KL(q_T || p_T)
```

The attention term is forward KL, teacher to student. For batch size `B`, and
`n_b` real particle slots in jet `b`:

```text
L_attention = (1/B) * sum_b [
    (1/n_b) * sum_valid_queries_i [
        sum_valid_keys_j A_teacher[b,i,j]
            * (log A_teacher[b,i,j] - log A_student[b,i,j])
    ]
]
```

Here `A_teacher` and `A_student` mean the head-averaged, masked, smoothed
distributions above. Sum the key distribution inside KL, average query rows
within a jet, then average jets. Do not average all batch pairs together, which
would overweight high-multiplicity jets, or divide the key sum by `n_b` again.

```text
L_control   = L_base
L_treatment = L_base + lambda_attention * L_attention
```

**Recommended first-run coefficient: `lambda_attention = 0.1`, constant at all
three rungs and all 30 epochs.** This is a proposed design choice, not a tuned
value or a setting measured to be optimal. Record it explicitly in the new
recipe before any science result is inspected. Do not normalize the three loss
weights back to one, apply the logit temperature to attention, or add another
temperature-squared factor. Attention temperature is 1.

Log unweighted and weighted attention loss, CE, logit KD, attention entropy,
and diagnostic gradient magnitudes. These diagnose a weak or overpowering term;
they must not automatically change lambda. A later coefficient study is a new
declared experiment, not an in-place repair of an inconvenient result.

All student parameters remain trainable. Gradients from the attention term
reach Q/K projections, the applicable pair embedding, and upstream particle
representations. This is not attention-only fine-tuning or frozen-backbone
training. Attention matching does not by itself supervise every V projection
or feed-forward component; CE and logit KD still train the full classifier.

Inherit the fixed30 settings from the [existing plan](plans/LUKA_ALLJETS_30E_PLAN.md)
and [recipe implementation](../src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py):

| Setting | Proposed value |
|---|---|
| Student | One standard 17-input / 11-class ParT; unchanged capacity |
| Initialization | Fresh at every D rung; paired seeds across arms |
| Training | Exactly 30 complete epochs; no early stopping |
| Batch | True 128; accumulation 1; retain final partial batch |
| Optimizer | AdamW; betas (0.9, 0.999), epsilon 1e-8, weight decay 0.01 |
| Learning rate | 3-epoch warmup to 3e-4, then cosine decay to 1.5e-5 at epoch 30 |
| Precision | Existing BF16 forward; FP32 classification and attention losses |
| Validation | Every epoch on all 74,895 validation jets; inference batch 128 |
| Selection | Maximum macro OVR AUC, then minimum CE, maximum macro log-R50, earliest update |
| Support | Native registered support up to 512; no constituent truncation |

Import the exact existing optimizer parameter groups/no-weight-decay exceptions
and selector implementation, not merely an approximate version of this table.
The original 50k subset and attention loss do not select checkpoints. Each
teacher bank comes from the selected checkpoint, not automatically epoch 30.

## 7. Frozen target banks and the runtime strategy

The expensive teacher is evaluated once for each required TRAIN bank, rather
than once per student epoch. During bank generation, set the teacher to eval,
disable gradients and dropout, and use its selected immutable checkpoint.

Required bank contents:

- Temperature-2 class probabilities for ordinary KD.
- For the attention arm, one dense head-averaged attention matrix per training
  jet, using the registered final particle block.
- Jet identities, real token counts, matrix offsets, and authenticated slot
  ordering/fingerprint metadata, separate from model inputs.

The common U100 export supplies both arms. `ATTN_D066` and `ATTN_D033` export
probabilities plus attention for their descendants. `LOGIT_D066` and
`LOGIT_D033` export probabilities only. Terminal D000 models need validation
predictions, not additional attention-teacher banks.

Prefer ragged, bounded, atomically published FP32 shards containing only real
`n x n` matrices. Do not persist batch padding or eight separate head matrices.
FP32 is the default to avoid introducing a target-quantization confound. A later
FP16/compressed format requires an explicit precision contract and measured
error checks; it must not appear silently because storage was tight.

For one bank, attention payload size alone is approximately:

```text
4 * sum_over_training_jets(n_jet^2) bytes
```

Use actual authenticated counts, not `number_of_jets * mean(n)^2`: multiplicity
variance matters. For illustration only, if every one of 299,234 jets had exactly
50 particles, one FP32 matrix bank would occupy about 2.79 GiB, before metadata,
probabilities, temporary publication, and reserve. This is **not** a measurement
of the real dataset. There are three attention-export checkpoints in this graph.

Bound shard read caching, prefetch, temporary storage, and host/GPU memory.
Authenticate once per admitted artifact/shard lifecycle with explicit immutable
ownership; do not hash a multi-gigabyte bank again on every minibatch. Conversely,
mere file existence is never evidence of integrity. Never delete parent inputs
or completed outputs automatically to make a quota check pass.

The student performs one forward and backward through one ParT. Capture its
attention from that same forward; a second full stochastic student forward
would give different hidden states/dropout and unnecessarily double work.
The frozen teacher does not remain on GPU during student training when targets
are served from the bank.

Expected advantage: no second trainable encoder, no four fusion cross-attention
injections, and no fusion-specific synchronous CPU activation offload. But
student attention extraction still has quadratic cost, can change the attention
backend used, and adds target I/O. **No speedup factor is claimed in advance.**
Benchmark extraction-enabled and ordinary paths on the intended SPORC A100.

## 8. Alignment and leakage guardrails

Before constructing any loss, verify:

1. Identical TRAIN jet identity and label joins for target and student caches.
2. Identical real token counts and persistent slot order along every D edge.
3. Matching assignment and view-convention hashes agree with the parent study.
4. Model-visible inputs contain only the registered features, vectors, and mask.
5. Probability and attention targets come from the same selected parent model.
6. An attention export is bound to its layer, head reduction, masks, precision,
   numerical convention, and original model/environment bytes.

No row identity, source file, match validity, match index, coordinate label, or
offline particle key is a deployable feature. These can exist in sidecar
authentication records or training loss alignment code only.

The final D000 inference path must run without the teacher, attention targets,
matching files, offline views, or target-generation code. Intermediate U100,
D066, and D033 models still require offline-derived inputs and are not HLT-only
deployment results.

## 9. Validation, comparisons, and interpretation

Primary endpoint: full-validation macro OVR AUC difference
`ATTN_D000 - LOGIT_D000`. Also report accuracy, cross-entropy, per-class QCD
rejection at 50% signal efficiency, parameter count, training GPU-hours, and
inference latency. Retain explicit censored rejection values when no background
passes; do not substitute an arbitrary finite ceiling.

Report every D rung for both arms, including weak ones. Use existing matched
M0HLT, pure OFFLINE, direct-KD, and fusion endpoints as clearly labelled
authenticated references when available. A fused model and a single-ParT model
are not capacity- or compute-matched. The primary two arms are capacity-matched,
but attention-target export and auxiliary training increase total compute.

Recovery percentages use the same population's matched M0HLT and pure-OFFLINE
references. Always display raw metrics and their reference gap; ratios can be
unstable when that gap is small. Never mix original-50k metrics with full-74895
metrics in one recovery calculation.

Paired source-file bootstrap intervals on saved validation predictions can be
reported as development diagnostics, with the bootstrap protocol frozen before
analysis. They do not cover training-seed uncertainty or selection/tuning on
this validation set. A single-seed gain is not independent confirmation.

Interpret outcomes carefully:

- Better final D000 performance supports the usefulness of the added supervision
  under this recipe, not a universal explanation of all earlier KD failures.
- Better attention agreement without classification gains means matching this
  surface was insufficient; it is not grounds to suppress the result.
- Worse performance may reflect conflicting views, imperfect assignments,
  loss weighting, or insufficient optimization. These are hypotheses for later
  controlled studies, not conclusions proven by one run.
- Attention weights are model operations, not guaranteed physical explanations
  or truth-level particle associations.

No independent test is available in this trial. Do not relabel the original
50k validation subset or consumed historical test as a fresh holdout.

## 10. Technical acceptance, without reinstating the one-step pilot

Skipping scientific screening does not waive implementation checks. Required
before submitting the complete ladder:

- **Input/slot replay:** all D views have the same real support and known slot
  transport; native D000 fingerprints remain unchanged.
- **Attention extraction:** compare the exposed probability surface against the
  installed Weaver computation, including Q/K semantics and pair bias.
- **Native parity:** with auxiliary weight zero, check logits, parameter/input
  gradients, optimizer updates, masks, and relevant buffers against ordinary
  ParT, in FP32 and separately in production BF16 within declared tolerances.
- **Loss tests:** identical target/student attention gives zero KL; a deliberately
  mismatched nonuniform matrix changes it; pad masking, diagonal behavior,
  head permutation, and short/one-token cases behave as defined.
- **Transport tests:** jointly permuting tokens and both matrix axes preserves
  the loss; an untransported asymmetric target fails the identity check or changes
  the loss. Do not rely only on equal tensor shapes.
- **Gradient isolation:** attention loss produces finite student gradients in the
  intended parameters; teacher parameters and target banks receive none.
- **Artifact tests:** altered labels, counts, order, parent checkpoint, layer,
  precision, masks, or payload hashes fail closed; interrupted shards cannot
  become accepted complete banks.
- **End-to-end miniature:** exercise both three-rung graphs, selected-checkpoint
  export, bank reuse, and final HLT-only readback with production workers.
- **New resource evidence:** measure representative and longest-support TRAIN
  backward batches, validation inference, bank export/storage/I/O, peak memory,
  full-epoch projections, and debug walltime headroom on the intended site.

A short technical miniature is non-scientific: it does not select lambda, the
layer, a model, or whether a promising-looking method proceeds. Old fusion
resource admission cannot certify the new attention-readout path. No resource
failure authorizes silently shortening the schedule or dropping particles.

## 11. Implementation and provenance boundaries

This is a new campaign, not an edit to immutable fixed30 v6 specifications.
The old fixed30 no-checkpoint-import rule remains true for that campaign; this
new proposal explicitly permits importing **one authenticated scientific U100
anchor and matched reporting references** into a separate namespace. It does
not rewrite the old contracts or add attention targets to old logit-only banks.

The implementation uses distinct versioned contracts for the experiment
recipe, imported anchor, attention surface, TRAIN target shards/manifest,
technical acceptance, execution spec, task receipts, predictions, and results.
The registered namespace is `LUKA_D_ATTENTION_*/v1`; array payloads are bound by
those manifests and receipts, not relabelled old probability banks.
Every consumer must validate content/parent/source/population hashes and atomic
publication. Preserve historical-test provenance flags from ALLJETS.

Suggested logical execution sequence:

1. Authenticate the parent study, common U100, selected split, views, and quota.
2. Run installed-model parity and real technical/resource acceptance.
3. Publish the common U100 TRAIN target bank.
4. Train both D066 students; select and export each arm's own teacher.
5. Train both D033 students; select and export each arm's own teacher.
6. Train both D000 students and publish native-HLT validation predictions.
7. Aggregate all models, matched contrasts, costs, and limitations.

Bind exact dependencies and immutable inputs. No broad cancellation, controller
change, environment upgrade, auto-retry, or modification of the currently running
fusion campaign is implied. The runbook separates the bounded technical gate
from the measured science plan; no science job can bypass successful acceptance.

Useful existing code to inspect, not silently reinterpret:

- [Persistent views](../src/hlt_classification/cms_proxy_ladder/views.py) and
  [durable Luka caches](../src/hlt_classification/luka_fullsim/regular_cache.py).
- [17-feature input adapter](../src/hlt_classification/cms_proxy_ladder/inputs.py)
  and [tracking transform wrapper](../src/hlt_classification/literature_context/transform.py).
- [Single-ParT model and logit loss](../src/hlt_classification/jetclass2_delphes/model.py)
  and [current training/selection kernel](../src/hlt_classification/jetclass2_delphes/runner.py).
- [Fixed30 recipe](../src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py)
  and [immutable fixed30 contract](contracts/LUKA_ALLJETS_30E.md).

An older [HCWDL attention re-optimization contract](contracts/HCWDL_TRI100_FOUR_SPINE_ATTENTION_REOPTIMIZATION.md)
uses a different dataset, all-block residual-delta Gram targets, a live teacher,
staged parameter freezing, and forbids durable dense targets. This proposal is
**not** that campaign: it uses one late-block attention-probability target,
ordinary all-parameter training, and explicitly declared cached TRAIN targets.
Its policies cannot be imported under the older artifact names.

Generic repository examples with 10 classes, trimming, or CE-first checkpoint
selection are also not this trial's recipe. The Luka/fixed30 plan and explicit
new attention proposal govern these campaign-specific choices.

## 12. Research basis and limits

[Particle Transformer for Jet Tagging (Qu, Li, Qian)](https://arxiv.org/abs/2202.03772)
introduces particle-pair interaction features into attention. It motivates
examining a relational surface rather than only class outputs; it does not
establish that offline-to-HLT attention transfer improves this dataset.

[MiniLM (Wang et al.)](https://arxiv.org/abs/2002.10957)
studies transformer distillation using self-attention distributions and value
relations, including late-layer supervision. It is an NLP result and a source
of methodological motivation, not evidence for this particle-level adaptation.

[MiniLMv2 (Wang et al.)](https://aclanthology.org/2021.findings-acl.188/)
develops multi-head relation distillation. The head-averaged, geometrically
biased, cross-view loss specified here is our proposed simplification, not a
claim to reproduce MiniLM or MiniLMv2 exactly.

The hypothesis to test is narrow: **on this fixed persistent-slot D ladder,
does adding aligned late-block attention supervision improve the final
single-HLT model over the otherwise matched ordinary-KD ladder?**

## 13. Decision summary

- Proceed with the full `U100 -> D066 -> D033 -> D000` design, not the one-step pilot.
- Use single-view ParT throughout, with ordinary KD plus late-block attention KD.
- Include a matched ordinary-KD three-step control from the same U100 anchor.
- Keep the trial ALLJETS split, true batch128, fixed30 schedule, and HLT-only endpoint.
- Treat lambda 0.1 and head-averaged final-block supervision as frozen first-run
  settings, not tuned or scientifically validated results.
- Preserve the existing campaigns. Implementation and new contracts are present;
  real SPORC acceptance, reviewed pushed source/plan, and explicit live submission
  remain execution prerequisites, not accomplishments of the local tests.

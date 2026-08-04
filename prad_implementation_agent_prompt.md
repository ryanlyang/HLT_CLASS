# Implementation Task: Privileged Relational Attention for HLT Jet Classification

## Role

Act as the lead ML research engineer for this project. Your job is to implement, train, evaluate, and compare a family of privileged-training methods whose only goal is to improve final HLT jet-tagging performance.

You are not being asked to produce a superficial prototype. Build a reliable research implementation that can answer whether offline reconstruction can teach an HLT-only Particle Transformer better particle relationships and whether those relationships improve attention and final tagging.

Use the attached design document, `privileged_relational_attention_proposal.md`, as the scientific specification. This prompt defines the execution requirements, experiment order, required controls, and deliverables.

Do not silently simplify the central method. Any necessary adaptation to the existing repository or dataset must be documented clearly.

---

# 1. Primary objective

The deployed classifier must map only HLT inputs to jet classes:

\[
\hat y = S(X^{\mathrm{HLT}}).
\]

Offline information may be used during training, but no offline feature, offline object, offline target, offline match, or offline teacher output may be required at inference.

The main method to implement is **Privileged Relational Attention Distillation**, abbreviated **PRAD**:

1. Train an offline Particle Transformer teacher.
2. Make the teacher learn a class-relevant relation vector for every particle pair.
3. Use that relation vector to create an additive attention bias inside the teacher.
4. Train an HLT-only relation predictor to reproduce the teacher relation for matched HLT particle pairs.
5. Insert the HLT-predicted relation into later HLT Particle Transformer attention blocks through zero-initialized learnable gates.
6. Optimize the final HLT classification objective together with relation distillation and selected offline auxiliary targets.

The success criterion is not relation-prediction accuracy by itself. The success criterion is improved HLT tagging on the untouched test set.

---

# 2. First inspect the repository and data

Before modifying code:

1. Identify the current strongest HLT Particle Transformer implementation and its training entry point.
2. Identify the current offline input representation.
3. Identify how HLT and offline jets are paired.
4. Determine whether HLT particles retain direct identities or indices linking them to offline particles.
5. Identify all available HLT features, offline features, labels, masks, event IDs, particle IDs, track IDs, vertex assignments, and jet metadata.
6. Identify the existing normalization, truncation, particle ordering, class balancing, optimizer, scheduler, checkpointing, metric, and evaluation conventions.
7. Run a small data audit and record:
   - total paired jets;
   - class counts;
   - number of particles per jet;
   - missing-value rates;
   - HLT-to-offline particle match coverage;
   - matched transverse-momentum coverage;
   - availability of offline vertex assignments;
   - whether multiple jets can come from one physical event.

Write the audit to `reports/data_audit.md` before full training.

Use the existing repository conventions wherever they do not conflict with this prompt. Do not replace a working ParT implementation with a new generic transformer.

---

# 3. Non-negotiable dataset split

Create one deterministic split and use it for every teacher, student, baseline, and ablation:

- **500,000 training jets**
- **150,000 validation jets**
- **500,000 test jets**

Total required sample size:

\[
1{,}150{,}000 \text{ paired jets}.
\]

## Split requirements

1. Split by physical event, not merely by jet row, whenever event IDs exist.
2. All HLT and offline representations of the same jet must remain together.
3. No physical event may appear in more than one split.
4. Stratify by class so each split follows the overall class proportions as closely as possible.
5. Use deterministic split seed `1337`.
6. Fit normalization statistics using the 500,000 training jets only.
7. Determine class weights, sampling weights, or balancing factors using the training split only.
8. Save exact split manifests containing stable sample or event identifiers:
   - `splits/train_500k.txt`
   - `splits/val_150k.txt`
   - `splits/test_500k.txt`
9. Save a split summary with counts by class and checks proving zero overlap.
10. Compute and save a checksum for every manifest.
11. Never use the test set for hyperparameter selection, early stopping, architecture selection, threshold selection, or debugging.

If fewer than 1,150,000 valid paired jets exist after required integrity filtering, stop and report the exact shortfall. Do not silently change the requested split sizes.

---

# 4. Baseline reproduction

Before implementing PRAD, reproduce the strongest existing HLT ParT baseline on the fixed split.

The baseline must use:

- the existing HLT features;
- the existing particle selection;
- the existing particle ordering;
- the existing truncation and padding;
- the existing ParT pairwise interaction branch;
- no offline information;
- class supervision only.

Save:

- model configuration;
- parameter count;
- best validation checkpoint;
- training curves;
- validation predictions;
- test predictions;
- per-class tagging metrics;
- inference latency;
- peak inference memory.

This baseline is the comparison point for every later experiment.

The PRAD student must be initialized from the best baseline checkpoint unless a specific ablation requires training from scratch.

---

# 5. Main architecture: full PRAD

## 5.1 Preserve the baseline ParT

Let the baseline contain \(L\) particle-attention blocks and \(H\) attention heads.

Preserve all existing baseline dimensions and modules.

The default design assumes \(L=8\), but the implementation must read these values from configuration rather than hard-code them.

Use the first two particle-attention blocks as a contextualization stem. If the baseline has fewer than four particle-attention blocks, report this as an architectural blocker rather than inventing a new insertion scheme.

The privileged relation is injected into every particle-attention block after the first two.

## 5.2 Offline teacher

The offline teacher receives only offline particle inputs and the jet class.

Architecture:

1. Offline particle embedding.
2. Two ordinary ParT particle-attention blocks using the standard offline pairwise interaction bias.
3. A contextual pair-relation module.
4. The remaining particle-attention blocks, each receiving both:
   - the standard offline ParT interaction bias;
   - the learned offline relation bias.
5. The usual class-attention blocks and classifier.

The teacher relation module must affect the teacher class prediction. It must not be an auxiliary head disconnected from attention.

## 5.3 HLT student

The HLT student receives only HLT particle inputs.

Architecture:

1. The existing HLT particle embedding.
2. The first two existing HLT ParT particle-attention blocks.
3. An HLT contextual pair-relation predictor.
4. The remaining existing particle-attention blocks, each receiving:
   - the standard HLT ParT interaction bias;
   - the HLT-predicted privileged relation bias.
5. The existing class-attention blocks and classifier.

At deployment, the forward function must accept HLT inputs only.

---

# 6. Contextual pair-relation module

For contextual token embeddings \(h_i\) and \(h_j\), construct a symmetric pair representation:

\[
s_{ij}
=
\left[
h_i+h_j,\;
|h_i-h_j|,\;
h_i\odot h_j,\;
q_{ij}
\right].
\]

Here \(q_{ij}\) must include the standard ParT pair quantities:

\[
\left(
\log \Delta R_{ij},
\log k_{T,ij},
\log z_{ij},
\log m_{ij}^{2}
\right).
\]

Also append symmetric pair combinations of available per-particle scalar features:

\[
\left[
x_i+x_j,\;
|x_i-x_j|,\;
x_i\odot x_j
\right].
\]

Embed categorical particle features before forming pair combinations.

Use a three-layer relation MLP:

- input dimension inferred from the constructed pair representation;
- hidden dimension 256;
- hidden dimension 128;
- output relation bottleneck dimension 16;
- GELU activations;
- dropout 0.1 after the first two layers;
- LayerNorm on the 16-dimensional output.

The output is:

\[
r_{ij}\in\mathbb R^{16}.
\]

The relation must be symmetric:

\[
r_{ij}=r_{ji}.
\]

Add a unit test that checks symmetry numerically.

---

# 7. Relation-to-attention projection

Project each relation vector to one scalar per attention head:

\[
\widetilde B_{ij}
=
3\tanh(W_Br_{ij}+b_B),
\qquad
\widetilde B_{ij}\in\mathbb R^H.
\]

Center the bias across valid key particles for every query and head:

\[
B_{ij,h}
=
\widetilde B_{ij,h}
-
\frac{
\sum_{k\in\mathcal V}
\widetilde B_{ik,h}
}{
|\mathcal V|
}.
\]

Respect padding masks when centering.

For each later attention block \(\ell\) and head \(h\), use:

\[
A_{ij}^{(\ell,h)}
=
\frac{Q_i^{(\ell,h)}K_j^{(\ell,h)\top}}{\sqrt{d_h}}
+
U_{ij,h}
+
\alpha_{\ell,h}B_{ij,h}.
\]

Here \(U_{ij,h}\) is the existing ParT interaction bias.

Parameterize the gate as:

\[
\alpha_{\ell,h}=\tanh(a_{\ell,h}).
\]

Initialize every \(a_{\ell,h}=0\).

With all gates at zero, the modified student must reproduce the baseline logits to numerical precision. Add an automated test requiring maximum absolute logit difference below \(10^{-6}\) in evaluation mode for identical weights and inputs.

Compute one relation matrix per jet and reuse it across all later blocks. Use separate gates by layer and head.

---

# 8. Offline relation targets

## 8.1 Learned relation target

The primary target is the frozen offline teacher relation bottleneck:

\[
r_{ij}^{T}.
\]

Also distill the teacher's explicit centered attention-bias output:

\[
B_{ij}^{T}.
\]

Do not distill the full post-softmax attention matrix.

## 8.2 Multiscale offline branch targets

Using offline constituents, recluster each jet with the Cambridge/Aachen algorithm using the original jet radius.

Create exclusive subjet assignments for:

\[
K\in\{2,3,4\}.
\]

For every offline particle pair, define three binary labels indicating whether both particles belong to the same exclusive \(K\)-subjet.

These targets must always be implemented because they can be derived from offline four-vectors.

## 8.3 Common-vertex target

For charged offline tracks with valid reconstructed vertex assignments, define a binary target indicating whether both tracks belong to the same reconstructed vertex.

Mask this target unless both tracks have valid vertex assignments.

Do not treat a missing vertex assignment as a negative label.

If the dataset has no usable offline vertex information, set the vertex-loss coefficient to zero and record this explicitly in the report.

---

# 9. HLT-to-offline particle association

Prefer direct stored particle identities or reconstruction associations.

If direct associations exist, use them and document their meaning.

If they do not exist, implement one-to-one matching separately for charged and neutral particles using Hungarian assignment.

## Charged-particle candidate requirements

- equal charge;
- compatible particle category;
- \(\Delta R < 0.03\);
- \(0.5 < p_T^{\mathrm{HLT}}/p_T^{\mathrm{OFF}} < 2.0\).

Cost:

\[
c_{ia}
=
\left(\frac{\Delta R_{ia}}{0.01}\right)^2
+
\left(
\frac{\log(p_{T,i}^{\mathrm{HLT}}/p_{T,a}^{\mathrm{OFF}})}{0.2}
\right)^2.
\]

Reject assigned pairs with \(c_{ia}>9\).

## Neutral-particle candidate requirements

- compatible neutral category;
- \(\Delta R < 0.06\);
- \(0.25 < E^{\mathrm{HLT}}/E^{\mathrm{OFF}} < 4.0\).

Cost:

\[
c_{ia}
=
\left(\frac{\Delta R_{ia}}{0.02}\right)^2
+
\left(
\frac{\log(E_i^{\mathrm{HLT}}/E_a^{\mathrm{OFF}})}{0.35}
\right)^2.
\]

Reject assigned pairs with \(c_{ia}>9\).

Unmatched particles remain valid classifier inputs but are masked from offline relation losses.

Create a pair supervision mask:

\[
M_{ij}
=
\mathbb 1[\pi(i)\neq\varnothing]
\mathbb 1[\pi(j)\neq\varnothing],
\qquad
M_{ii}=0.
\]

Report matching coverage by class, particle type, jet \(p_T\), and particle multiplicity.

---

# 10. Teacher training

Train the offline teacher first.

Use:

\[
\mathcal L_T
=
\mathcal L_{\mathrm{class}}
+
0.2\mathcal L_{\mathrm{tree}}
+
0.2\mathcal L_{\mathrm{vertex}}.
\]

Each loss must be normalized by its own number of valid examples before weighting.

Use class-balanced binary cross entropy for the multiscale branch and vertex targets. Calculate positive weights from the training split only.

Select the teacher checkpoint using the primary validation tagging score, not relation accuracy.

After selection, freeze every teacher parameter.

Save the teacher's:

- relation bottleneck outputs;
- relation-bias outputs;
- class logits;
- confidence on the true class;
- validation and test tagging metrics.

Caching teacher outputs is allowed if storage permits, but the cache must be indexed by the immutable split sample identifier and checked against the manifest checksum.

---

# 11. Student losses

Use four student loss components.

## 11.1 Hard class loss

\[
\mathcal L_{\mathrm{hard}}
=
\mathrm{CE}(y,p_S).
\]

Coefficient:

\[
1.0.
\]

## 11.2 Relation distillation loss

Use Smooth L1 for both the relation bottleneck and attention bias:

\[
\mathcal L_{\mathrm{rel}}
=
0.5\mathcal L_{\mathrm{bottleneck}}
+
0.5\mathcal L_{\mathrm{bias}}.
\]

Apply the valid pair mask.

Use the teacher outputs with stop-gradient.

Use reliability weighting:

\[
w_{\mathrm{jet}}
=
\operatorname{clip}
\left(
p_T(y_{\mathrm{true}}),
0.1,
1.0
\right),
\]

\[
w_{\mathrm{pair}}
=
\operatorname{clip}
\left(
1+
2\cdot \operatorname{mean}_h |B_{ij,h}^{T}|,
1,
5
\right),
\]

\[
w_{ij}=w_{\mathrm{jet}}w_{\mathrm{pair}}.
\]

Default relation coefficient:

\[
\lambda_{\mathrm{rel}}=1.0.
\]

## 11.3 Semantic pair loss

Train the student relation bottleneck to predict:

- common exclusive 2-subjet membership;
- common exclusive 3-subjet membership;
- common exclusive 4-subjet membership;
- common reconstructed vertex where available.

Default coefficient:

\[
\lambda_{\mathrm{sem}}=0.2.
\]

## 11.4 Teacher class-logit distillation

Use temperature \(T=2\):

\[
\mathcal L_{\mathrm{KD}}
=
T^2
\mathrm{KL}
\left[
\operatorname{softmax}(z_T/T)
\|
\operatorname{softmax}(z_S/T)
\right].
\]

Default coefficient:

\[
\lambda_{\mathrm{KD}}=0.5.
\]

## Full student loss

\[
\mathcal L_S
=
\mathcal L_{\mathrm{hard}}
+
1.0\mathcal L_{\mathrm{rel}}
+
0.2\mathcal L_{\mathrm{sem}}
+
0.5\mathcal L_{\mathrm{KD}}.
\]

Do not allow an unnormalized pairwise loss to dominate because it has many more terms than the class loss.

---

# 12. Student training schedule

## Stage A: relation warm-up

Initialize the HLT student from the best HLT baseline checkpoint.

Copy from the teacher:

- the relation-to-bias projection \(W_B,b_B\);
- the semantic relation heads.

For five epochs:

- freeze the HLT embedding;
- freeze all baseline transformer blocks;
- freeze class-attention blocks and classifier;
- freeze copied bias projection and semantic heads;
- hold all relation gates at zero;
- train only the HLT relation MLP.

Loss:

\[
\mathcal L_{\mathrm{rel}}+0.2\mathcal L_{\mathrm{sem}}.
\]

Learning rate for the relation MLP:

\[
3\times10^{-4}.
\]

## Stage B: controlled integration

For the next five epochs:

- unfreeze the relation gates;
- unfreeze the final half of particle-attention blocks;
- unfreeze class-attention blocks and classifier;
- keep the earlier particle-attention blocks frozen;
- use the full loss.

Learning rates:

- relation module and gates: \(3\times10^{-4}\);
- unfrozen pretrained model weights: \(3\times10^{-5}\).

Linearly ramp \(\lambda_{\mathrm{KD}}\) from 0 to 0.5 over these five epochs.

## Stage C: end-to-end fine-tuning

Unfreeze the complete student.

Use:

- relation module and gates learning rate: \(1\times10^{-4}\);
- pretrained baseline weights learning rate: \(1\times10^{-5}\);
- AdamW;
- weight decay \(1\times10^{-5}\);
- cosine learning-rate decay;
- gradient norm clipping at 1.0;
- mixed precision if already supported;
- early stopping with patience 10 based on the primary validation tagging score;
- maximum 50 epochs.

Save both the best validation checkpoint and the final checkpoint.

---

# 13. Oracle test before expensive PRAD training

Implement an explicitly non-deployable oracle model in which the HLT classifier receives the frozen teacher's true offline relation bias for matched pairs.

The oracle must still use HLT particle tokens and HLT class inputs. Only the privileged relation bias comes from the offline teacher.

Use the same additive bias interface and zero-initialized gates.

The oracle answers:

> Would this offline relation improve HLT classification if it were known perfectly?

Run this immediately after the teacher and baseline are available.

Decision guidance:

- less than 2 percent relative improvement in the primary validation score: the relation definition has little headroom;
- 2 to 5 percent: continue only as a limited pilot;
- more than 5 percent: proceed with full PRAD;
- more than 10 percent: prioritize relation-predictor quality because the channel has substantial potential.

Do not conceal a weak oracle result. It is one of the most important scientific outcomes.

---

# 14. Required experiment matrix

Implement every experiment through configuration, not by maintaining separate copied training scripts.

## Core experiments

| ID | Configuration | Purpose |
|---|---|---|
| E0 | HLT ParT baseline | Establish current performance |
| E1 | Offline teacher | Measure offline ceiling |
| E2 | Oracle offline relation attention | Measure relation headroom |
| E3 | Capacity control | Test extra parameters without offline supervision |
| E4 | Logit KD only | Measure ordinary teacher distillation |
| E5 | Semantic auxiliary only | Test offline pair labels without attention injection |
| E6 | Learned relation auxiliary only | Distill relations but keep privileged gates at zero |
| E7 | Semantic relation attention | Use tree and vertex relation prediction in attention without learned teacher relation |
| E8 | Full PRAD without logit KD | Isolate relational attention mechanism |
| E9 | Full PRAD | Maximize final tagging performance |
| E10 | Shuffled relation control | Test whether correctly paired offline structure matters |

## Exact definitions

### E3: capacity control

Use the same HLT relation module, relation-to-bias projection, and gates as full PRAD.

Train with hard class loss only.

Do not use any offline target or teacher output.

### E4: logit KD only

Use the unmodified baseline HLT ParT.

Train with hard class loss plus class-logit KD.

Do not add a relation module.

### E5: semantic auxiliary only

Predict multiscale branch and vertex labels from a contextual pair module.

Do not insert its outputs into attention.

### E6: learned relation auxiliary only

Use relation distillation and semantic pair losses.

Fix every privileged attention gate at zero.

### E7: semantic relation attention

Use only the semantic branch and vertex targets to train the relation module.

Insert its learned relation bias into attention.

Do not match the learned teacher relation bottleneck or teacher relation bias.

### E8: full PRAD without logit KD

Use hard class loss, learned relation distillation, semantic pair losses, and relation-biased attention.

Set \(\lambda_{\mathrm{KD}}=0\).

### E9: full PRAD

Use the complete method and default loss weights.

### E10: shuffled relation control

Shuffle teacher relation targets across jets within each training batch.

Do not preserve class when shuffling.

Keep all other settings identical to E8.

This experiment should not beat the capacity control if the method is using meaningful paired offline structure.

---

# 15. Alternative PRAD variants

The implementation must expose configuration switches for the following variants. Run them after the core experiments, prioritizing them when the oracle shows useful headroom.

## V1: bias-only distillation

Distill only \(B_{ij}^{T}\).

Do not match the 16-dimensional relation bottleneck.

Purpose:

- determine whether the teacher bottleneck is unnecessary or difficult to align.

## V2: bottleneck-only distillation

Distill only \(r_{ij}^{T}\).

Generate the student's bias from the copied teacher projection.

Purpose:

- determine whether a shared latent relation space is more stable than direct bias regression.

## V3: raw-pair predictor

Calculate the relation directly from raw HLT particle and pair features before any transformer contextualization.

Inject it into all particle-attention blocks.

Purpose:

- test a cheaper PairEmbed-like version;
- determine whether jet context is necessary.

## V4: contextual predictor after four blocks

Move the relation module from after block 2 to after block 4 and inject it into the remaining blocks.

Purpose:

- trade fewer affected layers for a more contextual relation estimate.

## V5: semantic-only learned relation

Use multiscale branch and common-vertex labels as the only relation supervision.

No teacher relation distillation and no logit KD.

Purpose:

- determine whether interpretable offline structure transfers better than a free teacher representation.

## V6: direct teacher PairEmbed target

If the existing offline ParT already has a standard PairEmbed or interaction-bias branch, distill its output directly into an HLT PairEmbed network and add it to attention.

Purpose:

- provide a simpler and lower-risk version of the method.

## V7: relation dimension

Test relation bottleneck dimensions:

\[
d_r\in\{8,16,32\}.
\]

Use 16 as the default.

## V8: attention injection depth

Test:

- all particle-attention blocks;
- blocks after the first two;
- final half of particle-attention blocks only.

Keep all other settings fixed.

## V9: gate structure

Compare:

- one scalar gate per layer and head;
- one scalar gate per layer;
- one global scalar gate.

The default is one gate per layer and head.

## V10: remove standard ParT pair bias

Replace:

\[
U^{\mathrm{HLT}}+\alpha B^{\mathrm{priv}}
\]

with:

\[
\alpha B^{\mathrm{priv}}.
\]

This is a diagnostic only. Do not use it as the main model unless validation results clearly support it.

---

# 16. Experiment order and resource discipline

Use this order:

1. Data audit and split construction.
2. Baseline reproduction.
3. Offline teacher.
4. Oracle relation model.
5. Capacity control.
6. Logit KD only.
7. Semantic auxiliary only.
8. Learned relation auxiliary only.
9. Full PRAD without KD.
10. Full PRAD.
11. Shuffled relation control.
12. Highest-priority alternative variants.

Low-cost pilot runs on a smaller training subset are allowed for debugging only.

Every result used for scientific comparison must be retrained on the full 500,000-jet training split and selected using the full 150,000-jet validation split.

Use one seed for broad screening. Then retrain:

- the baseline;
- full PRAD;
- full PRAD without KD;
- the best alternative variant;
- any other model within 1 percent of the best validation score

with five seeds:

\[
\{11,22,33,44,55\}.
\]

Evaluate the untouched 500,000-jet test set only after model and hyperparameter selection is complete.

---

# 17. Evaluation metrics

## Primary metric

For each class, calculate one-vs-rest background rejection at 50 percent signal efficiency:

\[
\mathrm{Rej}_c(50\%)
=
\frac{1}{\epsilon_{\mathrm{bkg},c}}
\quad
\text{at}
\quad
\epsilon_{\mathrm{sig},c}=0.5.
\]

Use linear interpolation between neighboring ROC points when necessary.

Define the scalar model-selection score:

\[
S_{\mathrm{macro}}
=
\frac{1}{C}
\sum_{c=1}^{C}
\log \mathrm{Rej}_c(50\%).
\]

Use \(S_{\mathrm{macro}}\) for validation checkpoint selection and architecture comparison.

## Required secondary metrics

Report:

- per-class AUC;
- macro AUC;
- multiclass accuracy;
- confusion matrix;
- expected calibration error;
- loss curves;
- parameter count;
- training time;
- inference throughput;
- per-jet inference latency;
- peak inference memory;
- relation bottleneck loss;
- relation-bias loss;
- multiscale branch-label AUC;
- vertex-label AUC;
- learned gate values by layer and head;
- matched-particle and matched-pair coverage.

## Stratified evaluation

Report the primary metric versus:

- jet \(p_T\);
- jet \(\eta\);
- HLT particle multiplicity;
- matched-particle fraction;
- matched-\(p_T\) fraction;
- offline teacher confidence;
- class.

---

# 18. Statistical reporting

For final five-seed models, report:

- mean;
- standard deviation;
- paired change from the baseline for the same seed;
- relative percentage change;
- 95 percent bootstrap confidence interval using test predictions.

A claimed improvement must be larger than both:

- seed-to-seed variation;
- finite-test-sample uncertainty.

Do not describe a numerically tiny or statistically unstable change as a meaningful gain.

---

# 19. Required tests and safeguards

Add automated tests for:

1. Exact split counts: 500,000, 150,000, and 500,000.
2. Zero identifier overlap among splits.
3. Zero event overlap among splits where event IDs exist.
4. HLT and offline jet labels agree for paired samples.
5. Pair mask excludes unmatched particles and self-pairs.
6. Relation output is symmetric.
7. Padding does not affect centered relation bias.
8. Relation bias remains bounded.
9. All-zero gates reproduce baseline logits within \(10^{-6}\).
10. A nonzero gate allows gradients from class loss to reach the relation module.
11. Frozen teacher parameters receive no gradients during student training.
12. Student inference succeeds when all offline fields are removed.
13. Shuffling offline targets changes relation supervision but not hard class labels.
14. Test-set metrics cannot be invoked from the training or tuning entry point without an explicit final-evaluation flag.
15. Checkpoint resumption restores optimizer, scheduler, scaler, epoch, random state, and configuration.

---

# 20. Code quality requirements

The implementation must be:

- configuration driven;
- compatible with the existing repository structure;
- reproducible;
- restartable;
- compatible with mixed precision if the baseline supports it;
- compatible with distributed training if the baseline supports it;
- free of hard-coded local paths;
- explicit about tensor shapes and masks;
- documented with type hints for new public functions;
- covered by tests for the new scientific behavior.

Create separate reusable modules for:

- split construction;
- particle matching;
- offline structural target construction;
- contextual relation prediction;
- relation-to-attention projection;
- gated attention injection;
- teacher-output caching;
- relation losses;
- experiment configuration;
- evaluation and plotting.

Do not duplicate the entire ParT implementation to create variants. Add clean extension points.

---

# 21. Experiment tracking and artifacts

For every run, save:

- run ID;
- experiment ID;
- git commit;
- full resolved configuration;
- split-manifest checksums;
- random seed;
- parameter count;
- best epoch;
- best validation score;
- checkpoint path;
- training curves;
- validation metrics;
- test metrics when final evaluation is authorized;
- prediction file keyed by stable sample ID.

Produce:

- `results/all_runs.csv`
- `results/final_comparison.csv`
- `reports/final_report.md`
- `plots/` containing ROC curves, rejection comparisons, gate heatmaps, relation-quality plots, matching diagnostics, and stratified performance plots.

---

# 22. Final report requirements

The final report must answer these questions directly:

1. How much better is the offline teacher than the HLT baseline?
2. Does the oracle offline relation improve an HLT classifier?
3. Can HLT inputs predict the useful offline relation?
4. Does relation supervision help when it is only auxiliary?
5. Does feeding the predicted relation into attention add improvement beyond auxiliary supervision?
6. Does ordinary class-logit KD explain the gain?
7. Does an equal-capacity model without offline supervision explain the gain?
8. Does shuffled offline relation supervision remove the gain?
9. Which relation target works best:
   - learned teacher relation;
   - teacher bias only;
   - multiscale substructure;
   - vertex structure;
   - direct teacher PairEmbed?
10. Which attention-injection depth works best?
11. How much of the oracle gain does the deployable student recover?
12. Where does the method help or fail across classes and kinematic regions?
13. What is the inference cost relative to baseline?
14. Is the final improvement statistically reliable?

Include one explicit recommendation:

- adopt full PRAD;
- adopt a simpler variant;
- use offline supervision only as an auxiliary loss;
- use ordinary KD;
- or abandon this relation target.

Base the recommendation on final tagging performance, robustness, and inference cost, not on how interesting the architecture appears.

---

# 23. Acceptance criteria

The task is complete only when all of the following are true:

1. The deterministic 500k/150k/500k split exists and passes leakage tests.
2. The existing HLT baseline has been reproduced.
3. The offline teacher has been trained and frozen.
4. The oracle relation experiment has been completed.
5. The main PRAD architecture has been implemented.
6. The student uses only HLT inputs at inference.
7. The core causal ablations have been run.
8. At least the highest-priority alternative variants have been evaluated when the oracle shows headroom.
9. Finalists have been evaluated across five seeds.
10. The untouched 500,000-jet test set has been evaluated only after selection.
11. Code, configs, commands, checkpoints, metrics, and the final report are provided.
12. The final conclusion distinguishes real privileged-information gains from capacity, KD, regularization, and leakage.

---

# 24. Working style

Proceed independently and make reasonable repository-specific decisions when the answer can be determined from the code or data.

Do not repeatedly ask for approval for routine engineering choices.

Do not hide blockers or substitute an easier objective without saying so.

When a genuine blocker prevents correct implementation, report:

1. the exact blocker;
2. the evidence;
3. the scientific consequence;
4. the least invasive proposed resolution.

At each major milestone, provide a concise update containing:

- what was completed;
- what was verified;
- current quantitative results;
- the next experiment;
- any deviation from this specification.

The project goal is a trustworthy answer to whether privileged offline relational attention can materially improve HLT tagging. Optimize for that answer.

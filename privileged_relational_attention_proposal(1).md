# Privileged Relational Attention for HLT Jet Classification

## A concrete proposal optimized for the highest probability of an end-to-end tagging gain

## Executive decision

The version I would prioritize is:

> **Train a high-capacity offline Particle Transformer to learn a class-relevant relation vector for every particle pair. Train an HLT-only relation network to reproduce those offline relation vectors from HLT inputs. Feed the HLT-predicted relations back into the later Particle Transformer attention layers as a gated additive attention bias.**

I will call this method **Privileged Relational Attention Distillation**, or **PRAD**.

The central equation is:

\[
A_{ij}^{(\ell,h)}
=
\frac{Q_i^{(\ell,h)}K_j^{(\ell,h)\top}}{\sqrt{d_h}}
+
U_{ij,h}^{\mathrm{HLT}}
+
\alpha_{\ell,h} B_{ij,h}^{\mathrm{HLT\text{-}pred}},
\]

where:

- \(A_{ij}^{(\ell,h)}\) is the pre-softmax attention score from particle \(i\) to particle \(j\), in layer \(\ell\) and head \(h\);
- \(U_{ij,h}^{\mathrm{HLT}}\) is the ordinary ParT interaction bias computed from HLT pairwise kinematics;
- \(B_{ij,h}^{\mathrm{HLT\text{-}pred}}\) is an additional relation bias predicted using only HLT information;
- \(\alpha_{\ell,h}\) is a learnable gate initialized to zero.

The offline reconstruction is never passed into the deployable classifier. It is used only to provide dense training supervision for \(B^{\mathrm{HLT\text{-}pred}\).

This proposal is deliberately not limited to predicting a hand-designed label such as "same subjet." The primary target is a **learned offline relation representation that is optimized directly for the jet classification objective**. Physics-defined offline targets such as common-subjet and common-vertex labels are added as anchors, not treated as the complete definition of a useful relationship.

That choice gives the method the best chance of increasing final tagging performance because it combines:

1. **Task alignment:** the offline relation is learned to improve the class decision.
2. **Dense supervision:** a jet with \(N\) HLT particles provides up to \(N^2\) pair targets.
3. **Architectural leverage:** the learned relation is inserted at the same location where ParT already benefits from pairwise information.
4. **Safe optimization:** the added branch begins with zero influence, so training starts exactly from the best HLT baseline.
5. **Deployment honesty:** every quantity used at inference is calculated from HLT inputs alone.

---

# 1. The problem

For every training jet, assume that we have:

\[
\left(X^{\mathrm{HLT}}, X^{\mathrm{OFF}}, y\right),
\]

where:

- \(X^{\mathrm{HLT}}\) is the HLT particle representation;
- \(X^{\mathrm{OFF}}\) is the corresponding offline particle representation of the same physical jet;
- \(y\) is the jet class.

The deployed classifier must be:

\[
\hat y = S(X^{\mathrm{HLT}}).
\]

It must not depend on \(X^{\mathrm{OFF}}\) at inference.

The objective is not to reconstruct every offline number. The objective is to use offline reconstruction to answer a more targeted training-time question:

> What particle-to-particle structure does the higher-quality offline reconstruction reveal that would help an HLT classifier organize its evidence?

The student then learns to infer the recoverable part of that structure from HLT information.

---

# 2. Why relational attention is the best attack surface

ParT already demonstrates that the location at which information enters a model matters. Its pairwise interaction quantities are calculated from four-vectors that are also present in the ordinary particle inputs. Therefore, the pairwise branch does not necessarily add new physical information. It makes useful relationships easier for attention to exploit.

In the original ParT ablation, replacing particle multi-head attention with ordinary attention reduced background rejection by roughly 20 to 30 percent for most signal classes, even though the removed pairwise quantities were derived from information already available to the model [1].

That establishes an important principle:

> A strong transformer can still benefit substantially when useful particle relationships are represented explicitly inside attention.

GN2 establishes a second useful principle. It trains a transformer with auxiliary objectives for track origin and track-pair vertex compatibility. Removing both auxiliary objectives reduced charm-jet and light-jet rejection by up to 30 percent in the reported ablation [2].

The proposed method combines these principles:

- use offline information to teach particle relationships;
- do not leave those predictions as auxiliary outputs only;
- insert the predicted relationships back into the transformer attention calculation.

The method is more targeted than global denoising. It does not ask the model to reproduce every offline feature, including features irrelevant to tagging or impossible to infer from HLT. It asks the model to reproduce the offline pair structure that a trained classifier actually uses.

---

# 3. Why the primary target should be learned rather than only hand-designed

A first idea would be to predict labels such as:

- same offline subjet;
- same decay branch;
- same reconstructed vertex;
- same heavy-flavour decay system.

These labels can be useful, but none is guaranteed to be the optimal relation for the final classification task.

For example:

- a fixed subjet definition depends on a clustering algorithm and scale;
- a common-vertex target is highly relevant for heavy-flavour tagging but less universal for all jet classes;
- two particles can be useful to compare even when they do not belong to the same physical branch;
- a classifier may need repulsive relationships, not only attractive ones;
- useful relationships may depend on the entire jet context, not only on an isolated pair.

The primary relation target should therefore be learned by an offline teacher whose own attention depends on that relation. The teacher is trained for the same class objective as the HLT classifier. Its relation representation is consequently pressured to contain information that improves tagging.

Physics-defined relation labels are still valuable. They make the learned relation more stable, interpretable, and physically grounded. They are included as secondary losses.

The final design is therefore a hybrid:

1. **A learned class-optimized offline relation target.**
2. **Explicit offline substructure targets that anchor the relation representation.**

---

# 4. Required data objects

For each jet, store the following objects.

## 4.1 HLT particles

\[
X^{\mathrm{HLT}}
=
\{x_i^{\mathrm{HLT}}\}_{i=1}^{N_H}.
\]

Use exactly the particle selection, ordering, truncation, masks, and normalization employed by the strongest existing HLT ParT baseline. Do not change the baseline preprocessing while testing the new method.

## 4.2 Offline particles

\[
X^{\mathrm{OFF}}
=
\{x_a^{\mathrm{OFF}}\}_{a=1}^{N_O}.
\]

The offline teacher may use every offline feature that is legitimately available in the paired training sample. This can include more accurate kinematics, particle identification, track quantities, vertex association, and other offline reconstruction outputs.

## 4.3 HLT-to-offline association

Define:

\[
\pi(i) =
\begin{cases}
a, & \text{if HLT particle } i \text{ is associated with offline particle } a,\\
\varnothing, & \text{if no reliable association exists.}
\end{cases}
\]

The primary implementation assumes that the paired dataset preserves particle identity, so \(\pi(i)\) is directly available.

If particle identity is not preserved, use the explicit matching procedure in Section 15. Do not force uncertain matches. Unmatched particles remain valid classifier inputs, but they are masked out of particle-level and pair-level offline supervision.

## 4.4 Valid pair mask

A pair \((i,j)\) receives an offline relational target only when both particles have valid offline associations:

\[
M_{ij}
=
\mathbb{1}[\pi(i)\neq\varnothing]
\mathbb{1}[\pi(j)\neq\varnothing].
\]

Self-pairs are excluded from the distillation loss:

\[
M_{ii}=0.
\]

They may remain present in ordinary self-attention.

## 4.5 Event splitting

Split the dataset by physical event before training either model.

The offline and HLT versions of one event must always remain in the same split.

Use:

- training split;
- validation split;
- final untouched test split.

The offline teacher is trained only on the training split. It is then frozen before student training.

---

# 5. Model overview

PRAD contains two models during training.

## 5.1 Offline teacher

The teacher receives offline particles:

\[
T(X^{\mathrm{OFF}}).
\]

It has four components:

1. offline particle embedding;
2. two contextualization blocks;
3. an offline pair-relation module;
4. six relation-biased particle-attention blocks followed by the usual class-attention blocks.

The teacher relation module produces a relation vector for every offline particle pair. That vector is transformed into an additive attention bias and therefore directly affects the teacher's class prediction.

## 5.2 HLT student

The student receives only HLT particles:

\[
S(X^{\mathrm{HLT}}).
\]

It uses the same high-level structure:

1. the strongest existing HLT particle embedding;
2. two ordinary HLT ParT blocks;
3. an HLT pair-relation predictor;
4. six relation-biased HLT particle-attention blocks;
5. the usual class-attention blocks and classifier.

The HLT relation predictor is trained to reproduce the frozen teacher relation for matched particle pairs.

At inference, the teacher and all offline targets are removed.

---

# 6. Exact architecture

Assume the existing ParT baseline uses:

- \(L=8\) particle-attention blocks;
- \(H=8\) attention heads;
- particle embedding width \(d=128\);
- the standard ParT interaction branch with 8 output channels.

These are the original baseline ParT dimensions [1]. If the existing HLT model uses different dimensions, preserve those dimensions and set the privileged bias output dimension equal to the number of attention heads.

## 6.1 Contextualization stem

Use the first two particle-attention blocks as a contextualization stem.

For the HLT student:

\[
\{h_i^{\mathrm{HLT}}\}
=
\mathrm{Stem}_{\mathrm{HLT}}
\left(
X^{\mathrm{HLT}},
U^{\mathrm{HLT}}
\right).
\]

For the offline teacher:

\[
\{h_a^{\mathrm{OFF}}\}
=
\mathrm{Stem}_{\mathrm{OFF}}
\left(
X^{\mathrm{OFF}},
U^{\mathrm{OFF}}
\right).
\]

Each stem consists of:

- the particle input embedding;
- two standard ParT particle-attention blocks;
- the ordinary ParT pairwise interaction bias for the corresponding reconstruction level.

The relation module is placed after two blocks rather than directly on raw particles because many meaningful relationships are contextual. Whether two particles form a useful branch can depend on the other particles in the jet.

The first two blocks give the relation module access to jet context while leaving six later blocks in which the predicted relation can affect information flow.

## 6.2 Symmetric pair representation

For every pair, construct:

\[
s_{ij}
=
\left[
h_i+h_j,\;
|h_i-h_j|,\;
h_i\odot h_j,\;
q_{ij}
\right],
\]

where:

- \(h_i\) and \(h_j\) are contextual token embeddings;
- \(\odot\) is elementwise multiplication;
- \(q_{ij}\) contains raw pair features.

The use of \(h_i+h_j\), \(|h_i-h_j|\), and \(h_i\odot h_j\) makes the learned relation symmetric:

\[
r_{ij}=r_{ji}.
\]

This is appropriate for relations such as common branch or common vertex. Attention remains directional through the ordinary \(Q_iK_j^\top\) term.

## 6.3 Raw pair features

For both teacher and student, \(q_{ij}\) must include the ordinary ParT pair quantities:

\[
\left(
\log \Delta R_{ij},
\log k_{T,ij},
\log z_{ij},
\log m_{ij}^{2}
\right).
\]

The offline version is calculated from offline four-vectors. The HLT version is calculated from HLT four-vectors.

In addition, append pairwise combinations of all available per-particle scalar features:

\[
q_{ij}^{\mathrm{extra}}
=
\left[
x_i^{\mathrm{scalar}}+x_j^{\mathrm{scalar}},
|x_i^{\mathrm{scalar}}-x_j^{\mathrm{scalar}}|,
x_i^{\mathrm{scalar}}\odot x_j^{\mathrm{scalar}}
\right].
\]

Categorical particle types are embedded before forming pair combinations.

Do not pass an offline-only feature directly into the HLT relation predictor. The teacher and student pair modules may have different input dimensions, but they must output relation vectors with the same dimension.

## 6.4 Relation vector

The relation module is a three-layer MLP:

\[
r_{ij}
=
\mathrm{LayerNorm}
\left(
\mathrm{MLP}_{\mathrm{rel}}(s_{ij})
\right),
\qquad
r_{ij}\in\mathbb{R}^{16}.
\]

Use hidden dimensions:

\[
256 \rightarrow 128 \rightarrow 16.
\]

Use GELU activations and dropout \(0.1\) after the first two layers.

The 16-dimensional bottleneck is intentional:

- it is much richer than a single same-subjet probability;
- it is still small enough to distill reliably;
- it prevents the relation branch from becoming a second unrestricted transformer;
- it makes the added computational cost manageable.

## 6.5 Conversion to an attention bias

Convert the relation vector into one scalar per attention head:

\[
\widetilde B_{ij}
=
3\tanh(W_Br_{ij}+b_B),
\qquad
\widetilde B_{ij}\in\mathbb{R}^{H}.
\]

With \(H=8\), \(W_B\in\mathbb{R}^{8\times16}\).

The factor of 3 bounds every raw privileged bias component to \([-3,3]\). This prevents an unstable relation module from overwhelming the ordinary dot-product attention early in training.

Attention is invariant to adding the same constant to every key position for a fixed query. Remove this unidentifiable degree of freedom by centering over valid keys:

\[
B_{ij,h}
=
\widetilde B_{ij,h}
-
\frac{
\sum_{k \in \mathcal V}
\widetilde B_{ik,h}
}{
|\mathcal V|
},
\]

where \(\mathcal V\) is the set of valid, unmasked particles in the jet.

This centering is applied in both teacher and student.

## 6.6 Injection into later attention blocks

For particle-attention blocks \(\ell=3,\dots,8\):

\[
A_{ij}^{(\ell,h)}
=
\frac{
Q_i^{(\ell,h)}
K_j^{(\ell,h)\top}
}{
\sqrt{d_h}
}
+
U_{ij,h}
+
\alpha_{\ell,h}B_{ij,h}.
\]

Use a separate gate for every later layer and head:

\[
\alpha_{\ell,h}=\tanh(a_{\ell,h}).
\]

Initialize:

\[
a_{\ell,h}=0.
\]

Therefore:

\[
\alpha_{\ell,h}=0
\]

at initialization, and the student initially behaves exactly like the baseline HLT ParT.

This is one of the most important implementation choices. It ensures that adding the new module cannot immediately damage the already strong baseline. Each attention head must learn that the privileged relation is useful before the relation is allowed to affect it.

## 6.7 Shared versus layer-specific relation

Compute one relation matrix \(B\) per jet and reuse it in blocks 3 through 8, with different gates \(\alpha_{\ell,h}\).

Do not initially train a separate relation matrix for every layer.

Reasons:

- ParT itself reuses its interaction matrix across particle-attention blocks [1];
- a shared relation target is easier to distill;
- layer-specific targets multiply memory and optimization difficulty;
- per-layer gates already allow different layers to use the relation differently.

A layer-specific extension should be tested only after the shared version works.

---

# 7. Offline teacher targets

The teacher relation representation is trained using two sources of pressure.

## 7.1 Primary pressure: jet classification

The teacher's relation bias is inserted into its own attention blocks. The normal classification loss therefore trains the relation module to produce relationships that improve the final jet class decision.

This is the primary source of task alignment.

The teacher class loss is:

\[
\mathcal L_{\mathrm{teacher,class}}
=
\mathrm{CE}
\left(
y,
p_T(y\mid X^{\mathrm{OFF}})
\right).
\]

## 7.2 Physics anchor 1: multiscale offline branch membership

Recluster the offline jet constituents using the Cambridge/Aachen algorithm with the same jet radius \(R\) used to define the original jet.

Construct exclusive subjet partitions with:

\[
K\in\{2,3,4\}.
\]

For each offline particle pair \((a,b)\), define:

\[
y_{ab}^{(K)}
=
\mathbb{1}
[
a \text{ and } b
\text{ belong to the same exclusive } K\text{-subjet}
].
\]

This produces three binary pair labels:

\[
y_{ab}^{\mathrm{tree}}
=
\left[
y_{ab}^{(2)},
y_{ab}^{(3)},
y_{ab}^{(4)}
\right].
\]

Predict them from the relation vector:

\[
\hat y_{ab}^{\mathrm{tree}}
=
\sigma(W_{\mathrm{tree}}r_{ab}+b_{\mathrm{tree}}).
\]

Use class-balanced binary cross entropy independently for each \(K\). Compute positive weights from the training set.

These targets give the relation bottleneck an explicit multiscale notion of prong and branch structure without committing to one arbitrary subjet multiplicity.

## 7.3 Physics anchor 2: common reconstructed vertex

For charged offline tracks with a valid reconstructed vertex assignment, define:

\[
y_{ab}^{\mathrm{vtx}}
=
\mathbb{1}
[
v(a)=v(b)
].
\]

Use a binary prediction head:

\[
\hat y_{ab}^{\mathrm{vtx}}
=
\sigma(W_{\mathrm{vtx}}r_{ab}+b_{\mathrm{vtx}}).
\]

Mask the loss unless:

- both particles are charged tracks;
- both have valid offline vertex assignments.

Do not label an unassigned pair as a negative. An absent vertex assignment is missing supervision, not proof that the tracks come from different vertices.

The common-vertex target is optional only when the dataset genuinely contains no usable offline vertex association. In that case, set the entire vertex loss to zero and keep the rest of the method unchanged.

## 7.4 Teacher objective

Normalize every component by its number of valid examples, then use:

\[
\mathcal L_T
=
\mathcal L_{\mathrm{teacher,class}}
+
0.2\mathcal L_{\mathrm{tree}}
+
0.2\mathcal L_{\mathrm{vtx}}.
\]

The classification term remains dominant. The semantic targets shape the relation space but do not define it completely.

---

# 8. What is distilled into the HLT student

Freeze the complete offline teacher after training.

For every HLT particle \(i\) with offline match \(\pi(i)\), extract:

\[
r_{ij}^{T}
=
r_{\pi(i)\pi(j)}^{\mathrm{OFF}},
\]

and:

\[
B_{ij}^{T}
=
B_{\pi(i)\pi(j)}^{\mathrm{OFF}}.
\]

The HLT student predicts:

\[
r_{ij}^{S}
=
R_{\mathrm{HLT}}
\left(
h_i^{\mathrm{HLT}},
h_j^{\mathrm{HLT}},
q_{ij}^{\mathrm{HLT}}
\right),
\]

and:

\[
B_{ij}^{S}
=
B(r_{ij}^{S}).
\]

The student relation loss matches both the bottleneck and the actual bias:

\[
\mathcal L_{\mathrm{rel}}
=
\frac{1}{2}
\mathcal L_{\mathrm{bottleneck}}
+
\frac{1}{2}
\mathcal L_{\mathrm{bias}}.
\]

Use Smooth L1 loss:

\[
\mathcal L_{\mathrm{bottleneck}}
=
\frac{
\sum_{i,j}
M_{ij}w_{ij}
\operatorname{SmoothL1}
\left(
r_{ij}^{S},
\operatorname{stopgrad}(r_{ij}^{T})
\right)
}{
\sum_{i,j}M_{ij}w_{ij}
},
\]

\[
\mathcal L_{\mathrm{bias}}
=
\frac{
\sum_{i,j}
M_{ij}w_{ij}
\operatorname{SmoothL1}
\left(
B_{ij}^{S},
\operatorname{stopgrad}(B_{ij}^{T})
\right)
}{
\sum_{i,j}M_{ij}w_{ij}
}.
\]

## 8.1 Reliability weighting

Do not trust every teacher output equally.

For jet \(n\), define:

\[
w_{\mathrm{jet}}
=
\operatorname{clip}
\left(
p_T(y_{\mathrm{true}}\mid X^{\mathrm{OFF}}),
0.1,
1.0
\right).
\]

For pair \((i,j)\), define:

\[
w_{\mathrm{pair}}
=
\operatorname{clip}
\left(
1+
2\cdot
\frac{1}{H}
\sum_h
|B_{ij,h}^{T}|,
1,
5
\right).
\]

Then:

\[
w_{ij}
=
w_{\mathrm{jet}}w_{\mathrm{pair}}.
\]

This gives more weight to:

- jets on which the offline teacher supports the correct label;
- pair relations that have a nontrivial teacher bias.

It still retains low-weight supervision from other pairs, which prevents the student from learning only a few extreme edges.

---

# 9. Student losses

The deployable student is trained with four losses.

## 9.1 Hard class loss

\[
\mathcal L_{\mathrm{hard}}
=
\mathrm{CE}
\left(
y,
p_S(y\mid X^{\mathrm{HLT}})
\right).
\]

This loss always has coefficient 1.

## 9.2 Privileged relation distillation

\[
\mathcal L_{\mathrm{rel}}
\]

is defined in Section 8.

Its default coefficient after warm-up is:

\[
\lambda_{\mathrm{rel}}=1.0.
\]

## 9.3 Explicit offline structure prediction

Apply the same semantic heads to the student relation vector and train the student to predict the offline tree and vertex labels:

\[
\mathcal L_{\mathrm{sem}}
=
\mathcal L_{\mathrm{tree}}^{S}
+
\mathcal L_{\mathrm{vtx}}^{S}.
\]

Use:

\[
\lambda_{\mathrm{sem}}=0.2.
\]

This prevents the student from matching only an arbitrary teacher coordinate system. It must also recover physically interpretable offline relationships.

## 9.4 Teacher class-logit distillation

Let \(z_T\) and \(z_S\) be the teacher and student class logits. Use temperature \(T=2\):

\[
\mathcal L_{\mathrm{KD}}
=
T^2
\operatorname{KL}
\left[
\operatorname{softmax}(z_T/T)
\;\|\;
\operatorname{softmax}(z_S/T)
\right].
\]

Use:

\[
\lambda_{\mathrm{KD}}=0.5.
\]

The class-logit term is not the novelty of the method. It is included because the stated goal is the largest final tagging gain, and jet-tagging studies have shown that knowledge distillation can improve a student classifier even without changing its inference architecture [4].

The full student objective is:

\[
\boxed{
\mathcal L_S
=
\mathcal L_{\mathrm{hard}}
+
1.0\mathcal L_{\mathrm{rel}}
+
0.2\mathcal L_{\mathrm{sem}}
+
0.5\mathcal L_{\mathrm{KD}}
}
\]

after warm-up.

All losses must be normalized independently before applying these coefficients.

---

# 10. Training procedure

## Stage 0: reproduce the HLT baseline

Train the current strongest HLT ParT with the exact intended split, preprocessing, and evaluation code.

Save:

- weights;
- optimizer-independent checkpoint;
- validation predictions;
- test predictions;
- per-class metrics for five random seeds.

The PRAD student must be initialized from this checkpoint.

Do not compare a carefully tuned PRAD model against a poorly reproduced baseline.

## Stage 1: train the offline teacher

Train the offline teacher from scratch using:

\[
\mathcal L_T
=
\mathcal L_{\mathrm{teacher,class}}
+
0.2\mathcal L_{\mathrm{tree}}
+
0.2\mathcal L_{\mathrm{vtx}}.
\]

Selection criterion:

- choose the checkpoint with the best primary tagging metric on the offline validation set;
- do not select by relation-prediction accuracy alone.

After selection:

- freeze every teacher parameter;
- store its validation and test performance as an offline ceiling;
- do not update the teacher during student training.

## Stage 2: run the oracle experiment before full student training

Create an HLT classifier with the PRAD attention interface, but replace the HLT-predicted bias with the frozen offline teacher bias:

\[
B^{S} \leftarrow B^{T}.
\]

Train and evaluate this model using paired HLT and offline data.

This model is not deployable. Its purpose is to answer:

> If the HLT transformer had access to the offline teacher's pair relations, would those relations improve HLT tagging?

Use the same zero-initialized gates and allow them to train.

Decision rule:

- if the oracle improves the primary tagging metric by less than 2 percent relative, stop this relation definition;
- if the oracle improves it by 2 to 5 percent, continue only as a low-cost pilot;
- if the oracle improves it by more than 5 percent, proceed with the full student;
- if the oracle improves it by more than 10 percent, the relation channel has substantial headroom and deserves priority.

This test prevents spending weeks on a relation predictor whose target does not help classification even when known perfectly.

## Stage 3: relation warm-up

Initialize the student from the best HLT baseline.

Copy the teacher's:

- relation-to-bias matrix \(W_B,b_B\);
- tree prediction head;
- vertex prediction head.

During warm-up:

- freeze the HLT particle embedding;
- freeze all eight baseline particle-attention blocks;
- freeze class-attention blocks and classifier;
- freeze copied \(W_B,b_B\) and semantic heads;
- keep all relation gates fixed at zero;
- train only the HLT relation MLP.

Use for five epochs:

\[
\mathcal L_{\mathrm{warm}}
=
\mathcal L_{\mathrm{rel}}
+
0.2\mathcal L_{\mathrm{sem}}.
\]

This stage teaches the HLT relation branch the teacher coordinate system before its outputs are allowed to modify attention.

## Stage 4: controlled integration

For the next five epochs:

- unfreeze relation gates;
- unfreeze particle-attention blocks 5 through 8;
- unfreeze class-attention blocks and classifier;
- keep particle-attention blocks 1 through 4 frozen;
- use the full student loss.

Use learning rates:

- relation module: \(3\times10^{-4}\);
- gates: \(3\times10^{-4}\);
- unfrozen transformer and classifier weights: \(3\times10^{-5}\).

Linearly ramp \(\lambda_{\mathrm{KD}}\) from 0 to 0.5 over these five epochs.

## Stage 5: full end-to-end fine-tuning

Unfreeze the entire student.

Use:

- relation module learning rate: \(1\times10^{-4}\);
- all pretrained baseline weights: \(1\times10^{-5}\);
- gates: \(1\times10^{-4}\);
- AdamW;
- weight decay \(1\times10^{-5}\);
- cosine decay;
- gradient clipping at norm 1.0;
- early stopping on the primary validation tagging metric with patience 10 epochs.

Continue for at most 50 epochs.

Keep the class loss coefficient fixed at 1.0. Do not let relation matching dominate the actual tagging objective.

## Stage 6: loss-weight sweep

After the default run works technically, run this limited sweep:

\[
\lambda_{\mathrm{rel}}\in\{0.25,0.5,1.0\},
\]

\[
\lambda_{\mathrm{sem}}\in\{0,0.1,0.2\},
\]

\[
\lambda_{\mathrm{KD}}\in\{0,0.25,0.5\}.
\]

Do not perform a full Cartesian product initially.

Use:

1. vary \(\lambda_{\mathrm{rel}}\) with the other coefficients fixed;
2. choose the best value;
3. vary \(\lambda_{\mathrm{sem}}\);
4. choose the best value;
5. vary \(\lambda_{\mathrm{KD}}\).

This requires seven additional configurations rather than 27.

---

# 11. Important rule: never teacher-force the deployable student attention

During ordinary PRAD student training, the attention calculation must use:

\[
B^{\mathrm{HLT\text{-}pred}},
\]

not:

\[
B^{\mathrm{OFF}}.
\]

Offline relations appear only in losses.

The only exception is the explicitly labeled oracle experiment.

If the deployable student receives the true offline bias during training and its own predicted bias during inference, it will experience a serious train-test mismatch. It may learn to depend on relation quality that the HLT predictor cannot achieve.

---

# 12. Why raw offline attention maps are not the target

Do not force the HLT student's complete attention matrix to equal the offline teacher's complete attention matrix.

A transformer attention matrix contains:

\[
\frac{QK^\top}{\sqrt d}
+
\text{pair bias}.
\]

The \(QK^\top\) portion depends on the model's internal token representation. Two models can implement similar functions while using different heads or attention patterns. Head permutations and internal reparameterizations make direct map matching unnecessarily fragile.

Recent studies of ParT attention also show that the relationship between its ordinary attention term, interaction matrix, sparsity, and performance is not simple [5].

PRAD therefore distills a deliberately separated object:

> the teacher's learned pair-relation bottleneck and its explicit additive pair bias.

That object has a fixed location and purpose in the architecture. It is more stable than the full post-softmax attention map.

---

# 13. Exact ablation suite

The following experiments are required. Without them, a gain would be difficult to interpret.

## A. Baseline HLT ParT

Standard class loss only.

## B. Capacity control

Add the same HLT relation module and attention gates, but train it only from the hard class loss.

Purpose:

- determines whether gains come merely from extra parameters.

## C. Logit KD only

Baseline HLT ParT plus \(\mathcal L_{\mathrm{KD}}\), without relation prediction or relation-biased attention.

Purpose:

- measures ordinary teacher-student distillation.

## D. Semantic auxiliary only

Predict offline tree and vertex labels, but do not feed the relation into attention.

Purpose:

- tests ordinary multitask representation learning.

## E. Semantic attention

Predict offline tree and vertex labels and use the resulting relation vector in attention, but do not distill the learned teacher relation.

Purpose:

- tests whether hand-defined offline structure is sufficient.

## F. Learned relation auxiliary only

Distill \(r^T\) and \(B^T\), but keep all \(\alpha_{\ell,h}=0\).

Purpose:

- tests whether relational supervision helps only by regularizing shared representations.

## G. PRAD without logit KD

Use relation distillation, semantic losses, and relation-biased attention, but set:

\[
\lambda_{\mathrm{KD}}=0.
\]

Purpose:

- isolates the new mechanism from ordinary class distillation.

## H. Full PRAD

Use all losses and HLT-predicted relation-biased attention.

## I. Oracle relation attention

Use the true frozen offline teacher relation bias.

Purpose:

- estimates the maximum gain available from this relation definition.

## J. Shuffled relation control

Within each batch, randomly permute the teacher relation matrices across jets before relation supervision.

Keep jet classes unconditioned during shuffling.

Purpose:

- confirms that gains depend on the correct paired offline structure rather than generic regularization or target statistics.

Expected result:

- shuffled relations should not outperform the capacity control.

---

# 14. Evaluation protocol

## 14.1 Primary metric

For each signal class \(c\), measure background rejection at the standard fixed signal efficiency used by the existing project:

\[
\mathrm{Rej}_c
=
\frac{1}{\epsilon_{\mathrm{bkg},c}}.
\]

Before running experiments, freeze the efficiency point for every class. Do not choose a different working point after seeing results.

For a single scalar model-selection score, use:

\[
S_{\mathrm{macro}}
=
\frac{1}{C}
\sum_{c=1}^{C}
\log
\left(
\mathrm{Rej}_c
\right).
\]

Use the log because rejection values can vary by orders of magnitude.

The final report must still show every class separately.

## 14.2 Secondary metrics

Report:

- multiclass accuracy;
- macro AUC;
- per-class AUC;
- confusion matrix;
- calibration error;
- parameter count;
- inference latency;
- peak memory;
- average and distribution of learned \(\alpha_{\ell,h}\);
- relation prediction loss on validation;
- tree-label AUC;
- vertex-label AUC;
- matched-particle and matched-pair coverage.

## 14.3 Statistical procedure

Run every main configuration with five random seeds.

Report:

- mean;
- standard deviation;
- paired difference relative to baseline for each seed;
- bootstrap 95 percent confidence interval on the test predictions.

A claimed gain should exceed both:

- the seed-to-seed variation;
- the uncertainty from the finite test sample.

## 14.4 Distributional robustness

At minimum, stratify results by:

- jet \(p_T\);
- jet \(\eta\);
- particle multiplicity;
- fraction of matched HLT particles;
- class;
- offline teacher confidence.

If alternative simulation or reconstruction conditions exist, evaluate them without retraining.

The method should not improve only the easiest, best-matched jets.

---

# 15. Particle matching when direct identity is unavailable

Use direct detector or reconstruction associations whenever possible. They are preferred over geometric matching.

If direct associations do not exist, perform matching independently for charged and neutral particles.

## 15.1 Charged-particle matching

Require:

- equal reconstructed charge;
- compatible particle category;
- \(\Delta R < 0.03\);
- \(0.5 < p_T^{\mathrm{HLT}}/p_T^{\mathrm{OFF}} < 2.0\).

Define cost:

\[
c_{ia}^{\mathrm{charged}}
=
\left(
\frac{\Delta R_{ia}}{0.01}
\right)^2
+
\left(
\frac{
\log(p_{T,i}^{\mathrm{HLT}}/p_{T,a}^{\mathrm{OFF}})
}{0.2}
\right)^2.
\]

Use Hungarian assignment to obtain a one-to-one minimum-cost matching.

Reject an assigned pair if:

\[
c_{ia}^{\mathrm{charged}}>9.
\]

## 15.2 Neutral-particle matching

Require compatible neutral category and:

- \(\Delta R < 0.06\);
- \(0.25 < E^{\mathrm{HLT}}/E^{\mathrm{OFF}} < 4.0\).

Use:

\[
c_{ia}^{\mathrm{neutral}}
=
\left(
\frac{\Delta R_{ia}}{0.02}
\right)^2
+
\left(
\frac{
\log(E_i^{\mathrm{HLT}}/E_a^{\mathrm{OFF}})
}{0.35}
\right)^2.
\]

Again use Hungarian assignment and reject matches with:

\[
c_{ia}^{\mathrm{neutral}}>9.
\]

These are initial operational thresholds, not universal detector constants. Validate them by plotting the accepted and rejected residual distributions. Tighten them if obvious false matches remain.

## 15.3 Matching-quality requirements

Before training PRAD, report:

- fraction of HLT particles matched;
- fraction of HLT transverse momentum carried by matched particles;
- matched fraction by particle type;
- matched fraction by class and jet \(p_T\);
- distribution of \(\Delta R\);
- distribution of \(p_T^{\mathrm{HLT}}/p_T^{\mathrm{OFF}}\).

Do not hide low-coverage regions. Relation distillation can only teach event-specific offline structure where correspondence is reliable.

---

# 16. Pseudocode

```python
# Stage 1: train offline teacher
for x_off, y, offline_structure in train_loader:
    h_off = teacher.embed_and_first_two_blocks(x_off)

    r_off = teacher.relation_module(
        contextual_pairs(h_off),
        raw_offline_pair_features(x_off),
    )

    b_off = centered_bounded_bias(
        teacher.bias_projection(r_off)
    )

    logits_off = teacher.remaining_blocks_and_classifier(
        h_off,
        base_pair_bias=standard_part_bias(x_off),
        privileged_pair_bias=b_off,
    )

    loss = (
        cross_entropy(logits_off, y)
        + 0.2 * tree_relation_loss(r_off, offline_structure)
        + 0.2 * vertex_relation_loss(r_off, offline_structure)
    )

    loss.backward()
    optimizer.step()

freeze(teacher)
```

```python
# Stage 2: train HLT student
for x_hlt, x_off, y, match_map, offline_structure in train_loader:
    with torch.no_grad():
        h_off = teacher.embed_and_first_two_blocks(x_off)
        r_off = teacher.relation_module(
            contextual_pairs(h_off),
            raw_offline_pair_features(x_off),
        )
        b_off = centered_bounded_bias(
            teacher.bias_projection(r_off)
        )
        logits_off = teacher.remaining_blocks_and_classifier(
            h_off,
            standard_part_bias(x_off),
            b_off,
        )

        r_target, b_target, pair_mask = gather_matched_pairs(
            r_off, b_off, match_map
        )

    h_hlt = student.embed_and_first_two_blocks(x_hlt)

    r_hlt = student.relation_module(
        contextual_pairs(h_hlt),
        raw_hlt_pair_features(x_hlt),
    )

    b_hlt = centered_bounded_bias(
        student.bias_projection(r_hlt)
    )

    logits_hlt = student.remaining_blocks_and_classifier(
        h_hlt,
        base_pair_bias=standard_part_bias(x_hlt),
        privileged_pair_bias=b_hlt,
    )

    loss = (
        cross_entropy(logits_hlt, y)
        + 1.0 * relation_distillation_loss(
            r_hlt, b_hlt, r_target, b_target, pair_mask
        )
        + 0.2 * semantic_offline_relation_loss(
            r_hlt, offline_structure, match_map
        )
        + 0.5 * temperature_kl(
            logits_hlt, logits_off, temperature=2.0
        )
    )

    loss.backward()
    optimizer.step()
```

The student forward pass used in deployment is simply:

```python
logits = student(x_hlt)
```

No offline object is involved.

---

# 17. Failure modes and the exact diagnosis for each

## 17.1 The oracle gives no gain

Interpretation:

- the chosen teacher relation does not add useful information to HLT attention;
- or the relation is redundant with the baseline HLT pair bias;
- or the student attention architecture cannot use it.

Action:

- do not continue optimizing the predictor;
- inspect whether gates remain near zero;
- try a different teacher relation definition before adding more student complexity.

## 17.2 The oracle gives a large gain, but full PRAD does not

Interpretation:

- useful offline information exists;
- the HLT relation predictor cannot recover enough of it.

Action:

- measure performance versus matching quality;
- increase contextual stem depth from 2 to 4;
- add better HLT track-quality and particle-type features;
- increase relation bottleneck from 16 to 32;
- use confidence-gated biases;
- do not increase transformer depth first.

## 17.3 Auxiliary-only matches full PRAD

Interpretation:

- offline supervision improves the shared representation;
- feeding the prediction into attention adds little.

Action:

- prefer the simpler auxiliary model;
- the research conclusion is still valuable, but the attention mechanism is not necessary.

## 17.4 Capacity control matches full PRAD

Interpretation:

- gain is caused by extra parameters or optimization, not offline information.

Action:

- reject the privileged-learning claim;
- verify the shuffled relation control;
- reduce or match parameter counts more carefully.

## 17.5 Relation loss decreases but tagging does not improve

Interpretation:

- the student is matching teacher relations that are easy but irrelevant;
- the relation coefficient may be too large;
- semantic targets may dominate the useful learned channels.

Action:

- reduce \(\lambda_{\mathrm{rel}}\);
- weight relation loss more strongly by teacher bias magnitude;
- select teacher checkpoints by tagging, not relation accuracy;
- inspect whether learned gates remain near zero.

## 17.6 Training becomes unstable

Likely cause:

- relation bias magnitude becomes too large;
- gates open too quickly;
- relation and baseline learning rates are too similar.

Action:

- confirm use of \(3\tanh(\cdot)\);
- confirm key-wise centering;
- keep gates at zero through warm-up;
- reduce gate learning rate;
- clip gradients at norm 1.0.

## 17.7 Improvement appears only in high-match-quality jets

Interpretation:

- the method depends strongly on exact particle correspondence.

Action:

- report this honestly;
- consider set-level or optimal-transport relation distillation for unmatched particles;
- do not claim a universal HLT improvement until low-match regions are addressed.

## 17.8 Simulation robustness worsens

Interpretation:

- the offline teacher relation may encode reconstruction or generator-specific shortcuts.

Action:

- reduce reliance on free learned relation channels;
- increase semantic physics anchoring;
- test alternative generators and reconstruction settings;
- add domain-adversarial or decorrelation constraints only after identifying the specific unstable variable.

---

# 18. What not to do first

## 18.1 Do not reconstruct every offline feature

A full HLT-to-offline regression objective treats every offline discrepancy as equally important. Many discrepancies are irrelevant to tagging, and some are not recoverable from HLT.

Raw denoising can be tested as a baseline, but it should not be the central method.

## 18.2 Do not use only one global auxiliary label

Predicting a jet-level quantity such as offline mass provides one target per jet. The pairwise method provides many more training signals and acts closer to the transformer mechanism responsible for particle interaction.

## 18.3 Do not hard-mask attention using predicted subjets

A wrong hard mask can permanently prevent useful interactions.

Use an additive, bounded, learnable bias. The ordinary attention term must remain free to override it.

## 18.4 Do not use the true offline relation in normal student training

That creates a train-inference mismatch.

Use true offline relations only as loss targets and in the explicitly labeled oracle experiment.

## 18.5 Do not match complete attention maps

Distill the separated relation bottleneck and explicit pair bias, not the full post-softmax attention map.

## 18.6 Do not remove the original ParT interaction branch

The privileged bias should initially be additive:

\[
U^{\mathrm{HLT}}+\alpha B^{\mathrm{priv}}.
\]

This lets the model retain the known useful inductive bias while testing whether the new relation adds complementary information.

---

# 19. Minimum viable implementation

The minimum version that still tests the central hypothesis is:

1. train an offline ParT teacher;
2. expose its standard 8-channel PairEmbed output as \(B^T\);
3. train an HLT PairEmbed network to predict \(B^T\);
4. add the predicted bias to blocks 3 through 8 through zero-initialized gates;
5. train with hard class loss, relation-bias Smooth L1, and logit KD;
6. compare against baseline, capacity control, KD-only, auxiliary-only, and oracle.

This version avoids:

- contextual relation embeddings;
- tree labels;
- vertex labels;
- a 16-dimensional bottleneck.

It is useful as a fast engineering test.

However, it is not the version I expect to maximize tagging gain. The full contextual relation module should be stronger because it can represent relationships that depend on the rest of the jet rather than only on isolated raw pair features.

---

# 20. Recommended full experiment order

Run the work in this order.

## Experiment 1: offline ceiling

Train the strongest offline teacher.

Question:

- how much better is offline classification than HLT classification?

A very small offline-HLT gap limits all privileged methods.

## Experiment 2: oracle teacher relation

Inject the true offline teacher relation into an HLT classifier.

Question:

- does the relation itself contain useful headroom?

## Experiment 3: bias predictability

Train the HLT relation predictor with the backbone frozen.

Question:

- can HLT inputs predict the offline relation better than a constant or class-average baseline?

## Experiment 4: auxiliary-only model

Use relation losses without attention injection.

Question:

- does dense offline supervision improve the HLT representation?

## Experiment 5: full PRAD without logit KD

Question:

- does predicted offline relation improve attention beyond auxiliary regularization?

## Experiment 6: full PRAD with logit KD

Question:

- what is the largest achievable final tagging gain?

## Experiment 7: robustness and ablations

Question:

- is the gain real, paired, stable, and attributable to privileged relation learning?

This sequence gives an interpretable answer even if the full model fails.

---

# 21. The strongest scientific claim this setup could support

If the full model succeeds and the ablations behave as expected, the result would support the following claim:

> High-fidelity offline reconstruction can be compressed into a deployable HLT-only particle-relation predictor. When the predicted relation is inserted as a learned attention bias, it improves online jet classification beyond ordinary hard-label training, class-logit distillation, auxiliary substructure prediction, and an equal-capacity architectural control.

That is a much sharper claim than:

> Offline information helps HLT classification.

It identifies:

- what knowledge transfers;
- where it enters the model;
- why it affects the classifier;
- what is required at inference;
- how the gain differs from generic multitask learning and generic knowledge distillation.

---

# 22. Final recommendation

The main bet should be:

> **A contextual, class-optimized offline relation teacher whose pair representation is distilled into an HLT-only relation predictor and inserted through zero-initialized additive gates into later ParT attention blocks.**

The relation predictor should be trained with:

1. the offline teacher relation bottleneck;
2. the offline teacher attention-bias output;
3. multiscale offline branch labels;
4. offline common-vertex labels when available;
5. the hard jet class;
6. weak class-logit distillation.

The most important early experiment is the oracle relation test.

If the oracle has no gain, abandon this relation definition quickly.

If the oracle has a large gain and the predicted version recovers a meaningful fraction of it, this is likely one of the strongest ways to use offline reconstruction for improved HLT tagging.

---

# References

1. H. Qu, C. Li, and S. Qian, **Particle Transformer for Jet Tagging**, arXiv:2202.03772.  
   https://arxiv.org/abs/2202.03772

2. ATLAS Collaboration, **Transforming jet flavour tagging at ATLAS**, arXiv:2505.19689, published in Nature Communications 17 (2026) 541.  
   https://arxiv.org/abs/2505.19689

3. W. Wang et al., **MiniLMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers**, arXiv:2012.15828.  
   https://arxiv.org/abs/2012.15828

4. R. Liu et al., **Efficient and Robust Jet Tagging at the LHC with Knowledge Distillation**, arXiv:2311.14160.  
   https://arxiv.org/abs/2311.14160

5. T. Legge et al., **Why is Attention Sparse in Particle Transformer?**, arXiv:2512.00210.  
   https://arxiv.org/abs/2512.00210

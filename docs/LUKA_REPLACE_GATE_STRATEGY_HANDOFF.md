# REPLACE_GATE: privileged attention-gate teachers distilled into an HLT-only ParT

Written 2026-10-20. This document describes the **cold-start REPLACE_GATE**
strategy selected for further investigation, its implementation, the reported
Luka trial results, and how to adapt it without silently changing the experiment.
It is a handoff, not a new scientific registration or authorization to submit jobs.

Reading shortcuts:

- [Teacher chain and permitted inputs](#3-the-actual-teacher-chain).
- [Gate architecture](#5-the-additional-gate-precisely) and
  [training objectives](#6-training-objectives).
- [Reported results](#9-reported-results-and-what-they-support).
- [Code map](#10-where-to-find-the-implementation).
- [Historical reproduction](#12-reproducing-the-historical-setup) and
  [porting to another setting](#13-reproducing-the-idea-in-a-slightly-different-setting).

## 1. The idea in one paragraph

Keep ParT's ordinary particle features, Q/K/V attention and native pairwise
physics bias. Add a small network that produces an **additional, bounded
pairwise attention correction**. During the teacher ladder, that network can
look at the current particle view and an aligned, immediately richer view.
The main ParT still starts from the current view. At the final replacement
stage, train a fresh ParT whose extra gate sees **HLT only**, using both the
privileged teacher's class probabilities and its additional gate correction as
targets. At deployment, keep this student and its small HLT gate; discard the
teacher and all richer-view inputs.

This is not two-encoder fusion, particle concatenation, full-attention-matrix
distillation, or reconstruction of the teacher's native physics bias.

The important claim is limited: a richer-view teacher may learn a useful
attention correction, and an HLT-only student may learn enough of that correction
to improve classification. Neither perfect reconstruction nor recovery of
information absent from HLT is guaranteed.

## 2. Which implementation this document means

Use these authorities in order:

1. [Cold-start scientific plan](plans/LUKA_COLD_GATE_TIER3_PLAN.md).
2. [Cold-start v1 contracts](contracts/LUKA_COLD_GATE_TIER3.md).
3. The immutable gate and science specifications for the execution being read.
4. This explanatory handoff and the
   [cold-start queue/results runbook](LUKA_COLD_GATE_TIER3_RUNBOOK.md).

The reusable gate mathematics originated in the
[privileged-gate plan](plans/LUKA_PRIVILEGED_GATE_PLAN.md) and
[shared gate contract](contracts/LUKA_PRIVILEGED_GATE.md). The **cold package**
overrides the earlier experiment's initialization and artifact lifecycle.

### Historical execution identifiers

| Item | Identity |
| --- | --- |
| Cold execution source | `a1d3a2c6caa20871ce2d0b3c1525fd347fd27df6` |
| Cold campaign root | `/home/ryreu/atlas/HLT_Classification/checkpoints/luka_cold_gate_a1d3a2c6_r1` |
| Cold science specification | `<cold root>/science/campaign_spec.json` |
| Expected historical cold worktree | `/home/ryreu/atlas/HLT_Classification_lukacold_a1d3a2c6` |
| Fixed30 parent source | `74aa58c582ff6f497a2c6d91c503f5489fac3e9a` |
| Fixed30 parent specification | `/home/ryreu/atlas/HLT_Classification/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json` |
| Reused selected anchor | Parent's `COARSE_U100`, not the pure `OFFLINE` control |
| Cold source-compatibility base | `84fcbf25529091a2f32daf96bf8b910dec50dc3f` |

Resolve `project_dir`, source identity, parent references and artifact hashes
from the saved specifications before using these paths. A directory name is a
locator, not proof of provenance. The local shared-gate and cold-package source
was checked against the cold execution commit while writing this handoff; there
were no differences in those two packages. That does not authenticate remote
payloads or other source trees.

**Do not confuse this with the warm campaign**
`luka_gate_4e4a38a1_r1`. Both contain a row named `REPLACE_GATE`.
The warm implementation copies compatible selected teacher weights into its
students; the cold implementation does not. Warm `REPLACE_STAGED` additionally
freezes the native backbone for its first three epochs. There is no
`REPLACE_STAGED` in the registered cold campaign.

## 3. The actual teacher chain

```text
Existing selected COARSE_U100 anchor
  |
  | class-probability KD
  v
PRIV_D066: main ParT input D066; extra gate inputs D066 + U100
  |
  | class-probability KD
  v
PRIV_D033: main ParT input D033; extra gate inputs D033 + D066
  |
  | class-probability KD
  v
PRIV_D000: main ParT input D000; extra gate inputs D000 + D033
  |
  | class-probability KD + centered extra-gate MSE
  v
REPLACE_GATE: main ParT input D000; extra gate inputs D000 + D000
             HLT-only training forward pass and HLT-only inference
```

All new students in this diagram are freshly initialized. The already-trained
U100 anchor is retained as an evaluation-only teacher.
That anchor came from the regular `U000 -> COARSE_U050 -> COARSE_U100`
ascent. Those earlier U views progressively remove the unmatched offline
particle tail; they should not be confused with the fixed-slot D descent.

| Fit | Immediate teacher | Student main view | Student gate context | Extra gate-target loss | HLT-only deployment? |
| --- | --- | --- | --- | --- | --- |
| `PRIV_D066` | `COARSE_U100` anchor | D066 | U100 | None | No |
| `PRIV_D033` | selected `PRIV_D066` | D033 | D066 | None | No |
| `PRIV_D000` | selected `PRIV_D033` | D000 | D033 | None | No |
| `REPLACE_GATE` | selected `PRIV_D000` | D000 | Its own D000 features | Weight 0.1 | Yes |

Every stage also uses the true class labels. The privileged teachers learn
their own gates through classification and logit KD; they are **not** trained
to copy the previous teacher's gates. Explicit gate imitation is introduced
only in the final `REPLACE_GATE` fit.

When exporting a privileged teacher's targets, its original richer context is
still required. For example, exporting `PRIV_D000` requires D000 and D033.
The replacement student receives neither D033 features nor the privileged
gate in its forward pass. It receives teacher probabilities and gate factors
as detached supervision during training only.

The registered comparison campaign also contains LOGIT and SELF ladders,
`REPLACE_LOGIT`, and `SELF_REDISTILL`: **12 fits plus one aggregate job**.
The core route above needs four new fits once a valid U100 anchor exists.
The current queue helper does not offer a “four-fit route only” switch.
Pruning the DAG is a new registration, not an edit to an existing campaign JSON.

## 4. Dataset, split and particle correspondence

### 4.1 The historical population

This experiment uses Luka's genuine paired CMS FullSim **trial** ntuples,
originally under:

```text
/eos/user/l/llambrec/jetclass/output_fullsim_630k_trial
```

The historical SPORC raw container is:

```text
/home/ryreu/atlas/datasets/cms_fullsim_630k_trial_luka_v1
```

It is not the synthetic NOISE_V3/CONTEXT_V1 proxy dataset, and it is not a later
FullStat dataset. The inventory contained 890 ROOT files and 647,935 raw
entries before selections. The ALLJETS population has 374,129 eligible jets:

| Role | Jets | Use |
| --- | ---: | --- |
| TRAIN | 299,234 | Labels, teacher exports and gradient updates |
| Full validation | 74,895 | Checkpoint selection and development reporting |
| Original50k | 50,000, contained within full validation | Report-only subset, same selected checkpoints |
| Independent test | None | The historical test was explicitly retired into TRAIN |

Selections include **offline `jet_pt > 200 GeV`**, matched nonempty
offline/HLT pairs, and QCD jets only from the QCD source files. Preserve the
authenticated population's composition; do not rebalance signal/QCD or impose
new class quotas while claiming the same experiment.

The split construction is source-file-disjoint; event independence was not
established. The two validation tables are not independent experiments.
The original50k table must never select a different epoch.

### 4.2 What U100, D066, D033 and D000 actually mean

The down ladder keeps the ordered **HLT particle slots**. One frozen matching
assignment maps offline measurements onto eligible HLT slots.

| View | Meaning on those HLT slots |
| --- | --- |
| U100 | Full assigned offline measurements on matched slots; unmatched HLT slots stay HLT |
| D066 | Retain the registered `2/3` offline fraction on matched slots |
| D033 | Retain the registered `1/3` offline fraction on matched slots |
| D000 | Exact native HLT endpoint |

U100 is therefore not identical to the pure OFFLINE particle collection:
it does not append unmatched offline particles. D066/D033 are view coordinates,
not instructions to retain 66%/33% of particles.

Four-vectors interpolate according to the registered coordinate. Charge/PID
switch atomically and deterministically; tracking values and validity follow
the registered validity-aware interpolation/switching rules. Do not replace
these rules with a linear blend of every normalized feature.

Implementation:
[view construction](../src/hlt_classification/cms_proxy_ladder/views.py),
especially `match_particles`, `structural_switches`, and `build_view`.
The [D-attention contract](contracts/LUKA_D_ATTENTION.md) documents the
aligned down-ladder setting reused here.

### 4.3 Alignment is essential, not optional

For every jet used for gate supervision, slot `i` in both teacher and student
must refer to the **same constructed HLT slot**. Merely having equal particle
counts does not establish this.

Preserve:

- Ordered jet identities, labels, split membership and per-jet particle counts.
- The common slot order and mask across the D views and U100.
- The authenticated matching assignments and view-builder lineage.
- The distinction between metadata used for checking and model input tensors.

Do not independently sort each rung by its current pT, independently rematch
each rung, append offline tails, or silently truncate long jets.
Matching indices, source indices, degradation indices and row identities are
not deployable input features.

This is constructed correspondence, not a claim of generator-truth identity
between every offline and HLT reconstructed particle.

### 4.4 Feature preprocessing and units

The model consumes 17 particle features plus physical four-vectors and a mask.
The actual Luka adapter is reached through
[`luka_fullsim/stage2.py`](../src/hlt_classification/luka_fullsim/stage2.py)
and the cache preparation in
[`regular_cache.py`](../src/hlt_classification/luka_fullsim/regular_cache.py).

The 17 features comprise five normalized kinematic quantities, charge, five
PID indicators, four tracking quantities, and relative eta/phi. The kinematic
axis is computed from the **own view**, not borrowed from a richer view.
The shared
[`cms_proxy_ladder/inputs.py`](../src/hlt_classification/cms_proxy_ladder/inputs.py)
builds the basic representation. Luka uses
[`literature_context/transform.py::build_inputs`](../src/hlt_classification/literature_context/transform.py)
to override its tracking normalization:

```text
feature 11: asinh(d0 / 0.1) / 4
feature 12: log1p(d0err / 0.02) / 2
feature 13: asinh(dz / 0.2) / 4
feature 14: log1p(dzerr / 0.05) / 2
```

These tracking transforms do not clip the tails. Do not accidentally substitute
the base adapter's tanh/clipped-error tracking features. Reusing the
`literature_context` input function does **not** mean the synthetic context
degradation is applied to this FullSim dataset.

For this execution, both sides were explicitly declared **provisionally mm**;
that is an experiment convention, not producer-confirmed units. Zero
uncertainties were treated as unavailable errors, without automatically
discarding a separately valid impact-parameter value. A new dataset requires
its own declaration and audit.

Capacity is 512 particles, with no truncation and minimum padding 16.
Padding can vary with the batch. Increasing capacity or changing particle
selection requires new resource tests as well as a new input contract.

## 5. The additional gate, precisely

### 5.1 Native ParT remains in place

The existing model is the installed-Weaver Particle Transformer wrapper:
17 input channels, 11 classes, embedding dimensions `[128, 512, 128]`,
pair-embedding dimensions `[64, 64, 64]`, eight attention heads, eight particle
blocks, two class-attention blocks, GELU, and no additional hidden classifier
layers. Trimming is disabled.

The gate does not replace or separately train a second particle encoder.
It does not replace the native pair embedding, Q/K/V projections, class token
or classifier. In the cold experiment, these native parameters remain
trainable; “leave the native physics bias intact” means preserve its mechanism,
not freeze its learned weights.

### 5.2 A small token-local factor network

For particle `i`, let `x_i` be the own-view 17-vector and `c_i` its aligned
context 17-vector:

```text
[x_i, c_i]                         34 values
    -> LayerNorm(34)
    -> Linear(34, 32)
    -> GELU
    -> Linear(32, 9)
    -> (u_i[4], v_i[4], b_i[1])
```

Privileged mode uses the immediately richer `c_i`. Self mode, including
`REPLACE_GATE`, supplies `c_i = x_i`; it retains the same 34-input architecture.
“Self” does not mean no KD. It means no external context in the gate's forward
inputs.

With rank parameter four, define the FP32 pair correction:

```text
r_ij = log(4) * tanh( dot(u_i, v_j) / sqrt(4) + b_j )
```

It is directed: `r_ij` need not equal `r_ji`. “Rank four” describes the
factorized interaction before the nonlinearity; the post-tanh matrix is not
guaranteed to have matrix rank four.

Because `-log(4) <= r_ij <= log(4)`, its multiplicative effect on the
**unnormalized** attention weights lies between 1/4 and 4. Normalized softmax
weights can change differently because all keys share the normalization.

### 5.3 Where it enters attention

Conceptually, for head `h`:

```text
attention_ij^h =
    softmax_over_valid_j(
        dot(q_i^h, k_j^h) / sqrt(head_dimension)
        + native_pair_bias_ij^h
        + r_ij
    )
```

The implementation adds `r` to the existing floating attention mask/bias at
particle blocks **6 and 7, zero-indexed**: the seventh and eighth particle
blocks. The same correction is shared across heads and across those two
blocks. It does not enter class attention.

The richer view directly enters only the small gate. Nevertheless, once a
privileged gate changes attention, it changes the resulting particle features,
which influence subsequent computation. It would be incorrect to claim richer
information can never affect later Q/K/V or the class representation.

This is why `PRIV_D000` remains an **offline-dependent oracle** despite its main
input being D000.

### 5.4 Initialization and padding

The final factor projection initializes the `v` and `b` outputs to zero, while
leaving the `u` projection normally initialized. Thus the initial gate
correction is zero and the model initially has the native forward behavior,
without making the entire factor network a symmetric all-zero initialization.

Padding is sanitized before the gate's LayerNorm. Invalid pairs have zero
correction and are excluded from the auxiliary loss. Nonfinite real factors
fail closed.

The model rejects externally supplied context in `none` and `self` modes.
This is a useful deployment guard, not merely a documentation promise.

## 6. Training objectives

### 6.1 Labels plus logit KD at every new rung

Let `z_s` and `z_t` be student and selected-teacher logits. The teacher runs
in evaluation mode without gradients. With `T = 2`:

```text
q_t    = softmax(z_t / T)
L_main = 0.25 * CE(y, z_s)
       + 0.75 * T^2 * KL(q_t || softmax(z_s / T))
```

The implementation uses `batchmean` KL, computes the classification loss in
FP32, and applies `T^2` exactly once. Teacher banks store softened probabilities
at temperature two, not arbitrary raw logits or temperature-one predictions.

The code's logged `classification` loss is this **combined CE + KD term**,
not just supervised cross-entropy.

### 6.2 Additional loss for REPLACE_GATE

For a jet with `n` valid particles, center each query row over its valid keys:

```text
centered(r)_ij = r_ij - (1/n) * sum_over_valid_k(r_ik)

L_gate_for_jet =
    (1/n^2) * sum_over_valid_i,j(
        centered(r_student)_ij - centered(r_teacher)_ij
    )^2

L_gate    = mean_over_jets(L_gate_for_jet)
L_replace = L_main + 0.1 * L_gate
```

Implementation:
[`luka_gating/model.py::centered_gate_loss`](../src/hlt_classification/luka_gating/model.py).

Important details:

- The teacher target is detached.
- Loss includes all valid pairs, including diagonal pairs.
- It is normalized within each jet, then averaged across jets; large jets do
  not automatically receive weight proportional to their number of pairs.
- Unmatched-but-valid HLT slots are not silently removed.
- A one-particle jet has zero centered-gate loss.
- Centering removes a row-wise constant, which cannot affect that row's
  attention softmax.
- The coefficient 0.1 is a loss coefficient, not a guarantee of “10% of the
  gradients” or “10% of the information.”

We supervise the **resulting bounded correction**, not the nine factor
coordinates themselves. Factorizations can differ while representing the same
pair correction. We also do not match the full attention matrix, which depends
on native Q/K/V, native pair bias, head and layer.

### 6.3 What the final student sees

During every replacement training forward pass:

```text
HLT features/p4/mask -> student native ParT + student HLT-predicted gate -> logits
```

Privileged probabilities and factors are consulted only in the loss.
There is no teacher forcing that feeds the richer teacher gate directly into
the student's attention and then removes it at inference.

At deployment, retain both `mod.*` and `gate.*` checkpoint parameters.
Loading only the native ParT keys and dropping the gate is a different model.
The checkpoint should be reconstructed with `gate_mode="self"` and loaded
strictly; external context is forbidden.

## 7. Exact cold-start training schedule

| Setting | Registered value |
| --- | --- |
| New-student initialization | Fresh seeded native weights, gate weights, buffers and optimizer |
| Copied teacher parameters | None |
| Frozen-backbone epochs | Zero |
| Epochs per fit | Exactly 30 complete passes; early stopping disabled |
| Batch size | 128 |
| Gradient accumulation | 1 |
| Optimizer | AdamW |
| Peak learning rate | `3e-4` |
| Learning-rate floor | `1.5e-5` |
| Schedule | Linear warmup through epoch 3, cosine decay through epoch 30 |
| AdamW betas / epsilon | `(0.9, 0.999)` / `1e-8` |
| Weight decay | `0.01`, with the wrapper's declared no-decay parameter exception |
| Main loss | CE 0.25 + logit KD 0.75, temperature 2 |
| Gate-target weight | 0.1 only for `REPLACE_GATE` |
| CUDA forward precision | BF16 autocast; gate construction and loss arithmetic in FP32 |
| Validation | Entire 74,895-row validation population after each epoch |
| Selected checkpoint | AUC first; tie-break by lower CE, higher mean log R50, earlier update |

Each epoch visits the selected training population once without replacement.
The native wrapper excludes `mod.cls_token` from weight decay.
The learning rate is updated by fractional epoch/update position, not just once
per epoch.

Initialization and sampling seeds are inherited from the fixed30 parent's
coordinate-matched node registration. The seed derivation is
[`cms_proxy_ladder/campaign.py::paired_seed`](../src/hlt_classification/cms_proxy_ladder/campaign.py):
SHA256 of `JETCLASS2_CMS_PROXY_LADDER/v1/{domain}/{coordinate_name}`, taking the
first four bytes as an unsigned big-endian integer. Domains are
`initialization` and `sampler`. Replacement endpoints inherit D000 seeds.
The epoch permutation uses `SeedSequence([sampler_seed, epoch])`.

Use the saved node's actual seeds when reproducing an execution. Matching
seeds does not guarantee identical optimization trajectories across
architectures or hardware.

Source:
[`luka_gating_cold/training.py`](../src/hlt_classification/luka_gating_cold/training.py),
[`batch128_30e_recipe.py`](../src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py),
and [`runner.py::selection_key`](../src/hlt_classification/jetclass2_delphes/runner.py).
Fixed30 compares a fixed training budget, not demonstrated convergence.

## 8. Controls: what each comparison answers

| Row | What is changed relative to the route of interest? |
| --- | --- |
| `M0HLT` | Supervised HLT baseline; no richer teacher or additional gate |
| `DIRECT_D000` | Direct KD control from the parent's U000 teacher, without the down ladder |
| `LOGIT_D000` | Ordinary single-encoder cold down ladder; no extra gate |
| `SELF_D000` | Cold down ladder with an own-view gate, but no privileged gate inputs |
| `PRIV_D000` | Richer D033 context still available in the gate; oracle, not deployable |
| `REPLACE_LOGIT` | Same privileged teacher and final self-gated student as REPLACE_GATE, but no gate-target MSE |
| `REPLACE_GATE` | Privileged-teacher probabilities plus centered gate correction targets |
| `SELF_REDISTILL` | Extra cold fit from SELF_D000, controlling for another self-gated distillation stage |

The cleanest comparison for the value of **explicit gate imitation** is
`REPLACE_GATE` versus `REPLACE_LOGIT`, not versus `M0HLT` alone.
The latter comparison changes the teacher path, number of training stages,
and student architecture as well as the auxiliary loss.

The previous attention-KD ladder is another experiment. It matched selected
attention behavior; this method adds and then supervises a separate correction
network. The embedding-ladder experiments are also separate registrations.

## 9. Reported results and what they support

These numbers are transcribed from the user's completed-row results output,
not a fresh remote revalidation. Full validation selects the checkpoints;
all results are single-seed exploratory development results.

### Full validation

| Model | Accuracy | Macro OVR AUC | AUC recovery | Input status |
| --- | ---: | ---: | ---: | --- |
| M0HLT | 0.847573 | 0.960561 | 0.0% | HLT only |
| OFFLINE | 0.857334 | 0.964526 | 100.0% | Offline oracle |
| DIRECT_D000 | 0.848134 | 0.960913 | 8.9% | HLT only |
| LOGIT_D000 | 0.850190 | 0.961994 | 36.1% | HLT only |
| SELF_D000 | 0.850698 | 0.961920 | 34.3% | HLT only |
| PRIV_D000 | 0.852126 | 0.962922 | 59.5% | Privileged oracle |
| REPLACE_LOGIT | 0.850831 | 0.962376 | 45.8% | HLT only |
| REPLACE_GATE | 0.851939 | 0.962361 | 45.4% | HLT only |
| SELF_REDISTILL | 0.851512 | 0.962246 | 42.5% | HLT only |

Recovery uses the same-population baselines:

```text
AUC recovery (%) =
    100 * (AUC_model - AUC_M0HLT) / (AUC_OFFLINE - AUC_M0HLT)
```

It is not raw accuracy, a fraction of correctly classified jets, or a confidence
level. Values can exceed 100% or be negative.

### QCD rejection at 50% signal efficiency, full validation

| Model | X_bb | X_cc | X_gg | X_qq | X_ss |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0HLT | 409.8 | 109.8 | 25.9 | 68.7 | 63.7 |
| LOGIT_D000 | 489.6 | 126.9 | 28.7 | 76.4 | 66.6 |
| REPLACE_LOGIT | 472.9 | 133.6 | 29.4 | 81.0 | 72.5 |
| REPLACE_GATE | 564.5 | 142.2 | 30.0 | 82.8 | 71.8 |
| SELF_REDISTILL | 517.0 | 129.0 | 28.9 | 78.7 | 70.7 |

REPLACE_GATE is a reasonable strategy to investigate further:

- Versus M0HLT, accuracy improves by about **0.437 percentage points**, and
  AUC by **0.001800**.
- Versus REPLACE_LOGIT, accuracy improves by about **0.111 percentage points**,
  with higher bb/cc/gg/qq rejection in this table and slightly lower ss rejection.
- It does **not** win on macro AUC: REPLACE_LOGIT is numerically ahead by
  **0.000015**. That small difference does not establish a meaningful ranking
  without uncertainty estimates and repeated training seeds.

The original50k report-only rows were:

| Model | Accuracy | AUC | AUC recovery |
| --- | ---: | ---: | ---: |
| REPLACE_LOGIT | 0.850500 | 0.962143 | 42.0% |
| REPLACE_GATE | 0.851220 | 0.962042 | 39.5% |
| SELF_REDISTILL | 0.850640 | 0.962001 | 38.5% |

That subset also favors REPLACE_GATE's accuracy over REPLACE_LOGIT, but not
its AUC. It is not independent confirmation because the rows overlap the
selection population.

Choosing REPLACE_GATE now is a development choice about the accuracy/rejection
tradeoff, not proof it is universally strongest. Further tuning on these data
must remain labeled development; do not describe a future read of the retired
holdout as an untouched final test.

## 10. Where to find the implementation

Links below are repository-relative. The historical worktree must contain
the corresponding versions when replaying an old execution.

| Responsibility | File and entry points |
| --- | --- |
| Gate math, padding, attention injection, HLT-only context guard | [`luka_gating/model.py`](../src/hlt_classification/luka_gating/model.py): `Gate`, `log_gate`, `centered_gate_loss`, `GatedParT`, `new_model` |
| Shared rank/width/block constants | [`luka_gating/contracts.py`](../src/hlt_classification/luka_gating/contracts.py): `GATE_CONFIG` |
| Cold scientific recipe, node identities and dependency graph | [`luka_gating_cold/contracts.py`](../src/hlt_classification/luka_gating_cold/contracts.py): `recipe`, `nodes`, `tasks` |
| Fresh initialization, optimizer, losses, epoch history and selection | [`luka_gating_cold/training.py`](../src/hlt_classification/luka_gating_cold/training.py): `new_student`, `initial_digest`, `train` |
| Teacher exports, softened probabilities, ragged factors, alignment | [`luka_gating/banks.py`](../src/hlt_classification/luka_gating/banks.py): `paired`, `forward`, `export`, `Bank`, `predict` |
| Per-fit lifecycle, checkpoint reload, saved predictions, aggregation | [`luka_gating_cold/worker.py`](../src/hlt_classification/luka_gating_cold/worker.py): `teacher_refs`, `execute_task`, `validate_selected`, `completed`, `saved_results` |
| Owned local scratch, budgets, exact retirement inventory | [`luka_gating_cold/storage.py`](../src/hlt_classification/luka_gating_cold/storage.py) |
| Source compatibility and original-source parent authentication | [`luka_gating_cold/source.py`](../src/hlt_classification/luka_gating_cold/source.py) |
| Immutable gate and science specifications | [`luka_gating_cold/campaign.py`](../src/hlt_classification/luka_gating_cold/campaign.py) |
| Real resource and technical admission | [`luka_gating_cold/preflight.py`](../src/hlt_classification/luka_gating_cold/preflight.py) |
| Exact Slurm plans and journals | [`luka_gating_cold/submission.py`](../src/hlt_classification/luka_gating_cold/submission.py) |
| Optional automatic science after accepted gate | [`luka_gating_cold/followup.py`](../src/hlt_classification/luka_gating_cold/followup.py) |
| Native ParT wrapper and CE/KD math | [`jetclass2_delphes/model.py`](../src/hlt_classification/jetclass2_delphes/model.py) |
| Thin CLI, including read-only results | [`scripts/luka_gating_cold.py`](../scripts/luka_gating_cold.py) |
| Dry/live queue helper | [`scripts/queue_luka_gating_cold.sh`](../scripts/queue_luka_gating_cold.sh) |
| Slurm environment and worker launcher | [`sbatch/run_luka_gating_cold.sh`](../sbatch/run_luka_gating_cold.sh) |
| Gate mathematical/model tests | [`tests/test_luka_gating.py`](../tests/test_luka_gating.py) |
| Cold initialization, storage and lifecycle tests | [`tests/test_luka_gating_cold.py`](../tests/test_luka_gating_cold.py) |
| Follow-up and submission-safety tests | [`tests/test_luka_gating_cold_followup.py`](../tests/test_luka_gating_cold_followup.py) |

Use `luka_gating_cold/training.py`, not the warm package's training loop, for
the experiment described here. Shared `new_model` can load a teacher state,
but cold `new_student` deliberately does not pass one.

### Conceptual implementation order for a port

```text
authenticate selected teacher checkpoint and ordered training population
load teacher separately, freeze all teacher parameters, use eval mode
export TRAIN q_t at T=2
for the final replacement, also export privileged teacher gate factors
authenticate bank identities, counts, coordinates, temperature and lineage
release teacher model

construct a fresh student from its registered seed
for each of 30 complete training passes:
    shuffle TRAIN with the registered epoch seed
    run student's OWN view; external context only for registered PRIV nodes
    compute CE + logit KD
    if REPLACE_GATE:
        gather factors for exactly these jet identities and particle slots
        add 0.1 * centered correction MSE
    update student
    evaluate the entire registered validation population
    retain best state according to the declared selection key

publish selected weights, all epoch metrics and validation predictions
reload selected weights and verify prediction/metric agreement
record immutable selected-output proof
retire only this consumer's authenticated local target payloads
publish completion receipt
```

## 11. Storage and performance

### 11.1 Compact teacher targets

The dense correction is reconstructed for the current batch, not saved for
every jet. Each valid particle needs nine FP32 factors: **36 bytes per
particle**, plus identities/counts, class probabilities and metadata.
This changes persistent target storage from pairwise to linear in the number
of particles. Pair computation during training remains quadratic in sequence
length.

The factor bank uses ragged `.npy` shards, at most 4,096 jets per shard,
and a bounded four-shard memory-map cache. Shuffled training still incurs
target access and pair-loss compute. A small parameter count does not imply
zero runtime overhead.

For the cold implementation, both consumers of `PRIV_D000` export its factors;
`REPLACE_LOGIT` does not use those factors in its loss. This is the current
export policy, not a claim that logits-only distillation mathematically needs
factor targets.

### 11.2 Per-consumer local scratch

Each fit reconstructs its own teacher bank on verified node-local scratch:
`/tmp` by default, or explicitly registered `/local/scratch`.
There is no fallback to a home-directory target bank.
The teacher is released before fitting the student, so it does not remain
resident for every optimization step.

After selected outputs authenticate, only the exact owned scratch inventory
is retired. Failed jobs preserve their partial outputs; site policy may still
purge node-local storage independently. A later consumer can regenerate
targets from the retained selected teacher checkpoint.

The registered admission reserve is at least 2 GiB personally available
persistent space, plus a separately checked scratch allowance:

```text
36 * total TRAIN particles
+ 128 * TRAIN rows
+ 3 GiB reserve
```

This is specific to this campaign and its checks, not a universal memory
estimate for another dataset.

### 11.3 What survives per scientific fit

```text
science/train_REPLACE_GATE/
    selected.pt
    training_report.json
    validation_predictions.npz
    teacher_bank.json
    selected_outputs.json
    retired_targets/
        intent.json
        receipt.json
    receipt.json
```

Other fits follow the same layout. `training_report.json` contains every
epoch's metrics and the selected epoch. `validation_predictions.npz` contains
identities, labels, particle counts, probabilities and original50k membership.
`teacher_bank.json` is retained provenance for retired targets, not a promise
that its former scratch files still exist.

There are no durable per-epoch checkpoints or optimizer snapshots. The
training loop retains its best state in memory and publishes selected outputs
after the fit. An interrupted job is not automatically resumable from the
last epoch.

Keep selected checkpoints, predictions, reports and their parent lineage.
They enable future diagnostics such as R30 from saved predictions without
retraining. A printed R50 table alone cannot determine R30.
Do not delete the U100 anchor, its required lineage, shared input caches,
raw data, or source worktrees to save the relatively small selected-model
storage. Nothing in this lifecycle touches `Fresh_check`.

## 12. Reproducing the historical setup

### 12.1 Read the existing results without submitting anything

The following block reads the historical cold run. It does not retrain or
submit jobs. It assumes the historical paths still exist.

```bash
bash <<'BASH'
set -euo pipefail
ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/luka_cold_gate_a1d3a2c6_r1
SPEC="${ROOT}/science/campaign_spec.json"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
test -x "${PYTHON}"
test -f "${SPEC}"

PROJECT="$("${PYTHON}" -B -s -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
COMMIT="$("${PYTHON}" -B -s -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "${SPEC}")"
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"

source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_gating_cold.py" results --spec "${SPEC}"
BASH
```

Reading the two fields above only locates the code; the `results` implementation
performs the artifact authentication. Missing dependencies or failed hashes
are not a reason to skip validation and trust a directory's presence.

### 12.2 A new execution of the unchanged design

Use the [existing runbook](LUKA_COLD_GATE_TIER3_RUNBOOK.md), with:

1. The exact clean, pushed source that implements the chosen frozen recipe.
2. The authenticated fixed30 parent with selected U100 and its required
   foundation/prepared artifacts and comparison controls.
3. A **new, unused output root**, never the historical root above.
4. A real declaration of personally available persistent quota and node-local
   scratch.
5. A dry plan review, followed by explicit authorization of that exact hash.
6. A successful real preflight before science.

The helper interface is:

```text
bash scripts/queue_luka_gating_cold.sh \
    COMMIT ROOT PARENT AVAILABLE_GIB gate|auto|science [--execute PLAN_HASH]
```

Without `--execute`, it reviews the plan without submitting jobs; specification
creation can still write new metadata. `auto` reviews an outer follow-up plan
which, when explicitly executed, queues one preflight and submits the
registered 13-job science DAG only after verified success. Manual `gate`
followed by manual `science` is an alternative, not an additional submitter.
Wait for `ARMED` when using the automatic controller.

The registered site is **SPORC tier3**, account `reu-aisocial`, QoS `qos_tier3`,
one `gpu:a100:1`, six CPUs, and 128,000 MiB host memory. Gate walltime is eight
hours. Science fit walltimes are individually measured with a 24-hour ceiling;
aggregate requests one CPU, 16,384 MiB and one hour.
This is not an Oscar/Tigris/debug switchable command.

The worker sets `PYTHONNOUSERSITE=1`, prepends the conda library directory,
uses absolute project paths, and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`
**before Python starts**. Preserve those settings.

The cold source checker pins its compatibility base and restricts allowable
scientific changes. Blindly using the newest repository HEAD can fail if
unrelated scientific modules changed. Use the historical clean pin for an
unchanged replay, or explicitly reconcile and test a new integration source.
Never bypass the guard or modify an old artifact hash to make it load.

### 12.3 What preflight must establish

The cold preflight exercises:

- Native/gated parity in FP32 and BF16, including the initial zero correction.
- Reproducible cold initializations with no teacher parameter inheritance.
- Authenticated TRAIN/validation slot alignment.
- All 12 technical one-pass fits and 24 maximum-support stress cases:
  training/backward and validation/forward at every registered node.
- Selected-checkpoint reload and predictions.
- Local scratch budgets, persistent output budgets, host/GPU resource bounds.
- Owned-target retirement and retained reporting without erased payloads.

Technical preflight checkpoints are not scientific teachers. Good or bad
classification scores never determine technical admission; corrupt inputs,
nonfinite required values, forbidden context, lineage errors and resource
failures do.

## 13. Reproducing the idea in a slightly different setting

There are two distinct goals: **replay this frozen execution**, or **port the
method**. A port preserves the central idea but needs its own explicit
scientific and artifact identity.

### 13.1 A useful minimum specification for a new port

Before implementing or queueing, write down:

1. Dataset identity, selection, units, label map and allowed richer-view use.
2. TRAIN/validation/test policy, including prior use of any proposed holdout.
3. Exact native HLT endpoint and aligned view construction.
4. Anchor teacher identity and how its checkpoint was selected.
5. Privileged chain and final replacement; warm or cold initialization.
6. Student architecture, gate rank/width/bound/placement, losses and schedule.
7. Matched controls and the primary metric used for checkpoint selection.
8. Storage lifecycle, source compatibility, resource measurement and site.

Do not change the meaning of existing `LUKA_COLD_GATE_* /v1` artifacts.
Use a new version or campaign namespace and extend tests accordingly.
The present CLI has no generic flags for arbitrary counts, class maps,
gate weights or a shortened chain.

### 13.2 What can be reused, and what must be reconsidered

| New setting | Reuse | Required new work |
| --- | --- | --- |
| Same frozen data/recipe, another seed | Gate model/loss and storage design | Register paired seeds for anchor/controls/students, fresh source-bound spec and outputs; report seed uncertainty |
| More/fewer jets or a new split | Gate mathematics | New authenticated population/split, train-only preparation and teacher exports; no reuse of old probability banks; new preflight |
| New paired FullSim data | Gate architecture if feature semantics match | Schema/units/PID/selection audit, new matching and lineage, new teachers and resource checks |
| Different number of classes | Gate mathematics | Output head, label map, probability-bank dimensions and metric validation; several current checks assume 11 classes |
| Different particle feature count | Attention-correction idea | Replace the hardcoded 17+17 gate input and its normalization; new model/schema and tests |
| Different number of transformer blocks/heads | Factorized correction idea | Explicit placement and supported attention implementation; current code requires eight particle blocks and hooks blocks 6/7 |
| Different GPU, environment or scheduler partition | Scientific recipe, if deliberately kept | Supported execution-site registration, installed-Weaver/backend parity, scratch validation and measured admission |
| Warm starts instead of cold starts | Shared gate/loss | Separate initialization contract and matched controls; do not call it this cold experiment |
| Changed rank, bound, gate weight or freeze schedule | Code structure | Versioned recipe, fair controls, new technical/resource evidence |

For a modest first port, keep rank four, width 32, `log(4)` bound, blocks 6/7,
temperature two and gate weight 0.1 while changing only the necessary
dataset/site registration. This preserves interpretability of the comparison;
it is not a claim those hyperparameters are optimal elsewhere.

### 13.3 If the new views do not have particle correspondence

Do not simply zip their particles or pad them to the same length. The current
teacher gate is defined on the teacher's pair indices. A student pair target
needs an authenticated correspondence or an explicitly registered transport
onto the student's slots.

Possible scientific redesigns include constructing all intermediate views on
native HLT slots, as here, or defining and validating a new matching/transport
method. Unmatched particles and ambiguous matches need an explicit rule.
Until that exists, logits-only supervision remains meaningful at jet level,
but this particular pairwise gate loss is not well-defined.

### 13.4 If the new anchor is pure OFFLINE

The current U100 anchor already uses HLT-aligned slots. A pure OFFLINE teacher
can still supply jet-level probabilities, but its particle gate cannot
automatically supervise HLT pair indices. Do not relabel pure OFFLINE as U100.
Construct the intended aligned view and train/select its anchor, or register
a different transport-based experiment.

### 13.5 Controls worth preserving

At minimum retain a matched ordinary HLT/logit-KD reference and the
`REPLACE_LOGIT` ablation, since the latter isolates the extra gate loss.
`SELF_REDISTILL` helps distinguish privileged supervision from the benefit
of another training stage. Preserve the privileged oracle row to show how
much of its behavior is lost in the HLT-only replacement.

Use identical validation identities and checkpoint-selection rules.
If latency or deployable simplicity is important, measure the self-gate's
inference cost against native ParT as well as reporting classification.
Repeated seeds and a genuinely unused evaluation population are needed for
stronger generalization claims; the historical ALLJETS split supplies neither.

## 14. Testing and failure checklist

Focused source tests:

```text
python -B -m pytest -q \
    tests/test_luka_gating.py \
    tests/test_luka_gating_cold.py \
    tests/test_luka_gating_cold_followup.py
```

Use the repository's documented environment with the appropriate Weaver
installation and `src` import path. The
[testing ladder](TESTING.md) still applies; local tests do not replace real
installed-backend, full-population site acceptance.

For a port, specifically test:

- Initial zero-gate/native parity; gradients reach the gate.
- Only the registered particle blocks receive the extra correction.
- Padding, including poisoned padding, cannot affect valid factors or losses.
- External richer context is rejected by the final self-gated student.
- Swapped jet identities, counts, views or factor shards fail closed.
- Consistent particle permutations transport pair targets correctly.
- Centered gate loss ignores row-wise offsets and teacher gradients.
- Cold initialization is independent of teacher checkpoint contents.
- Teacher probabilities have the expected temperature and class map.
- Saved selected weights reproduce saved validation probabilities.
- Scratch retirement cannot delete shared inputs, other consumers or unknown files.
- Reporting remains valid after intentional scratch retirement.
- No subset-selected checkpoint or consumed-test claim appears in the report.

Do not “fix” an implementation failure by truncating particles, skipping
scientific rows, silently switching attention backends, deleting parent
evidence, or changing the batch/schedule inside an immutable execution.

### Documentation verification for this handoff

The local focused checks for graph registration, bounds/centering/gradients,
native gate insertion, ragged-bank integrity, cold/tier3 registration, and
teacher-independent initialization passed: **6 passed, 36 deselected**.
They used the existing local Weaver dependency and a fresh test directory.
Initial attempts encountered local import/temporary-directory access issues;
these were resolved without changing runtime source.
Repository-relative file links, code-fence balance, whitespace, and the Bash
results block (including the heredoc body) were also checked. The Bash block
was syntax-checked only, not executed against the cluster.

This documentation update does not claim a new cluster validation, new fit,
remote artifact authentication, commit, push or job submission.

## 15. Short handoff to the next implementer

> Port the **cold REPLACE_GATE** route, not the warm or staged replacement.
> Start from an authenticated aligned U100 anchor. Train fresh privileged
> D066, D033 and D000 teachers with labels plus logit KD, allowing only their
> extra bounded gate to consume the immediately richer aligned view. Train
> a fresh D000 self-gated student from the selected PRIV_D000 teacher using
> the same label/logit loss plus 0.1 valid-key-centered correction MSE.
> Keep the student gate at HLT-only inference. Preserve matched controls,
> validation selection, source/data lineage and per-consumer scratch retirement.
> Any new dataset, split, initialization or execution site gets an explicit
> new registration and measured acceptance; do not edit the historical specs.

# `smooth_w10`: offline-neighborhood supervision, code and porting handoff

Written 2026-10-28 against implementation commit
`62f9773aabd150004be9c319d6ed06c412fcc57a`.

This is an explanation of the implemented experiment, not a new scientific
recipe or evidence that it improves recovery. At writing, no completed
`smooth_w10` results have been supplied. The latest confirmed remote event was
a storage-admission failure before job submission; restart instructions were
provided, but successful submission has not been confirmed in this handoff.

Authority: [active plan](plans/LUKA_NEIGHBORHOOD_PLAN.md),
[v1 contract](contracts/LUKA_NEIGHBORHOOD.md), and the execution's immutable
`campaign_spec.json`. The [full scientific design](LUKA_OFFLINE_NEIGHBORHOOD_STRATEGY_HANDOFF.md)
describes the fourteen-arm comparison; the
[runbook](LUKA_NEIGHBORHOOD_RUNBOOK.md) owns release and queue instructions.
This document focuses on one arm and how another chat can understand or port it.

## 1. The idea in plain language

Train a normal HLT-only Particle Transformer to classify jets. Keep the existing
successful training-only logit and jet-embedding predictors. Add a small head
to each HLT particle's contextual representation and ask:

> Around this HLT particle's direction, what particle composition and transverse
> momentum would the offline reconstruction see, at three angular scales?

The answers are explicit physical summaries computed from paired OFFLINE
particles, not vectors produced by another neural teacher. They give the encoder
local supervision in addition to the existing global jet-level supervision.

The new head predicts **36 numbers per real HLT particle**. `smooth` means
neighbors contribute through Gaussian distance weights. `w10` means the
**total coefficient of the averaged neighborhood loss is 10**, not 10 per
particle, category, channel, or neighborhood scale.

The model never receives offline particles as forward inputs. Auxiliary
predictions are not fed into classification. At inference all auxiliary heads
are removed, leaving the original HLT-only classifier.

This is not unordered learned-token prediction, attention distillation, fusion,
SupCon, particle matching, or a new teacher ladder. The targets can include
offline particles with no HLT counterpart, and the two views need not have
equal particle counts. The query locations are always the native HLT particles.

## 2. What stays fixed from the successful baseline

The base is `k1_e40_plain`, with the repeated depth-screen `baseline` as the
primary comparison. Building on it means **the same recipe from scratch**, not
loading its selected checkpoint.

| Component | Exact setup |
| --- | --- |
| Deployable input | Native D000 HLT; 17 features, four-vectors and mask; 11 classes |
| Backbone | Existing installed-Weaver ParT adapter; no new attention operation |
| Initialization | Fresh, with the inherited baseline initialization and sampler seeds |
| Training population | 299,234 TRAIN jets |
| Validation | 74,895 jets; original50k is an overlapping, report-only subset |
| Schedule | All 30 epochs, physical batch 128, no early stopping |
| Optimizer | Inherited AdamW, original parameter grouping and optimizer settings |
| Learning rate | Three-epoch warmup to `3e-4`, cosine decay to `1.5e-5` at epoch 30; updated per batch |
| Main head | CE only, coefficient **0.5** |
| Auxiliary logit heads | Four separate heads, total KD coefficient **1**, temperature **2** |
| Auxiliary jet heads | Four separate 128-D predictors, total embedding coefficient **40** |
| New neighborhood head | Block 8, `128 -> 128 -> GELU -> 36`, total coefficient **10** |
| Excluded extras | No relational loss, learned-token target loss, teacher-diversity addition, or warm start |

The four frozen teachers, in registered order, are original `OFFLINE` and
`UE100`, `DE066`, `DE033` from the completed **c50 hybrid ladder**. The latter
are not the older LOGIT-only D066/D033 models. The compact targets are borrowed
from the authenticated original depth campaign; no teacher is retrained or
re-exported by this neighborhood campaign.

Each teacher supplies its actual native 128-dimensional pre-classifier jet
embedding, standardized using its TRAIN-only statistics, and its probabilities
at temperature 2. Do not substitute its embedding-predictor output. These
existing jet targets use a scale floor of `1e-3`; the new neighborhood targets
use a different floor, `0.1`.

This Luka trial has **no independent test**. Its historical split convention
must not be generalized to another dataset: preserve a genuine sealed test
when porting to a setup that has one.

## 3. Where the supervision attaches

```text
HLT features + HLT four-vectors + mask
    -> native particle blocks 1..8
         |-- block-8 tokens [B, N, 128]
         |      -> shared neighborhood MLP -> [B, N, 36] -> neighborhood loss
         |
         -> native class-attention pooling -> jet vector h [B, 128]
                |-- native class head -> [B, 11] -> CE (inference output)
                |-- four jet predictors -> 4 x [B, 128] -> embedding losses
                `-- four auxiliary logit heads -> 4 x [B, 11] -> KD losses
```

The block-8 tap is **after the eighth particle block and before class-attention
pooling**, not after the entire network. It captures the output of
`model.mod.blocks[7]` during the original native forward. The jet tap captures
the input to `model.mod.fc`. Hooks do not replace the forward implementation.

Input shapes are `features: [B,17,N]`, `vectors: [B,4,N]`, and boolean
`mask: [B,1,N]`. The tap returns tokens in `[B,N,128]` layout and checks that
support has not changed. The adapter disables Weaver sequence trimming and
pads to the batch's actual maximum support; do not silently truncate particles.

The new MLP is shared across particles and initialized with seed `94271` in an
isolated CPU RNG context. Adding it must not change the backbone, existing
head, dropout or sampler random streams. Auxiliary operations/losses run in
FP32; the native CUDA forward retains the inherited BF16 policy.

All auxiliary gradients reach the shared encoder. The neighborhood gradient
enters at block 8; it does not backpropagate through the downstream pooling or
main classification head. The existing jet-level auxiliary losses do reach
the shared pooled representation. The parameters of the final main classifier
receive exactly the `0.5 * CE` gradient, not KD directly.

## 4. Exact smooth target construction

### Physical inputs and geometry

For a paired TRAIN jet, use native HLT particles `H` as queries and native
OFFLINE particles `O` as context. Read the authenticated physical four-vectors,
not normalized model feature columns or separately recentered coordinates.

```text
pt = hypot(px, py)
eta = asinh(pz / pt)
phi = atan2(py, px)
delta_phi = wrap(phi_offline - phi_hlt) into [-pi, pi)
d(i,k)^2 = (eta_offline[k] - eta_hlt[i])^2 + delta_phi(i,k)^2
Q = sum of pt over all real HLT particles in this jet
```

Both views must use a common angular frame. Reject invalid/nonfinite geometry
or nonpositive required pt; do not drop those particles to make the run pass.
The current validated support is 1..512 real particles per view. Porting to
another support requires explicit validation, not clipping to this range.

Category order is fixed:

```text
0 charged hadron
1 neutral hadron
2 photon
3 electron
4 muon
5 other/unknown
```

These are reconstructed categories, not truth ancestry labels. Unknown is a
real category, not padding. Include all accepted offline particles, including
unmatched ones and any counterpart of the query. There is no self/counterpart
exclusion, Hungarian matching, nearest-k selection, or label-based selection.

### Gaussian neighborhoods

For `sigma_s = [0.02, 0.05, 0.10]` in delta-R units:

```text
K_s(d) = exp(-0.5 * (d / sigma_s)^2)

C[i,s,c] = sum_k K_s(d(i,k)) * 1[category_offline[k] == c]
F[i,s,c] = sum_k K_s(d(i,k)) * 1[category_offline[k] == c] * pt_offline[k] / Q
```

`C` is a weighted count and may be fractional. `F` is a scalar-pt contribution
relative to total HLT scalar pt, not a probability; it need not sum to one or
stay below one. Do not normalize the Gaussian weights over neighbors or apply
a softmax. There is no hard radial cutoff; the three neighborhoods overlap.

Transform each channel:

```text
x_count = log1p(C)
x_pt    = log1p(100 * F)
```

Flatten **all 18 count channels first**, then **all 18 pt channels**, with
scale-major/category-minor ordering inside each half:

```text
count_index = 6 * scale_index + category_index
pt_index    = 18 + 6 * scale_index + category_index
```

This is `3 scales * 6 categories * 2 quantities = 36` values. They are
correlated, not 36 independent labels. The factor 100 is a fixed transform
choice, not a fitted detector calibration.

Geometry and sums use FP64; transformed descriptors are stored as FP32. Context
particles are canonically ordered before reduction for numerical reproducibility.
Query order is never changed. The implementation chunks queries by 32, avoiding
a dataset-wide pair-distance tensor.

### TRAIN-only normalization

For each of the 36 channels, fit moments on stored TRAIN descriptors with
**equal weight per jet**, not equal weight per particle:

```text
mu[c] = mean_j( mean_i_in_jet_j(x[j,i,c]) )
var[c] = mean_j( mean_i_in_jet_j((x[j,i,c] - mu[c])^2) )
scale[c] = max(sqrt(var[c]), 0.1)
z[j,i,c] = (x[j,i,c] - mu[c]) / scale[c]
```

Accumulate moments in FP64. Keep empty-neighborhood zeros as valid supervision;
exclude padding only. No validation-fitted statistics, per-batch normalization,
target clipping, per-particle LayerNorm, or renormalization of the residual.

The `.npy` payload stores transformed `x`, not standardized `z`.
`NeighborhoodBank.batch(..., kind='smooth')` applies the saved moments at read
time and returns padded FP32 targets in `[B,N,36]` shape.

## 5. Exact loss, including what the weights mean

For four teachers `t`, let `q_t` be cached temperature-2 probabilities, `a_t`
the student's separate auxiliary logits, `g_t(h)` its jet-embedding prediction,
and `z_t` the standardized frozen teacher embedding:

```text
L = 0.5 * CE(main_logits, labels)
  + (1/4) * sum_t [4 * KL(q_t || softmax(a_t / 2))]
  + (40/4) * sum_t SmoothL1(g_t(h), z_t)
  + 10 * L_neighborhood
```

The KD temperature-squared factor `4` is applied once. The teacher probabilities
are already softened; do not soften them again. KD uses `batchmean` (sum over
classes, mean over jets), with teacher-to-student KL direction. Jet SmoothL1
uses `beta=1` and averages over the batch and 128 coordinates.

For predicted neighborhood `u[j,i,c]` and target `z[j,i,c]`:

```text
L_neighborhood = mean_j(
    mean_real_HLT_particles_i_in_j(
        mean_36_channels_c SmoothL1(u[j,i,c], z[j,i,c], beta=1)
    )
)
```

Thus a high-multiplicity jet does not get more total loss weight. Targets are
detached, padded entries contribute nothing, and every jet must have at least
one real particle. The code returns both a scalar and diagnostic family means.
For `smooth_w10`, training logs show `neighborhood_count` weighted **5** and
`neighborhood_pt` weighted **5**, because their equal average is multiplied by
10. This is not a total weight of 20.

The coefficients are not percentages of gradient contribution. Do not divide
the whole objective by its coefficient sum or retune the LR as part of a claimed
exact reproduction. Weight 10 here is not equivalent in gradient scale to jet
weight 10 or the older learned-token weight 10.

## 6. Code map and reading order for another chat

All paths below are repository-relative links. Read functions, not just file
names: much of the established baseline is deliberately reused.

| File | Relevant entry points / purpose |
| --- | --- |
| [Neighborhood contracts](../src/hlt_classification/luka_neighborhood/contracts.py) | `ARMS`, `recipe('smooth_w10')`, `target_recipe()`, `tasks()`; frozen experiment identity |
| [Physical targets](../src/hlt_classification/luka_neighborhood/targets.py) | `geometry`, `descriptors(..., smooth=True)`, `Moments`; exact equations above |
| [Target banks](../src/hlt_classification/luka_neighborhood/banks.py) | `physical_rows`, `required_bytes`, `build`, `NeighborhoodBank`, `Bundle`; joins, storage, normalization, borrowed targets |
| [Student and loss](../src/hlt_classification/luka_neighborhood/model.py) | `Student`, `new_model`, `neighborhood_loss`, `native_state` |
| [Training](../src/hlt_classification/luka_neighborhood/training.py) | `objective`, `batch_loss`, `train`, `expected_weights`, `validate_report` |
| [Inherited baseline objective](../src/hlt_classification/luka_supervision_depth/training.py) | `objective`; CE and four independent KD/jet losses |
| [Native particle tap](../src/hlt_classification/luka_supervision_depth/model.py) | `tapped_forward`; block-8 layout and full-support checks |
| [Native jet tap](../src/hlt_classification/luka_embedding_aux/model.py) | `head_input`; pre-classifier hook |
| [Existing auxiliary heads](../src/hlt_classification/luka_auxiliary_grid/model.py) | `AuxiliaryLogitParT`; four 128-D MLPs and four linear logit heads |
| [Existing optimizer](../src/hlt_classification/luka_auxiliary_grid/training.py) | `optimizer`, `finite_gradients`; inherited AdamW grouping |
| [Native model adapter](../src/hlt_classification/jetclass2_delphes/model.py) | `DelphesParticleTransformer`, `model_config`; 17-input/11-output, trim disabled |
| [Fixed30 schedule](../src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py) | `learning_rate`; exact per-update schedule |
| [Reference target export](../src/hlt_classification/luka_trial_hybrid/banks.py) | Existing `export`/`load`; actual teacher embeddings and T=2 probabilities |
| [Original raw conversion](../src/hlt_classification/luka_fullsim/particles.py) | `physical`, `identity_categories`; physical frame and reconstructed PID conventions |
| [Physical particle container](../src/hlt_classification/cms2jc2_response/bridge.py) | `Particles`; validated p4/category interface used by target builder |
| [Original-source import](../src/hlt_classification/luka_neighborhood/source.py) | `current`, `import_parent`, `validate_import`; exact old source, completed targets/baseline |
| [Campaign registration](../src/hlt_classification/luka_neighborhood/campaign.py) | `policy`, `create`, `check`; fresh immutable spec, resource/storage declaration |
| [Submission](../src/hlt_classification/luka_neighborhood/submission.py) | `render`, `submit`; dry plan, exclusive live claim, exact-ID DAG |
| [Worker](../src/hlt_classification/luka_neighborhood/worker.py) | `run`, `fit`, `validate_task`, `results`; production path, native readback and reporting |
| [In-job checks](../src/hlt_classification/luka_neighborhood/startup.py) | `run`, `check`; disposable parity, gradients, memory and timing probes |
| [Retention](../src/hlt_classification/luka_neighborhood/retention.py) | Explicit reviewed retirement of three owned payloads, not borrowed data |
| [Tests](../tests/test_luka_neighborhood.py) | Hand calculations, masks, normalization, objective, parity, worker and retirement tests |

Important nuance: the recipe retains inherited `depths=[]` because it has **no
learned teacher-token prediction objective**. Its separate
`neighborhood_block=8` still attaches the physical-summary head at block 8.

## 7. Using the Python surfaces without guessing the interfaces

### Inspect the registered recipe (read-only)

In an environment with this repository on `PYTHONPATH`:

```python
from hlt_classification.luka_neighborhood.contracts import recipe, target_recipe
from hlt_classification.luka_neighborhood.training import expected_weights

r = recipe('smooth_w10')
assert r['neighborhood'] == 'smooth'
assert r['neighborhood_weight'] == 10.0
assert r['ce_coefficient'] == 0.5
assert r['logit_total'] == 1.0 and r['embedding_total'] == 40.0
print(target_recipe())
print(expected_weights('smooth_w10'))
```

### Compute descriptors from decoded physical particles

This is a library sketch, **not a production export script**. `train_pairs`
must be a reiterable collection or reader of authenticated `(HLT, OFFLINE)`
`Particles` objects in native HLT order; labels are not inputs to the function.

```python
import numpy as np
from hlt_classification.luka_neighborhood.targets import descriptors, Moments

moments = Moments()
for hlt, offline in train_pairs:
    x = descriptors(hlt, offline, smooth=True)  # FP32 [n_hlt, 36]
    moments.add(x)
mean, scale = moments.result()                 # FP64 [36], TRAIN only

# Later, for a TRAIN jet, using these same frozen moments:
x = descriptors(hlt, offline, smooth=True)
z = ((x.astype(np.float64) - mean) / scale).astype(np.float32)
```

Production uses `banks.physical_rows` to verify ordered jet identities, labels
and exact reconstructed HLT feature/vector bytes against the existing D000
cache. Use `banks.build` rather than the sketch for provenance and atomic
publication. Do not normalize each jet independently.

### Add the loss to an existing baseline

This sketch assumes a correctly seeded `Student` and authenticated caches/bank.
The production entry is `training.batch_loss` or `training.train`; it already
adds all losses, so do not add the neighborhood term again around it.

```python
from hlt_classification.luka_neighborhood.model import neighborhood_loss

student.capture = True
main_logits = student(features, vectors, mask)
prediction = student.neighborhood_predictions  # [B, N, 36], connected

# target_z is the detached, TRAIN-standardized tensor for these exact queries.
local_loss, (count_loss, pt_loss) = neighborhood_loss(prediction, target_z, mask)
total_loss = baseline_loss + 10.0 * local_loss
total_loss.backward()
```

`baseline_loss` must be the CE0.5/K1/E40 objective from this document, not CE
alone. Do not detach the particle features or run the capture forward under
`no_grad`; that would prevent the new supervision from shaping the encoder.

For validation set `student.capture=False` as well as `student.eval()`; `eval()`
alone does not switch off auxiliary capture. For deployment, authenticate the
selected checkpoint, then load its native-only state strictly into
`DelphesParticleTransformer`. `Student.native_state()` filters out `predictors.*`,
`logit_heads.*`, and `neighborhood_head.*`. The production worker verifies that
the restored native model reproduces the selected validation probabilities.

## 8. Running it in this repository

The existing [queue helper](../scripts/queue_luka_neighborhood.sh) launches the
**entire registered screen**, not only `smooth_w10`: one shared preparation job,
fourteen student fits and one report, all on debug. There is no standalone
preflight, new teacher fit, or baseline rerun. The first two fits are
`smooth_w3` and `smooth_w10`, after shared preparation succeeds.

Use the [runbook](LUKA_NEIGHBORHOOD_RUNBOOK.md) from a clean pinned worktree at
the full pushed implementation commit. It invokes:

```bash
bash "${PROJECT}/scripts/queue_luka_neighborhood.sh" \
    /home/ryreu/atlas/HLT_Classification/checkpoints/luka_depth_9e90b7de_debug_r1/campaign_spec.json
```

`PROJECT` must be that exact worktree, not the main working checkout. Review
the full dry plan, paste its actual 64-character `PLAN_HASH`, and separately
authorize live submission. Do not run the helper again on an existing campaign
root. Confirm the final sixteen-job ledger before assuming submission finished.

The current confirmed implementation paths are:

```text
Commit: 62f9773aabd150004be9c319d6ed06c412fcc57a
Project: /home/ryreu/atlas/HLT_Classification_neighborhood_62f9773a
Study: /home/ryreu/atlas/HLT_Classification/checkpoints/luka_neighborhood_62f9773a_debug_r1
Parent: /home/ryreu/atlas/HLT_Classification/checkpoints/luka_depth_9e90b7de_debug_r1/campaign_spec.json
```

These are provenance/context, not proof of accepted jobs. The failed 22-GiB
creation wrote only `campaign_spec.json`. The exact free-space check showed
21.398438 GiB available versus 21.999791 GiB required. A reviewed archive/restart
with an 18-GiB declaration was proposed; do not assume it was executed.

The inherited guard in
[embedding-weight worker](../src/hlt_classification/luka_embedding_weight/worker.py)
checks `free >= storage_budget_bytes - campaign_bytes_used`. Its hard-coded
"2 GiB" error is misleading for this campaign. `df -h` rounds, and a declaration
does not actually reserve filesystem space. Recheck real availability and leave
headroom for other jobs. Never edit a saved spec's budget/hash in place.

Execution uses `atlas_kd_sporc`, `PYTHONNOUSERSITE=1`, no bytecode, and
`${CONDA_PREFIX}/lib` prepended to `LD_LIBRARY_PATH`. The registered GPU envelope
is one A100, six CPUs, 90000 MiB and 12 hours per GPU task; the report is CPU-only.
Actual new-objective checks run inside the authorized workers and are discarded
before fresh training. Do not shorten the time limit manually: exact allocation
identity includes the registered time limit.

There is **no public single-arm submit option**. Although the thin
[CLI](../scripts/luka_neighborhood.py) has `run --task smooth_w10`, it is a
production-worker entry that requires an accepted campaign-bound Slurm job.
It is not a login-node shortcut or a way to bypass the target dependency.
A new single-arm campaign needs a separately scoped spec/plan, authorization,
ownership and tests; do not mutate this registered fourteen-arm campaign.

### Reading results

In the pinned worktree's established conda/PYTHONPATH environment, with `ROOT`
set to the exact accepted study root:

```bash
python -u -B -s "${PROJECT}/scripts/luka_neighborhood.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

This authenticates completed rows and explicitly reports incomplete ones. A
completed epoch in a live log is provisional until publication/validation
finishes. The exact log path comes from the submission ledger's output command;
the filename for this arm is `smooth_w10-JOBID.out`.

Relevant retained outputs:

```text
ROOT/targets/neighborhoods/{manifest.json,index.npz,smooth.npy,ann.npy,hlt_ann.npy}
ROOT/targets/references/BASELINE_REPEAT.npz
ROOT/targets/baseline_report.json
ROOT/smooth_w10/{bundle.json,startup.json,costs.json,receipt.json}
ROOT/smooth_w10/fit/{selected.pt,training_report.json,validation_predictions.npz,receipt.json}
ROOT/report/results.json
```

`training_report.json` retains all 30 `validation_history` rows, raw/weighted
losses, 36 channel-error diagnostics, `delta_repeat`, `recovery_percent`, and
the selected epoch. Training logs also print current-epoch completion percentage
every 128 batches. Selection is main-head full-validation macro OVR AUC, then
lower CE, then the registered QCD-rejection tie-breaker, then earlier update;
it is not selected on neighborhood error or original50k.

Reference values supplied by the user (use authenticated full precision in
reports) are:

```text
M0HLT:              0.96056122
OFFLINE:            0.96452569
K1_E40_PLAIN:       0.96315997  (~65.6% recovery)
Repeated baseline:  0.96315546  (~65.4% recovery)

recovery_percent = 100 * (AUC - M0HLT) / (OFFLINE - M0HLT)
delta_repeat = AUC - repeated_baseline
```

The new neighborhood arm has no measured gain recorded here. Comparing many
weights on the same validation set is exploratory; matched-seed confirmation
is needed before treating a small winning difference as reliable.

## 9. Copying or adapting it to another setup

There are two distinct tasks; do not mix their provenance claims.

**Exact reproduction here:** use this repository and its authenticated parent
artifacts at the pinned source. The new package intentionally depends on the
existing scientific runtime, target banks and validators. Copying just its
folder is not a standalone reproduction. `source.current` also rejects modified
historical scientific runtime; do not weaken that guard to force a port through.

**Porting the idea:** take the scientific pieces into a new versioned experiment:

1. Implement an adapter returning physical HLT/OFFLINE particles for the same
   TRAIN jet in a common frame. Preserve canonical jet IDs and the model's
   particle ordering. Adapt reconstructed categories explicitly if schemas
   differ; do not assume normalized inputs are physical coordinates.
2. Reuse or port `targets.descriptors` and `Moments` first. Their core is NumPy;
   the current `Particles` container enforces additional p4/PID/validity rules.
   Replace the package-level contract import deliberately if extracting these
   functions, and document any semantic changes.
3. Reuse the equal-jet masked SmoothL1 reduction from `model.neighborhood_loss`.
   Fit TRAIN moments once, cache compact targets, and join them by exact jet
   identity and query order. Preserve all real particles and genuine zeros.
4. Attach a small shared MLP to the final particle-token layer before pooling.
   For a different token width `D`, an analogous `D -> D -> GELU -> 36` is a
   **new adaptation**, not identical to the 128-D reference. Verify tensor
   layout, masks and any sort/trim operations before trusting the targets.
5. Keep the destination's established baseline and training budget fixed for
   the first ablation. The complete reference adds `10 * L_neighborhood` to
   CE0.5/K1/E40; adding it to CE-only is a different, valid question that must
   be named separately. Do not silently substitute teachers in a claimed
   K1/E40 reproduction.
6. Export native-only inference weights and compare them with the auxiliary
   wrapper's capture-disabled predictions. Prove that no offline context,
   target statistic, auxiliary output or construction index is an input to
   the deployed classifier.
7. Give the new experiment its own contracts, source hashes, split/teacher
   lineage, allocation checks, dry/live plan and owned output namespace.
   Do not copy old receipts, claim directories or scheduler IDs as evidence.

For a port, another model's token width, angular scales, label space, particle
acceptance, batching, loss reduction and data volume can all affect the useful
weight. Start by reproducing the definition, then register deliberate changes.
Do not promise that coefficient 10 transfers unchanged in effectiveness.

### Storage and computation

One smooth FP32 bank needs `144 * S` payload bytes for `S` TRAIN HLT particles.
The current full screen constructs **three** families once, or `432 * S`, plus
indexes, moments, outputs and reserve. All fourteen arms share these banks;
they are not fourteen copies. Do not budget this as a one-bank campaign merely
because this document explains `smooth_w10`.

Physical target construction evaluates HLT/offline pairs per jet; model-side
prediction is a small per-token MLP. Preparation is not free, and it is not a
new teacher forward. A dedicated smooth-only port could store one family, but
that requires corresponding changes to its own inventory/validators/retirement.

There is no automatic deletion. The existing retirement command only removes
the three newly owned payload files after all consumers and the final report
validate, with separate exact-plan authorization. Models, metadata, predictions
and histories remain. Never delete borrowed teacher banks, parent worktrees,
datasets or `Fresh_check` to make this run fit.

## 10. Tests and pitfalls to carry into a port

Use [the focused suite](../tests/test_luka_neighborhood.py). Particularly useful
entry points include:

- `test_hand_calculated_counts_momentum_unknown_and_smooth`
- `test_context_permutation_query_equivariance_and_chunking`
- `test_phi_wrap_and_unmatched_context`
- `test_equal_jet_moments_and_scale_floor`
- `test_loss_equal_jet_padding_and_target_detached`
- `test_all_objectives_ce_only_main_and_connected_heads[smooth_w10]`
- `test_native_parity_and_rng[cpu]`
- `test_complete30_zero_neighborhood_parity`

Run from an environment with the repository and the correct `weaver-core`
ParticleTransformer implementation available, not an unrelated `weaver` package:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider \
    tests/test_luka_neighborhood.py
```

For a port, also test that:

- Changing a jet label cannot change its physical neighborhood targets.
- Context permutation leaves targets unchanged; query permutation applies the
  same permutation to targets, without affecting the native input contract.
- No validation or final-test particles are used to build/normalize targets.
- A zero-new-loss control preserves the original forward, RNG, optimizer update
  and training behavior; do not let adding the head reseed the backbone.
- Auxiliary gradients reach the encoder, targets are detached, and main-head
  parameter gradients remain exactly CE-only.
- Shape-compatible but misordered banks, corrupt bytes and wrong source/split
  lineage fail before training.
- Native checkpoint loading is strict and reproduces capture-disabled outputs.
- Poor but finite classification results still complete and are reported.

Previous implementation evidence is recorded in [HANDOFF](HANDOFF.md); it
includes local synthetic worker/parity tests, not a genuine new SPORC result.
During this documentation turn, eight focused scaffold/target/loss/registration
checks passed before editing and again afterward. All 32 local links resolve;
the three Python snippets parse and the recipe-inspection example executes.
Two additional model checks failed to start the model because `weaver.nn` was
unavailable locally; that is an environment limitation, not new evidence of
model parity or a scientific outcome.

## 11. Copyable instruction for the next chat

> Read `docs/LUKA_SMOOTH_W10_NEIGHBORHOOD_HANDOFF.md`, then the linked active
> plan/contract and source at commit
> `62f9773aabd150004be9c319d6ed06c412fcc57a`. We want HLT-only classification
> with removable per-particle prediction of physical offline neighborhoods:
> Gaussian widths 0.02/0.05/0.10, six reconstructed categories, count and
> HLT-normalized scalar-pt summaries, log transforms, equal-jet TRAIN moments
> with scale floor 0.1, and equal-jet masked SmoothL1 at total weight 10.
> The reference baseline is fresh fixed30 CE0.5 plus four separate auxiliary
> KD heads with total weight 1 and four jet-embedding heads with total weight
> 40. No auxiliary prediction enters classification. Keep exact reproduction
> distinct from a port; preserve data boundaries, particle joins, native
> inference, artifact authentication and existing user work. No smooth_w10
> improvement or successful remote submission is established by this handoff.

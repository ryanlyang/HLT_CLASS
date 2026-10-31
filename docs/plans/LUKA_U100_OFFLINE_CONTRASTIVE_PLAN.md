# U100-w5 plus paired OFFLINE jet identification: twenty-run debug study

Date: 2026-10-31. Status: **implemented as an additive debug campaign**.
The [runbook](../LUKA_U100_CONTRASTIVE_RUNBOOK.md) provides commit, queue and
results commands; the [v1 contract](../contracts/LUKA_U100_CONTRASTIVE.md)
defines the artifacts. Local tests are recorded in HANDOFF. Real SPORC
admission runs inside shared setup before automatic scientific submission.
No jobs have been submitted from the local implementation session, and no
recovery improvement is claimed.

## 1. Decision and leading candidate

Keep the existing HLT-only **CE + 5 * U100 embedding prediction** model intact.
Add a separate auxiliary head that identifies the corresponding OFFLINE jet
among frozen OFFLINE embedding candidates. Give that matching loss its own
coefficient. The main classifier continues to receive only the native HLT
representation. Remove both auxiliary heads for deployment.

My leading hypothesis is **`mix_l0p3_s0`**:

- CE coefficient 1 and U100 embedding coefficient 5, unchanged;
- independent OFFLINE matching head, final coefficient **0.3**;
- matching coefficient ramps linearly over the first three of thirty epochs;
- fixed cosine-softmax temperature **0.1**;
- one exact positive and **1,024 distinct negatives per query**;
- negatives: **512 broadly sampled + 256 same-class + 256 moderately difficult
  different-class jets**, with the latter ranked using the frozen OFFLINE
  teacher;
- frozen original U100 and OFFLINE teachers; no new teacher training;
- fresh matched HLT initialization, thirty epochs, batch 128, debug partition.

This is an informed experimental preference, not a measured winner. Broad
comparisons should provide a stable signal, same-class comparisons ask for
more than coarse class identity, and a bounded difficult component emphasizes
confusable jets without deliberately selecting only the closest candidates.
The 0.1 and 1.0 mixed variants bracket the preferred weight. The random sampler
is a serious contender: teacher-similarity mining might emphasize details HLT
cannot observe. The experiment must be able to establish that outcome.

The target is improved **native HLT classification AUC/recovery**, not perfect
jet identification. Matching may exploit momentum, multiplicity, or instance
details without improving classification. There is no supported prediction
that any row will reach 45% recovery on the high-data population.

## 2. Scope and authority

This is a new additive **SPORC/debug, small-data** screen. The user explicitly
confirmed that U100-w5 remains unchanged and OFFLINE identification gets a
separate branch and weight. That clarification governs every row below.

In particular, the earlier brainstorming suggestion of a CE-plus-matching arm
without embedding regression is **outside these twenty runs**. All twenty
retain U100 embedding weight 5. No weight is redistributed from U100 to
OFFLINE, and no main-logit KD, auxiliary logit KD, frozen-classifier readout,
retained predictor, token loss, relational penalty, or neighborhood loss is
silently added. Other brainstormed methods are separate future studies.

Scientific baseline references:

- [Direct U100 plan](LUKA_U100_EMBEDDING_PLAN.md).
- [U100 target and training contract](../contracts/LUKA_U100_EMBEDDING.md).
- [Multiview/readout plan](LUKA_U100_MULTIVIEW_READOUT_PLAN.md), for the same
  historical baseline and native-deployment boundary.
- [Testing ladder](../TESTING.md).

The old direct-U100 plan's tier3 single-job authorization and the multiview
campaign's acceptance do not automatically authorize this new debug study.
The new scientific plan takes precedence over generic original-transfer
10-class/trimming defaults: this is the authenticated historical Luka
**11-class, 17-feature, full-support** population. Existing contracts and
immutable execution specs remain unchanged. Implement a separate versioned
artifact family for the new objectives and samplers.

## 3. Fixed population, teachers, and training recipe

| Item | Required first-screen setting |
| --- | --- |
| TRAIN / validation | Authenticated ALLJETS **299,234 / 74,895** jets |
| Student observation | Identical authenticated D000/HLT input in every row |
| Student architecture | Native installed-Weaver ParT, 128-D pre-classifier jet vector |
| Main classifier | Existing native Linear(128,11), CE coefficient **1** |
| U100 teacher | Original regular single-encoder **COARSE_U100** behind direct U100-w5 |
| OFFLINE teacher | Original fixed30 **OFFLINE** from the same foundation |
| U100 objective | Coordinate-mean SmoothL1(beta=1), total coefficient **5** |
| Training budget | **30 full epochs**, physical batch **128**, no accumulation |
| Optimizer | Inherit exact U100 AdamW groups; betas 0.9/0.999, eps 1e-8, weight decay 0.01 |
| Learning rate | Existing warmup3/cosine30, peak 3e-4, floor 1.5e-5 |
| Precision | Historical BF16 backbone; FP32 auxiliary heads, similarities, log-softmax, losses |
| Initialization | From scratch; matched seeds and row order within each replicate |
| Selection | Full-validation native macro OVR AUC; original deterministic tie-breaks |
| Early stopping | None; all finite scientific outcomes complete the registered budget |
| Deployment | Native HLT encoder and classifier only |

Inherit and hash the exact historical recipe, rather than reconstructing it
from the short table. Preserve normalization, parameter-group exclusions,
dropout, final partial batches, sampling order, and input support. Do not borrow
the CE0.5/K1/E40 recipe from the other grid.

Teacher discovery hints, **not proof of reusable artifacts**:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/
  luka_fixed30_74aa58c5_r1/science/train_COARSE_U100/
  luka_fixed30_74aa58c5_r1/science/train_OFFLINE/
  luka_u100_direct_ed6d0af4_tier3_r1/
```

Authenticate selected checkpoints, exact model/input definitions, original
source, completed receipts, reports, and paired population before reuse.
Never substitute c50 UE100, SupCon OFFLINE, a FullStat teacher, or a fusion
teacher just because its checkpoint exists or scores better.

The high-data recovery of approximately 33% mentioned in the conversation is
motivation, not the baseline value for this small-data study. Small-data
historical U100-w5 AUC is approximately 0.96206974; authenticate its exact saved
value. M0HLT and OFFLINE historical anchors are approximately 0.96056122 and
0.96452569. Original50k overlaps validation and is report-only. Final test and
unused data are not accessed.

## 4. Model, targets, and gradient flow

```text
HLT particles -> native ParT -> h_i, width 128
                                |
                                +-> native classifier -> 11 logits -> CE
                                |
                                +-> existing U100 MLP -> predicted U100 vector
                                |                       -> 5 * SmoothL1
                                |
                                +-> new matching MLP -> normalized query q_i
                                                        -> OFFLINE candidate loss

Frozen U100 model   -> paired TRAIN embedding -> U100 regression target
Frozen OFFLINE model -> paired TRAIN embedding -> positive / negative keys
```

Both heads take `h_i` directly. The matching head does not take the predicted
U100 embedding, teacher logits, true label, event ID, row number, or any OFFLINE
value as an input. Teacher candidates are read only by the loss/sampler.
Candidate-index lookup is supervision bookkeeping, never a network feature.

Preserve the existing U100 predictor exactly:

```text
g_U = Linear(128,128) -> GELU -> Linear(128,128)
```

Use an independent matching head of the same shape, with no BatchNorm, dropout,
learned temperature, or candidate-specific parameters. Initialize it in a
forked RNG scope so adding it cannot change native/U100 initial weights or the
backbone dropout stream. Recommended new matching-head seed: `2026103101`.

For teacher raw pre-classifier vectors `e_U` and `e_O`, compute separate
coordinatewise TRAIN population means/scales in FP64, with the existing
standard-deviation floor of `1e-3`:

```text
z_U = (e_U - mu_U) / sigma_U
t_O = (e_O - mu_O) / sigma_O
unit(v) = v / max(L2_norm(v), 1e-12)
k_i = stopgrad(unit(t_O_i))
q_i = unit(g_match(h_i))
```

Store FP32 targets/keys and FP64 normalization evidence. Moments are fixed,
TRAIN-only, and never refitted by batch, class, validation, or student run.
The U100 target is not L2-normalized; its existing regression loss is unchanged.
There is no trainable teacher projection, teacher optimizer, EMA encoder, or
moving target queue. Finite degenerate vectors are counted diagnostically,
not rejected for poor scientific target quality; nonfinite data fail closed.

Every loss gradient reaches the shared HLT encoder through its own head. U100
and OFFLINE teachers and their keys receive no gradients. Validation for model
selection runs only the native HLT classifier. Removing both auxiliary modules
must reproduce the native validation probabilities within registered tolerance.

## 5. Exact matching objective and weights

For TRAIN anchor `i`, let `N_i` contain exactly 1,024 distinct eligible negative
identities. The one positive is its authenticated paired OFFLINE jet.

```text
s_positive = dot(q_i, k_i) / tau
s_negative[j] = dot(q_i, k_j) / tau, j in N_i
L_match_i = logsumexp([s_positive, s_negative...]) - s_positive
L_match = mean_over_anchor_jets(L_match_i)

E_U = mean_over_jets_and_128_coordinates(
    SmoothL1(g_U(h_i), stopgrad(z_U_i), beta=1))

L_total = CE(native_logits, y) + 5 * E_U + lambda(update) * L_match
```

Use stable FP32 log-softmax/logsumexp. Do not divide the matching loss by the
number of candidates, multiply it by KD temperature squared, or average it
over the 128 embedding coordinates. Those are different objectives. Every
nonzero arm has 1,025 candidates and `tau=0.1`, so their nominal weights are
comparable under the same reduction, although gradient scales can still differ.

Let `S=ceil(N_train/128)=2338`, and let `u` be the one-based completed update:

```text
lambda(u) = lambda_max * min(1, u / (3*S))
```

Thus the matching coefficient reaches its registered maximum after epoch 3.
CE1 and U1005 apply from the first update. No adaptive weighting, validation
feedback, or early stopping changes the schedule. The zero-weight controls
bypass matching-head execution, candidate sampling, and OFFLINE bank access
inside their training workers; they use the original U100-only kernel.

These weights are proposals, not equivalents of embedding weight 5. Log raw
and weighted losses and periodic shared-encoder gradient diagnostics to check
their actual contribution. Do not silently rescale a coefficient after seeing
its gradients or results.

## 6. Candidate eligibility and reproducible sampling

### 6.1 Common rules

- Candidates come only from the authenticated TRAIN OFFLINE bank. No validation
  or test rows enter any negative pool, normalization, bins, or mining index.
- Exclude the exact anchor identity, any duplicate occurrence of that identity,
  and explicitly known replica aliases. If authenticated event-group IDs exist,
  also exclude the anchor's group. Record whether event grouping was available;
  a row hash alone does not prove distinct events.
- Negatives are unique within each query, including across mixture components.
  Different queries may reuse a negative. Another anchor in the minibatch can
  be a negative if it passes these rules; it is not added automatically.
- The student TRAIN minibatch remains the original shuffled population. Only
  the auxiliary candidate set is class-balanced or made harder. No example
  oversampling changes the CE/U100 optimizer trajectory.
- The positive has no learnable slot identity. Dot-product scoring is shared
  across all candidates. Moving the positive column and its CE index together
  must leave loss and gradients unchanged.
- Candidate selection uses frozen keys, TRAIN labels, and declared TRAIN HLT
  summary variables only. It never depends on the current student's score,
  validation performance, or a model's eventual selected epoch.
- Use independent deterministic RNG keyed by sampler version, root seed,
  replicate, epoch, anchor identity, and component tag. Do not use process hash,
  timestamps, worker count, batch arrival order, or the global training RNG.
- At the same replicate and epoch, different weights of the same strategy see
  exactly the same candidates. Shared components across strategies should use
  common random streams where their exclusions permit it.

Before submission, count eligible populations and verify each sampler can
produce its full quotas and hard-pool size. Insufficient capacity is an invalid
registered sampler, not a reason to repeat negatives, silently lower K, drop
anchors, or omit a run. A revised capacity rule needs an explicit plan/spec
revision. Production miniatures retain full candidate counts against the real
TRAIN bank even when their anchor subset is small.

### 6.2 Sampling primitives

**G: broad class-balanced random.** Draw across all 11 classes, assigning equal
integer quotas with a maximum one-row remainder difference. Randomize which
classes receive remainders using the isolated sampler RNG. Within a class,
sample uniformly without replacement from eligible, not-yet-selected rows.
The positive class is allowed; class balance prevents QCD prevalence alone
from determining almost every negative. G does not use similarity rankings.

**S: same-class random.** Uniform eligible rows with the anchor's true class.
These are distinct-instance negatives, not a claim that same-class jets are
semantically unrelated. This is an intentional test of within-class structure
and a possible source of harmful repulsion.

**H: bounded teacher-similarity mining across different classes.** For each
anchor and epoch, draw a class-balanced random proposal pool of **4,096**
eligible different-class rows. Rank them by `dot(k_i,k_j)`, decreasing, with
canonical identity as the tie-breaker. Sample uniformly without replacement
from **ranks 257 through 1,024 inclusive**. Use 256 or 512 of those 768 entries,
as the strategy specifies. H is moderately difficult relative to this finite
proposal pool, not global nearest-neighbor mining. Excluding its closest 256
avoids deliberately concentrating every hard draw on the extreme end. Those
identities can still be drawn by a separate broad component; they are not
globally banned. No student similarities or labels beyond the declared class
eligibility are used in ranking. This is not the triplet-loss definition of
"semi-hard" based on a positive-distance margin.

**X: different-class random.** Like G but exclude the anchor's class and balance
quotas over the remaining ten classes. This tests whether instance separation
within the same class is useful. It is easier to satisfy through class-level
information, so high matching accuracy is not evidence of a better classifier.

**K: kinematically similar broad negatives.** Replaces only G in `kinmix`.
Compute HLT jet pT from the sum of authenticated real HLT four-vectors and
multiplicity from the real-token mask, not offline quantities or provenance.
Fit TRAIN-only quantile edges: 20 bins for `log(max(pT,1e-6))` and 10 for
multiplicity, using NumPy's linear quantile convention; coalesce duplicate
edges. Begin with the anchor's two-dimensional bin, expanding by Chebyshev bin
radius until at least 4,096 eligible, not-yet-selected rows are available.
Within that region, draw 512 candidates with G's equal-quota rule over its
nonempty classes. If an individual class cannot meet its quota, expand again;
the full TRAIN region is the final explicit limit. Record expansion radius,
class composition, and full-region usage. No mass matching is introduced.
This tests whether easy pT/count matching hides a more useful structural task.
It can also remove useful variation, so it is a single exploratory arm here.

### 6.3 Five strategies

| Strategy | G | S | H | X | K | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `random` | 1024 | 0 | 0 | 0 | 0 | Broad contrastive reference |
| `mix` | 512 | 256 | 256 | 0 | 0 | Preferred mixture of broad, within-class, and confusable comparisons |
| `cross` | 0 | 0 | 0 | 1024 | 0 | Remove same-class repulsion |
| `hard` | 256 | 256 | 512 | 0 | 0 | Double the deliberately difficult fraction |
| `kinmix` | 0 | 256 | 256 | 0 | 512 | Preferred mixture with broad distractors matched in coarse HLT kinematics |

Assemble H first, then S, then G or K, excluding already selected identities at
each step. Random/cross use their sole component. The chosen sampling
distribution defines the objective; no unregistered importance correction is
applied to pretend it is a full-bank softmax. Each query still gets exactly one
positive and 1,024 negatives.

## 7. Exactly twenty student fits, in intended launch order

Every row uses **CE1 + U100 embedding5**. Lambda below is the final additional
matching coefficient. S0 is the inherited historical initialization/sampler
seed pair; S1 is the declared paired second replicate. "Priority" is study
order, not a claim to precedence over other cluster users.

| Priority | Run ID | Sampler | Lambda | Replicate | Purpose |
| --- | --- | --- | ---: | --- | --- |
| 1 | `mix_l0p3_s0` | mix | 0.3 | S0 | Leading performance hypothesis |
| 2 | `u100_w5_s0` | none | 0 | S0 | Fresh matched control alongside the lead |
| 3 | `mix_l1_s0` | mix | 1 | S0 | Stronger version of the lead |
| 4 | `random_l0p3_s0` | random | 0.3 | S0 | Is the preferred sampler better than broad sampling? |
| 5 | `mix_l0p1_s0` | mix | 0.1 | S0 | Gentler mixed supervision |
| 6 | `kinmix_l0p3_s0` | kinmix | 0.3 | S0 | Reduce easy kinematic matching |
| 7 | `cross_l0p3_s0` | cross | 0.3 | S0 | Check same-class negative interference |
| 8 | `hard_l0p3_s0` | hard | 0.3 | S0 | Increase bounded difficult comparisons |
| 9 | `mix_l3_s0` | mix | 3 | S0 | Test possible underweighting |
| 10 | `random_l1_s0` | random | 1 | S0 | Broad strategy at higher weight |
| 11 | `cross_l0p1_s0` | cross | 0.1 | S0 | Gentle class-separated supervision |
| 12 | `hard_l0p1_s0` | hard | 0.1 | S0 | Harder comparisons with weaker gradients |
| 13 | `mix_l0p3_s1` | mix | 0.3 | S1 | Predeclared repeat of the lead, regardless of S0 outcome |
| 14 | `u100_w5_s1` | none | 0 | S1 | Matched baseline for the second replicate |
| 15 | `random_l0p1_s0` | random | 0.1 | S0 | Gentle broad reference |
| 16 | `cross_l1_s0` | cross | 1 | S0 | Strong class-separated supervision |
| 17 | `hard_l1_s0` | hard | 1 | S0 | Stronger hard-sampling stress |
| 18 | `random_l3_s0` | random | 3 | S0 | Higher-weight broad reference |
| 19 | `mix_l0p03_s0` | mix | 0.03 | S0 | Near-baseline regularization limit |
| 20 | `mix_l10_s0` | mix | 10 | S0 | Deliberately strong upper boundary |

Count: **18 S0 fits + 2 S1 fits = 20**, including the two controls. There are
18 nonzero matching fits. There are no new teacher fits. Setup/export/report
or technical jobs, if required, must be counted separately in the queue plan,
not described as additional scientific configurations.

The order combines expected usefulness and information value: the moderate
mixed candidates are early; controls and sampler comparisons follow promptly;
extreme weights are late. S1 tests a fixed prior choice, not an adaptively
selected validation winner. No row is cancelled because an earlier row has
poor finite performance.

For S0, retain the exact parent backbone and epoch-sampler seeds and U100-head
seed `20261019`. For S1, add one to those three seeds, and add one to the new
matching-head and negative-sampler root seeds. Use negative-sampler root
`2026103102` for S0. Record the resolved integer seeds in the spec. Baseline and
matching runs within a replicate must share native/U100 initial tensors and
TRAIN permutations. New head construction, negative draws, and diagnostic
calls must not consume backbone RNG or change BatchNorm buffers.

## 8. Shared banks, storage, and cost

Use one authenticated TRAIN U100 target bank and one authenticated TRAIN
OFFLINE key bank for the entire study. Reuse existing payloads only after
validating their exact teacher, selected tap, source, moments, identities,
population, and publication evidence. If an old payload was retired, export
that teacher's compact targets once; do not retrain the teacher. No historical
files are overwritten and no per-arm teacher re-export is permitted.

Each bank joins by canonical identity and must reproduce the exact HLT TRAIN
ordering and labels. Labels and identities stay outside model inputs. The
OFFLINE artifact must record both its teacher standardization and the final
L2 key transformation, rather than masquerading as an old U100 target bank.

A `(299234,128)` FP32 array is approximately 0.143 GiB; two primary arrays are
approximately 0.286 GiB before identities, manifests, temporary files, and
student results. This is not the total quota request. Calculate an explicit
budget for twenty selected native/full states, predictions, bank publication
temporaries, and operational headroom. Do not register the user's entire
rounded `df` free-space figure as the campaign's required allowance.

No N-by-N similarity matrix or list of every jet's nearest neighbors is needed.
H ranks only a sampled pool. Compute candidates in bounded anchor chunks so
`batch * pool * width` temporary gathers cannot surprise memory accounting.
A full frozen bank may reside on GPU if measured headroom permits; otherwise a
registered bounded transfer path must preserve candidate identities and FP32
scores. Do not silently lower K or switch to low-precision mining.

For comparison, the same two 128-D arrays at 3M TRAIN jets occupy approximately
2.86 GiB before overhead. A high-data port needs its own storage/runtime study,
and must not reuse this screen's counts, baseline AUC, or teacher artifacts.

No model-level teacher forward is needed in the student loop. Matching still
costs a new MLP, key gathers, similarity calculations, and mining. Measure its
cost; do not call it free or assume old U100 timing covers every sampler.

## 9. Results and diagnostics that answer the question

Primary comparison: selected native validation AUC minus the fresh U100-w5
baseline **of the same replicate**. Report historical references separately.
Use original deterministic epoch-selection tie-breaks: lower validation CE,
higher macro log QCD rejection at 50%, then earlier update after AUC ties.

```text
recovery_percent = 100 * (AUC_model - AUC_M0HLT) / (AUC_OFFLINE - AUC_M0HLT)
```

Authenticate full-precision anchors. Do not replace the denominator with a
different teacher's score. Report accuracy, per-class AUC and QCD rejection at
30%/50% with zero-passing censoring. Original50k is report-only and overlapping.

Required epoch logs: native AUC, recovery, selected/best epoch, CE, U100 loss,
raw matching loss, weighted matching loss, realized lambda, elapsed time,
training-batch percent, and measured GPU/host peaks. Save epoch histories and
selected predictions for paired comparisons; live-log values remain provisional.

TRAIN-only matching diagnostics on a fixed, declared probe subset:

- positive cosine, negative cosine by component/class, and positive rank;
- top-1/top-10 identification and candidate duplicate/exclusion counts;
- actual same-class fraction, H rank coverage, K expansion/kinematic distances;
- query norms, degenerate-key counts, and uniform-class/confident-class shortcuts;
- occasional per-objective encoder gradient norms and cosine similarities.

Use a separate sampler stream for diagnostics, restore RNG/mode/buffers, and
do not add optimizer steps. Probe matching accuracy is a TRAIN diagnostic, not
generalization evidence. This first screen does not create a validation
OFFLINE retrieval bank. If held-out retrieval is later wanted, register it
explicitly as evaluation-only; it can never become a training-negative source.

Changing negative distributions changes retrieval difficulty. Do not compare
raw retrieval accuracy across samplers as though they faced the same task.
Better matching with worse native AUC is a negative scientific result. A useful
S0 gain should be checked against S1 before making a strong claim. Twenty-way
validation selection has selection bias; bootstrap intervals from saved paired
predictions do not remove it or estimate training-seed variability.

## 10. Debug ordering and admission

Intended execution: SPORC `debug`, account `reu-aisocial`, established
`atlas_kd_sporc` environment and A100 profile. This is not the Oscar/L40S
FullStat campaign. Resolve and record the actual site/QoS/GPU resources using
the repository's execution contract at implementation time.

Preferred graph: shared target authentication/export, then twenty independent
student fits, then a report after all terminal attempts. Reuse completed setup
receipts after exact validation; do not rebuild targets for different weights.
Baseline workers need U100 only. No scientific student depends on another
student's successful training.

Submit eligible fits in the table's order and allow up to two concurrent fits
for this study if the debug allocation permits it. If strict within-study
ordering is requested, use an explicit bounded dispatcher, or documented
two-lane `afterany` resource ordering; never a chain of `afterok` dependencies
that cancels later fits after an unrelated failure. Distinguish resource order
from true target-producer `afterok` dependencies. Submission order or scheduler
nice hints cannot guarantee cluster-wide start order. A report uses `afterany`
and identifies incomplete rows honestly.

The subsequent implementation request authorizes building this campaign for
debug. Actual submission uses the runbook's exact pushed source and complete
dry review. Shared setup supplies measured resources and genuine production
miniatures covering all five samplers and lambda-zero parity before the
dispatcher submits scientific fits. There is no standalone preflight job.
Local entrypoint readiness does not claim that real-SPORC admission has already
passed. The detached dispatcher waits for setup and automatically performs the
measured dry review/submission within the prospectively authorized envelope.

Learn from `lukauv_accept` job 21988800: all miniatures ran, but an estimate of
2,683 minutes rejected admission. Keep longest-support stress for memory and
numerical checks. Estimate runtime separately from warmed representative
production batches, including candidate mining, data/transfer time, validation,
and setup, with a recorded safety margin. Persist measurements/diagnostics
**before** any resource guard can reject the allocation. A memory stress maximum
must not automatically be treated as the cost of every batch in all epochs.
This note does not repair or authorize reuse of that old failed acceptance.

Workers set `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and prepend
`${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`; source helpers from absolute project
paths. No Fresh_check runtime import, dataset mutation, broad cancellation,
silent precision/batch/support changes, or automatic retry of partial claims.

## 11. Existing code reused and implementation map

Inspected repository HEAD: `8ad0b381547134569818b4258f3cd31b509e9efb`.
These are existing donor/reference surfaces. Actual adaptations and their
reviewed commit are recorded in [LEGACY_SOURCE_MAP](../LEGACY_SOURCE_MAP.md).
Historical runtime remains unchanged; the contrastive implementation is additive.

| Existing code | Relevant behavior / boundary |
| --- | --- |
| [U100 contracts](../../src/hlt_classification/luka_u100_embedding/contracts.py) | Exact regular U100 target, weight5 and matched recipe |
| [U100 training](../../src/hlt_classification/luka_u100_embedding/training.py) | Baseline objective, epoch sampler, optimizer groups and checkpoint selection |
| [U100 bank](../../src/hlt_classification/luka_u100_embedding/targets.py) | TRAIN-only export, frozen teacher, ordered identity/label joins |
| [Auxiliary model](../../src/hlt_classification/luka_embedding_aux/model.py) | Native pre-classifier tap, RNG-isolated predictor, stripped deployment state |
| [OFFLINE targets](../../src/hlt_classification/luka_embedding_aux/targets.py) | Original OFFLINE embedding normalization/export conventions |
| [U100 source](../../src/hlt_classification/luka_u100_embedding/source.py) | Original-source teacher/reference authentication; do not weaken its old guard |
| [Multiview banks](../../src/hlt_classification/luka_u100_multiview/targets.py) | Additional teacher-bound bank examples, not directly interchangeable artifacts |
| [OFFLINE cache replay](../../src/hlt_classification/luka_u100_multiview/offline_cache.py) | Missing-payload reconstruction only with exact historical bytes/fingerprint |
| [Native model](../../src/hlt_classification/jetclass2_delphes/model.py) | Existing installed-Weaver inference graph and environment |
| [Selection/runner](../../src/hlt_classification/jetclass2_delphes/runner.py) | Exact deployed predictions and checkpoint-selection semantics |
| [Fixed30 schedule](../../src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py) | Learning-rate/update semantics |
| [Exact DAG submission](../../src/hlt_classification/scouting/hcwdl_exact_dag_submission.py) | Dry/execute boundary and exact submission journals |
| [Existing U100 tests](../../tests/test_luka_u100_embedding.py) | Baseline loss, teacher binding, parity and publication references |

Implemented namespace: `src/hlt_classification/luka_u100_contrastive/`.
The following files and commands are available through the linked runbook:

```text
contracts.py    twenty arms, sampler/recipe versions, fixed coefficients/seeds
source.py       authenticate exact U100/OFFLINE anchors and source inventory
model.py        native student, separate U100 predictor and matching query head
targets.py      shared frozen OFFLINE keys, TRAIN moments and identity evidence
sampling.py     G/S/H/X/K eligibility, quotas, deterministic pools and ranks
losses.py       FP32 cosine candidate CE, reduction and lambda ramp
training.py     matched fixed30 loop, diagnostics, selected native states
resources.py    full-support memory checks, representative timing, diagnostics
worker.py       validated tasks, atomic artifact publication, inference readback
campaign.py     immutable twenty-row specification and shared setup graph
submission.py   dry review, exact journal, bounded concurrency/order, report

scripts/luka_u100_contrastive.py
scripts/queue_luka_u100_contrastive.sh
sbatch/run_luka_u100_contrastive.sh
tests/test_luka_u100_contrastive.py
```

The implemented artifact family `LUKA_U100_CONTRASTIVE_<KIND>/v1` binds
RECIPE, SAMPLER, KEYS/BUNDLE, SPEC, RESOURCE_DIAGNOSTIC/ACCEPTANCE,
REVIEW/PLAN/CLAIM, TRAINING_REPORT/FIT, and RESULTS. Original teacher IMPORT
and U100 BANK artifacts retain their original validated contracts and are bound
by the new SPEC/BUNDLE. Include source/checkpoint parents,
ordered TRAIN identities, standardization/key definitions, seed streams,
candidate counts, ranking/tie rules, ramp and loss reductions. Atomically
publish arrays/checkpoints before their completion receipts. Preserve failed
output and diagnostic evidence without pretending it is completed training.

## 12. Implementation checks before declaring this executable

1. Exactly twenty unique fits, the table order, two controls and the fixed S1
   pair; all use CE1/U1005, tau0.1, K1024 where nonzero, and thirty epochs.
2. Lambda zero follows the original baseline loss, sampler and update kernels;
   adding a query head in other rows cannot alter initial native/U100 tensors.
3. Analytical matching CE, gradient signs, candidate-column permutation,
   positive placement, batch averaging, partial batches and ramp endpoints.
4. Gradients reach both auxiliary heads and native encoder; teacher/key/moments
   remain frozen; class predictions have no dependence on candidate arrays.
5. Exact TRAIN joins reject corrupt, wrong-teacher, reordered, duplicate-ID,
   cross-split, wrong-normalization and wrong-embedding-tap artifacts.
6. Every sampler has 1,024 unique negatives, excludes the positive/known aliases,
   respects quotas and H rank bands, and reproduces candidates across batching
   and weight variants. K bins use only HLT TRAIN quantities and fixed edges.
7. Extreme/imbalanced eligibility fails visibly before submission if quotas
   cannot be met; no hidden repetition, smaller pools, or skipped anchors.
8. Genuine installed-Weaver baseline parity and complete small production-code
   fits for the new branches, with full-size candidate banks/pools; technical
   weights are discarded before from-scratch scientific initialization.
9. Host/GPU memory and representative timing include sampling/gathers/mining;
   resource rejection leaves a diagnostic artifact rather than losing evidence.
10. Selected native checkpoint reload works with both target-bank paths and
    teachers unavailable; auxiliary modules are absent from deployable state.
11. Poor finite AUC/matching/teacher quality completes; invalid/nonfinite data
    fail closed. No result-based omission or cancellation of other arms.
12. Dry-run/source-drift, duplicate/partial submission, failure isolation,
    exact-ID recovery and separate data-dependency/resource-order tests.

## 13. Interpretation and high-data follow-up

The most useful result is a repeatable positive delta versus the matched
U100-w5 control. Compare `mix` versus `random` at 0.1/0.3/1/3, and compare
`cross`/`hard` against their matched `mix` weights. `kinmix` is only a 0.3
mechanism probe. The 0.03 and 10 mixed rows bound under/overweighting; their
inclusion does not imply equal prior confidence.

If the preferred S0/S1 pair improves, that is stronger evidence than selecting
the largest AUC from eighteen S0 rows. If a different sampler wins, register a
new matched repeat before calling it established. Native classification and
per-class rejection take precedence over success at the auxiliary task.

A high-data confirmation must reuse that environment's original U100-w5
teacher, HLT views, split, training budget and matched baseline. It must obtain
its own OFFLINE keys and candidate statistics. Neither small-data weights nor
recovery percentages automatically transfer. Moving from 33% to 45% recovery
requires an extra `0.12 * (AUC_OFFLINE - AUC_M0HLT)` on that same high-data
population; the small-data historical gap is not the denominator for that goal.

## 14. Research context

[Contrastive Representation Distillation](https://arxiv.org/abs/1910.10699)
motivates transferring representations through contrastive comparisons. This
proposal is a specific one-way frozen-key cosine-softmax adaptation; it is not
an exact reproduction of that paper's estimator or memory-bank implementation.

[Debiased Contrastive Learning](https://arxiv.org/abs/2007.00224) motivates
checking the consequences of semantically related negative samples. Our
`cross` arm is an explicit supervised sampling ablation, not an implementation
of that paper's debiasing estimator. Same-class negatives are deliberately
tested rather than assumed to be either always useful or always invalid.

[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362) groups
same-class positives. Here the primary positive is the exact paired OFFLINE
jet, teachers remain frozen, and labels only control declared negative
sampling. This differs from training a new OFFLINE teacher with SupCon and
then predicting its embedding. All proposed gains remain to be measured.

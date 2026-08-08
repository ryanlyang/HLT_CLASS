# High-Coverage HLT-to-Offline Constituent Matching Research Plan

Status: **design draft and prerequisite study; not yet an implementation
mandate**.

This document defines the matcher research that must precede the proposed
confidence-warped, down-and-up distillation ladder. It does not modify the
already-executed PMARD pilot, authorize a new production campaign, or replace
the active
[PMARD campaign plan](SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md).
The existing pilot remains governed by the selective
[`fitted_strict` contract](../contracts/FITTED_STRICT_MATCHER.md).

The purpose of this phase is to answer one question well:

> Can we infer a high-coverage, high-purity correspondence from visible HLT
> constituents to native offline constituents that supports a coherent
> confidence-dependent repair path and a substantially stronger T100
> endpoint?

Only after that question is answered should a second implementation plan
freeze the complete cold-start and warm-start model ladders.

## 1. Desired eventual experiment

The eventual campaign keeps the deployable input fixed:

```text
HLT constituent view -> HLT-only Particle Transformer -> class prediction
```

Offline constituents are training-only privileged information. Each paired
jet row already contains one HLT jet and one offline jet; no jet-level nearest
neighbor search is needed or permitted. The unsolved problem is constituent
correspondence within that paired row.

For a visible HLT token `i`, the desired matcher returns either:

```text
offline constituent j, calibrated confidence q_i, correspondence type
```

or an explicit dustbin/abstention. Accepted real-particle assignments must be
one-to-one within a jet. The HLT token count, order, padding, and mask never
change. Extra offline constituents are not appended. At the full endpoint,
an accepted HLT slot contains the complete native record of its assigned
offline constituent: all 21 model fields and its four-vector. A dustbin slot
remains exactly HLT unless a separately named transport-completion experiment
is later authorized.

The intended model ladder is conceptually:

```text
offline-directed domains: D100 -> D75 -> D50 -> D25 -> D0
deployable ascent:                  D0 -> M1 -> M2 -> ... -> M6
```

The currently intended teacher lineage is:

```text
D100: confidence-warped T100 input, trained from labels
D75:  T75 input, taught by D100
D50:  T50 input, taught by D75
D25:  T25 input, taught by D50
D0:   exact HLT input, taught by D25

M0:   separate ordinary CE-only HLT ParT baseline
M1:   new HLT model taught by D0 only
M2:   HLT model taught by M1 plus privileged D25
M3:   HLT model taught by M2 plus privileged D50
M4:   HLT model taught by M3 plus privileged D75
M5:   HLT model taught by M4 plus privileged D100
M6:   HLT model taught by M5 plus native-offline TOFF
```

`M0`, `D0`, and `M1` are deliberately distinct. In particular, `M1` preserves
the bottom born-again/self-distillation rung instead of treating `D0` as the
first upward model or mixing the weaker `M0` teacher into its primary loss.
Every `M` model remains HLT-only at inference.

Two complete versions of this entire graph are intended:

- **cold track:** every child starts from the same deterministic random
  initialization policy and receives predecessor knowledge through KD only;
- **warm track:** every child starts from its immediate predecessor's selected
  weights and also receives the same KD supervision.

The root D100 may be shared when its source, seed, training recipe, and selected
checkpoint are identical, because it has no predecessor and therefore no
warm/cold distinction. Neither track is stopped because the other looks better,
and no rung-by-rung hybrid is selected after seeing validation results.

The present intent is a maximum of 60 epochs, validation after every complete
train pass, a minimum 40-epoch exposure, and optional predeclared patience
after epoch 40 while retaining the best validation checkpoints. Those model
semantics are recorded here only to explain why matching quality matters.
Their exact losses, temperatures, initialization seeds, early-stopping deltas,
checkpoint utility, and Slurm DAG will be specified in the future ladder plan,
not implemented from this document.

## 2. Why matching is the scientific bottleneck

The current fitted-strict matcher intentionally prioritizes purity over
coverage. On its canonical untouched sample it accepts 64.86% of HLT tokens
and 67.21% of aggregate HLT constituent pT. The pilot's T100 endpoint therefore
contains a mixture of complete offline particles in accepted slots and
unchanged HLT particles in all other slots.

That endpoint improved monotonically enough to be useful, but it closed only
about:

- 21.3% of the T0-to-native-offline CE gap;
- 16.2% of the macro-AUC gap;
- 18.8% of the accuracy gap;
- 17.5% of the mean-log-rejection gap.

The complete evidence and implementation map are in
[`PMARD_PILOT_PRELIMINARY_RESULTS.md`](../PMARD_PILOT_PRELIMINARY_RESULTS.md).

This does not prove that coverage is the only limitation. The remaining gap
can also come from wrong accepted pairs, the fixed HLT particle count, omitted
offline particles, mixed HLT/offline consistency, teacher optimization, or
limited KD transfer. It does show that a full ladder built on the present
selective endpoint would inherit a low teacher ceiling. Matcher research is
therefore the correct next isolation study.

A wrong high-confidence match is especially costly here. Repair does not move
only pT or direction: at alpha one it transfers charge, particle identity,
quality, track fit, impact parameters, b-tag-relative quantities, validity,
energy, and the full four-vector. High coverage obtained by arbitrary pairing
would create a stronger-looking but scientifically incoherent oracle.

## 3. Three meanings that must remain separate

### 3.1 Physical-correspondence assignment

One HLT reconstruction object is assigned to the single offline
reconstruction object most plausibly representing the same underlying object.
Assignments are one-to-one and may abstain. This is the primary matcher.

### 3.2 Set-transport completion

An HLT slot receives an offline-directed barycenter, cluster, or transported
mass because no defensible one-to-one object correspondence exists. This may
be useful for constructing a smooth privileged view, but it is not a physical
particle match and must use a separate contract, artifact field, and claim.

### 3.3 Forced-assignment oracle

Every HLT slot is paired with something solely to estimate the best possible
fixed-skeleton endpoint. It may use a forced Hungarian solution or another
deliberately optimistic rule. It is an oracle ablation only and cannot
authorize the primary ladder.

These three outputs may be compared. They must never be merged under one
`match_mask` or described collectively as matched particles.

## 4. Non-negotiable endpoint semantics

The matching study must preserve the existing 21-channel HLT model schema in
[`schema.py`](../../src/hlt_classification/scouting/schema.py) and the complete
mixed-type endpoint behavior in
[`repair.py`](../../src/hlt_classification/scouting/repair.py).

For accepted correspondence assignments:

1. `alpha = 0` is byte-identical to the canonical HLT input.
2. `alpha = 1` contains the exact assigned offline four-vector and all 21
   projected offline fields.
3. The HLT token count, order, mask, and padding are unchanged for every
   alpha.
4. Offline particle indices are unique within a jet and in bounds.
5. Particle identity is internally valid and mutually exclusive.
6. Charge and charged/neutral identity remain compatible.
7. Track fields and validity change coherently when charged applicability
   changes.
8. Phi interpolation follows the wrapped shortest path.
9. No match index, confidence, source coordinate, repair alpha, or offline
   field enters a deployable model input.
10. Dustbin tokens remain byte-identical HLT in the primary correspondence
    view, including at alpha one.

This study does not add or remove constituents to imitate the offline count.
The scientific endpoint is explicitly the best defensible offline particle
assignment on the original HLT skeleton.

## 5. Coverage objective and feasibility limits

The desired primary endpoint has fewer than 10% dustbin HLT tokens. The
initial acceptance targets are:

| Quantity | Primary target | Stretch target |
|---|---:|---:|
| Aggregate visible HLT-token coverage | at least 90% | at least 95% |
| Aggregate HLT scalar-pT coverage | at least 95% | at least 98% |
| Highest-pT HLT-token coverage | at least 99% | at least 99.5% |
| Aggregate dustbin fraction | below 10% | below 5% |
| Per major HLT particle category dustbin fraction | below 15% | below 10% |

These are research targets, not assignment quotas. The solver must never make
a low-quality assignment merely because a coverage counter is below target.

Before fitting any new model, a count-only feasibility audit must calculate:

```text
unconstrained ceiling = sum_j min(n_HLT,j, n_offline,j) / sum_j n_HLT,j
```

and tighter ceilings after hard particle-category and charge compatibility.
It must also report the distribution of `n_offline - n_HLT`, the fraction of
jets for which complete injective assignment is mathematically possible, and
coverage ceilings weighted by HLT scalar pT and token rank.

If hard one-to-one/category constraints make 90% coverage impossible, the
project must say so. It may then investigate set transport, split/merge-aware
clusters, or a revised endpoint, but it may not relabel those constructions as
particle matches.

Coverage must be reported globally and sliced by:

- particle category and charge;
- HLT pT rank and pT quantile;
- jet multiplicity and offline-minus-HLT count;
- local angular density and ambiguity/component size;
- jet pT, eta, and class;
- charged, neutral, and track-validity states.

A global 90% number is insufficient if the missing 10% contains most of the
jet pT, most heavy-flavor tracks, or an entire particle category.

## 6. Recommended matching strategy

The leading design is a **high-purity anchor plus contextual completion
matcher**. It combines interpretable local evidence, jet-level context, and a
global exclusive assignment. It is intentionally more capable than nearest
particle matching.

### 6.1 Tier A: immutable strict anchors

Run the authenticated fitted-strict matcher unchanged. Its accepted pairs are
high-purity anchors, not merely initialization suggestions. A later stage may
reject an anchor only in an explicit anchor-challenge ablation with a recorded
reason; the primary candidate keeps them fixed.

Anchors provide:

- a reliable partial correspondence map;
- empirical response and residual priors conditioned on category and scale;
- local spatial landmarks shared by the HLT and offline sets;
- positive pseudo-labels for contextual learning;
- a way to measure whether completion disturbs already-understood regions.

### 6.2 Tier B: broad but physical candidate graph

Build a sparse bipartite graph for remaining HLT and offline constituents.
Candidate gates should be broad enough to contain difficult matches while
rejecting physically impossible pairs. They should use, as applicable:

- wrapped delta-phi, delta-eta, delta-R, pT response, and energy response;
- HLT and offline pT ranks and rank displacement;
- charged/neutral category, PID transition, and charge transition;
- quality, lost-inner-hit, track-fit, dxy/dz, significance, and validity
  compatibility;
- distances and response relative to nearby strict anchors;
- local neighborhood summaries on both sides;
- jet-level count, pT, and category-balance context;
- evidence for reconstruction split, merge, loss, and duplicate ambiguity.

PID and charge must not automatically be treated as infallible measurements.
Some transitions may be real reconstruction disagreements. However, every
permitted transition needs a train-only empirical prior and an explicit
ablation. Completely unrestricted PID/charge matching is not acceptable.

### 6.3 Tier C: anchor-conditioned contextual scorer

Use a permutation-equivariant bipartite model to update HLT nodes, offline
nodes, and candidate edges through several rounds of within-set and cross-set
context. The existing sparse message-passing surfaces in
[`matching.py`](../../src/hlt_classification/scouting/matching.py) and
[`match_model.py`](../../src/hlt_classification/scouting/match_model.py) are
starting points, not automatically the final architecture.

The scorer should learn whether an edge is plausible in the context of the
entire ambiguous component. In particular, it should distinguish:

- two nearby particles crossing rank;
- one HLT object near several offline fragments;
- a dense shower from an isolated track;
- a local candidate that is reasonable alone but globally steals the only
  plausible partner of another HLT token;
- genuine PID/charge disagreement from a geometrically unrelated particle.

Training must not assume every non-anchor edge is negative. The preferred
formulation uses positive-unlabeled learning, hard alternative assignments,
synthetic corruption with known answers, and consistency objectives rather
than ordinary binary classification on fabricated negatives.

### 6.4 Tier D: global assignment with private dustbins

Convert contextual edge scores into a rectangular global assignment. The
primary solver must enforce:

- at most one offline endpoint per HLT token;
- at most one HLT token per offline endpoint;
- one private dustbin choice per HLT token;
- preservation of all fixed Tier-A anchors;
- deterministic tie handling and stable score quantization;
- an explicit alternative-assignment margin.

Hungarian min-cost assignment is the default. Entropic optimal transport may
be used for training or as an ablation, but the published correspondence map
must be discrete and one-to-one. A hard global dustbin budget is forbidden;
the dustbin cost may be calibrated, but it cannot compel an implausible real
edge to satisfy a quota.

### 6.5 Tier E: post-assignment confidence calibration

Confidence must estimate assignment correctness after exclusivity is solved,
not merely reproduce a local edge score. It should include:

- selected-edge probability or calibrated score;
- best local alternative margin;
- global constrained-alternative margin;
- mutual forward/reverse preference;
- anchor agreement and anchor-relative geometry;
- perturbation and feature-dropout stability;
- component size, density, and count imbalance;
- PID, charge, and track-validity transition rarity;
- solver agreement across Hungarian and transport diagnostics.

The artifact must preserve two concepts:

```text
p_correspondence = calibrated evidence that the selected objects correspond
q_repair         = monotone repair-confidence value used by the alpha warp
```

They may initially be equal, but they must remain separately named so later
calibration or conservative warping cannot be misrepresented as a probability.
Confidence is training-time metadata and never a Particle Transformer feature.

## 7. The confidence-warped repair path

For an accepted token with repair confidence `q_i` in `[0, 1]`, the initial
proposed effective repair strength is:

```text
gamma(q_i) = 1.3 - 0.6 q_i
a_i(alpha) = alpha ** gamma(q_i)
```

This gives approximately the desired behavior at nominal alpha 0.10:

| Repair confidence | Effective alpha |
|---:|---:|
| 0.0 | 0.050 |
| 0.5 | 0.100 |
| 1.0 | 0.200 |

It also preserves the two essential endpoint identities:

```text
a_i(0) = 0
a_i(1) = 1
```

Thus high-confidence pairs move earlier in the ladder, ambiguous accepted
pairs move later, and every accepted pair reaches its exact offline endpoint
at T100. Dustbin tokens have no endpoint and remain HLT for every alpha.

Continuous fields and the four-vector use `a_i` in place of uniform alpha.
Phi uses wrapped angular interpolation. Discrete and validity groups use the
existing deterministic identity-bound nested switches, with offline selection
when the fixed uniform variate is below `a_i`. Grouped switching preserves PID,
charge, track applicability, and validity coherence. No discrete field is
numerically interpolated.

This curve is a registered candidate, not an unquestionable choice. It must be
compared with uniform alpha, weaker/stronger confidence exponents, and a
piecewise monotone curve. Selection must consider teacher smoothness and
endpoint quality, not only the best single student metric.

The confidence warp cannot rescue a wrong T100 match: `a_i(1)` is one for every
accepted pair. Therefore the alpha-one acceptance gate must remain a genuine
quality gate even if lower rungs are conservative.

## 8. Learning and audit data discipline

The source split remains file-disjoint 60/20/20. Final-test particle branches
remain sealed until the later ladder campaign's finalist and execution locks.

Matcher development uses train-role rows only. The train role should be split
deterministically into five source-aware cross-fitting folds plus a frozen
matcher-audit subset. For each fold:

- fit response priors, contextual weights, and confidence calibration on the
  other folds;
- infer assignments for the held-out fold;
- publish only out-of-fold matcher diagnostics;
- never train and score confidence on the same edge population.

Validation may evaluate a completely frozen matcher and endpoint teacher, but
must not refit matching thresholds or confidence curves. If several matcher
families are screened on validation, that selection becomes part of the
future campaign multiplicity and requires a separately reserved confirmation
stage. Final test never chooses the matcher.

Because the dataset has no exact constituent truth, the project must not call
ordinary agreement labels ground truth. It should combine independent audit
proxies:

1. recovery of held-out fitted-strict anchors when those anchors are hidden;
2. known-answer synthetic perturbations, deletions, duplicates, splits, and
   merges applied independently to HLT and offline views;
3. stability under benign feature dropout and small physical perturbations;
4. forward/reverse and solver-consensus consistency;
5. post-assignment kinematic, PID, charge, track, and local-context closure;
6. blinded manual inspection of predefined diagnostic plots where useful;
7. downstream endpoint behavior, used only as evidence of utility and never
   as proof of physical correctness.

Reported `precision` must always name the proxy or benchmark under which it
was estimated. No result from this dataset alone authorizes the phrase
“physical truth match.”

## 9. Matcher acceptance gates

A candidate can become the primary high-coverage matcher only if all hard
integrity checks pass and the evidence supports both coverage and purity.

### 9.1 Hard integrity gates

- deterministic output under row, batch, chunk, and worker repartitioning;
- exact one-to-one accepted indices, with dustbins explicit and in-bounds;
- no final-test access during fitting or selection;
- finite scores/confidences and valid particle/category states;
- complete source, split, fold, feature, model, calibration, and threshold
  lineage;
- exact alpha-zero identity and exact alpha-one all-field endpoints;
- no offline-derived value in deployable inputs;
- unchanged fixed strict anchors in the primary candidate;
- fail-closed rejection of corrupted or cross-source artifacts.

### 9.2 Quantitative research gates

The initial promotion target is:

- at least 90% aggregate HLT-token coverage and below 10% dustbin;
- at least 95% HLT scalar-pT coverage;
- at least 99% highest-pT-token coverage;
- at least 99% lower-confidence-bound precision on the untouched strict-anchor
  benchmark;
- at least 97.5% estimated correctness on the declared independent extended
  benchmark, with the benchmark name and limitations reported;
- no major category below 85% coverage without an explicit exception;
- no material degradation of strict-core precision, perturbation stability,
  one-to-one integrity, or endpoint physicality.

The 97.5% value is a design target under an imperfect proxy, not a claim of
97.5% truth accuracy. If no defensible proxy can support it, the candidate
remains experimental regardless of its downstream classification score.

### 9.3 Scientific usefulness target

After freezing the matcher, train only endpoint teachers/oracles needed to
measure the ceiling. Compare identical-architecture models on:

1. HLT T0;
2. current 64.86%-coverage selective T100;
3. proposed high-coverage correspondence T100;
4. forced-assignment T100 oracle;
5. a separately named transport-completion endpoint, if implemented;
6. native offline TOFF.

The high-coverage endpoint should recover materially more of the native
offline gap than the current 16--21%. Recovering at least half of the CE and
macro-AUC gap is a useful go target; 70% is the stretch target. This is not a
hard correctness gate, because classification performance cannot validate a
match. It determines whether the expensive full ladder is scientifically
worth running.

## 10. Required ablations

### 10.1 Matching evidence

- current fitted-strict baseline;
- relaxed fitted-strict gates with recalibration;
- p4/rank-only contextual matcher;
- p4 plus PID/charge;
- p4 plus track/impact-parameter information;
- complete all-field contextual matcher;
- complete matcher without strict anchors;
- complete matcher without neighborhood or jet context.

### 10.2 Assignment mechanism

- nearest-delta-R diagnostic;
- greedy local score;
- local-score Hungarian;
- contextual-score Hungarian;
- contextual optimal transport followed by discrete projection;
- recommended fixed-anchor contextual Hungarian hybrid.

Nearest-delta-R is a negative control, not a plausible production choice.

### 10.3 Coverage and confidence

- threshold points near 65%, 75%, 85%, 90%, and 95% coverage;
- raw edge confidence versus post-assignment confidence;
- calibrated probability versus percentile/rank repair confidence;
- uniform alpha versus the proposed confidence warp;
- confidence warp with low-confidence pairs withheld at T100;
- forced 100% assignment as an oracle only.

### 10.4 Split/merge and count mismatch

- strict one-to-one correspondence;
- split/merge-aware features while retaining one-to-one output;
- explicit cluster/transport completion under a separate contract;
- highest-pT offline truncation on the HLT skeleton;
- native offline particle count and native offline model.

These isolate whether the remaining oracle gap is caused by incorrect
correspondence, missing offline constituents, or the fixed HLT skeleton itself.

## 11. Efficient artifact and execution design

The final frozen matcher should run once per selected jet at the beginning of
a campaign. Training jobs must never repeat matching every epoch.

Reuse the compact sparse-assignment pattern in
[`selective_assignment.py`](../../src/hlt_classification/scouting/selective_assignment.py):

- one shard per authenticated source unit;
- row offsets plus compact HLT and offline indices;
- quantized calibrated correspondence and repair confidence;
- an explicit correspondence-type byte if transport is later supported;
- dustbins implicit from missing HLT indices or explicitly bit-packed;
- per-shard content hashes and an immutable manifest;
- bounded lazy loading or process-RAM materialization;
- no durable repaired particle dataset.

Exact storage size must be measured on the pilot before production; it must
not be guessed from nominal row counts. The representation should remain small
relative to the 41 GB compressed source dataset. Repaired views may be built
once in process RAM under the existing
[`ephemeral view-cache contract`](../contracts/PMARD_EPHEMERAL_VIEW_CACHE.md),
then replayed across epochs. They must not be persisted as duplicated ROOT,
NPZ, or Parquet datasets.

The final-test assignment cache is built only after the later campaign's
finalist and execution locks. Train and validation assignment artifacts cannot
silently authorize final-test branch access.

## 12. Research phases and decision points

### Phase 0: count and category feasibility

Publish count-only coverage ceilings and determine whether sub-10% dustbin is
mathematically possible under honest one-to-one matching.

### Phase 1: immutable matcher benchmark

Freeze train-only folds, anchor definitions, synthetic corruption suites,
audit proxies, slice metrics, and acceptance thresholds before fitting the
new matcher.

### Phase 2: candidate development

Implement and compare relaxed local, contextual, global, and hybrid matchers.
All candidates use identical folds and candidate populations.

### Phase 3: cross-fitted calibration and audit

Calibrate post-assignment confidence out of fold. Produce coverage-purity
curves, dustbin slices, stability measurements, and failure galleries.

### Phase 4: frozen matcher selection

Select one primary correspondence matcher and, at most, one separately named
transport completion. Publish a versioned matcher artifact and exact threshold
before any endpoint model comparison.

### Phase 5: one-time assignment build

Build and authenticate compact train/validation assignments once. Verify exact
role coverage, source lineage, storage, and deterministic reload.

### Phase 6: endpoint ceiling study

Train T0, current selective T100, high-coverage T100, relevant oracle
ablations, and TOFF with identical budgets and per-epoch validation. This phase
determines whether the new endpoint justifies the full ladder.

### Phase 7: ladder-plan handoff

Only after the matcher and endpoint are frozen, write the authoritative ladder
implementation plan specifying both complete cold and warm tracks, 60-epoch
maximum training, validation every epoch, optional early stopping, KD mixtures,
teacher lineage, paired seeds, artifact reuse, Slurm DAG, and final-test locks.

## 13. Failure interpretation

| Observation | Most likely implication | Next action |
|---|---|---|
| Feasible coverage ceiling below 90% | One-to-one object matching cannot meet the desired endpoint | Design a separately named transport/cluster endpoint |
| Coverage rises but audit purity collapses | The extra pairs are not defensible correspondences | Keep abstention; improve context or candidate evidence |
| Coverage and proxy purity rise but T100 barely improves | Fixed skeleton, omitted offline particles, or mixed-domain consistency is limiting | Run count/transport/native-offline endpoint ablations |
| T100 becomes strong but KD remains weak | Representation or optimization transfer is limiting | Proceed to ladder/KD design with the stronger teacher |
| High-confidence matches help and low-confidence matches hurt | Calibration/gating is useful but T100 acceptance is too broad | Tighten T100 gate or keep low-confidence completion separate |
| Forced assignment approaches TOFF but honest matching does not | A transport endpoint may be useful, but a physical-correspondence claim is unsupported | Separate scientific claims and contracts |

A disappointing outcome is a scientific result. It must not cause registered
candidate rows to be skipped or artifacts to be relabeled after inspection.

## 14. Implementation boundary for this document

This document authorizes planning and local matcher investigation only after
the user explicitly asks for implementation. It does not yet authorize:

- changing the active PMARD matcher contract;
- replacing existing assignment artifacts;
- implementing or submitting either complete model ladder;
- opening final-test particle branches;
- calling transport output a physical particle match;
- launching a full production campaign.

The immediate next implementation task, once authorized, is Phase 0 plus the
frozen matcher-benchmark contract. The full ladder must wait until the matcher,
confidence definition, coverage threshold, and T100 endpoint have survived
that study.

## 15. Current code map

| Concern | Existing implementation |
|---|---|
| Scouting 21-field and collection schema | [`schema.py`](../../src/hlt_classification/scouting/schema.py) |
| Current fitted-strict matcher | [`fitted_strict.py`](../../src/hlt_classification/scouting/fitted_strict.py) |
| General candidate graph/global matching tools | [`matching.py`](../../src/hlt_classification/scouting/matching.py) |
| Sparse contextual matcher prototype | [`match_model.py`](../../src/hlt_classification/scouting/match_model.py) |
| Mixed continuous/discrete endpoint repair | [`repair.py`](../../src/hlt_classification/scouting/repair.py) |
| Compact persistent assignment cache | [`selective_assignment.py`](../../src/hlt_classification/scouting/selective_assignment.py) |
| Streaming repair construction | [`pmard_stream.py`](../../src/hlt_classification/scouting/pmard_stream.py) |
| Assignment authorization and lineage locks | [`locks.py`](../../src/hlt_classification/scouting/locks.py) |
| Current pilot evidence | [`PMARD_PILOT_PRELIMINARY_RESULTS.md`](../PMARD_PILOT_PRELIMINARY_RESULTS.md) |

New matcher work should extend these reusable surfaces or introduce a clearly
versioned successor. It must not create a parallel untracked data path or
reinterpret old v1 assignment artifacts under new semantics.

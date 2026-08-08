# High-Coverage Cold/Warm Distillation Ladder Implementation Plan

Status: **active, implementation-authoritative plan; runtime implementation is
not yet complete; numeric optimization recipe pending a versioned pre-launch
lock**.

Activated by the user's explicit implementation request on 2026-08-08. This
plan governs the successor to the original fitted-strict PMARD pilot. It does
not reinterpret old PMARD artifacts. The matcher research prerequisite is
documented in
[`HIGH_COVERAGE_HLT_OFFLINE_MATCHING_RESEARCH_PLAN.md`](HIGH_COVERAGE_HLT_OFFLINE_MATCHING_RESEARCH_PLAN.md).

Short name: **HCWDL** (High-Coverage Cold/Warm Distillation Ladder).

When this plan conflicts with the original PMARD plan for new HCWDL artifacts,
this plan governs HCWDL. It never changes the scientific identity or meaning
of an existing PMARD execution. The independently developed matcher donor is
commit `64be1a8` in:

```text
C:\Users\22rya\ComputerScience\FCV\high_coverage_matcher_research
```

Its portable handoff is
`docs/OUTSIDE_AGENT_IMPLEMENTATION_HANDOFF.md` in that repository. Every donor
file actually ported must be recorded in
[`LEGACY_SOURCE_MAP.md`](../LEGACY_SOURCE_MAP.md) with that exact commit.

## 1. Objective

Determine whether privileged offline-directed domains can be compressed down
to HLT and then progressively transferred back into a deployable HLT-only
Particle Transformer, and whether weight continuation or repeated from-scratch
function distillation is the better complete system.

Two full ladders are executed. No intermediate validation result may stop,
skip, splice, or hybridize either registered ladder.

## 2. Data populations

HCWDL reuses the authenticated Scouting source and whole-file-disjoint 60/20/20
split. It never copies or rewrites the ROOT dataset.

| Mode | Train | Validation | Final test |
|---|---:|---:|---:|
| Smoke | 4,096 | 4,096 | inaccessible |
| Pilot | 300,000 | 100,000 | 100,000 after locks |
| Production | every mapped train-role row | every mapped validation-role row | every mapped test-role row after locks |

Pilot selections are deterministic, class-proportional nested subsets of the
immutable roles. Production counts come from the realized split manifest;
nominal 60/20/20 targets are not substituted for whole-file realized counts.

## 3. Deployment and access invariants

- Every `M` model and `D0` consume HLT inputs only at inference.
- `D25`, `D50`, `D75`, and `D100` may consume their declared privileged view
  only during training and oracle evaluation.
- TOFF consumes native offline inputs and is never deployable.
- Match indices, confidence, alpha, source coordinates, and offline fields
  never enter a deployable model.
- Final-test particle branches remain sealed until finalist and execution
  locks exist.
- Scientific underperformance never fails a row or cancels descendants.
- Invalid lineage, forbidden access, corrupt artifacts, nonfinite required
  values, and non-exact resume fail closed.

## 4. Frozen high-coverage matching system

### 4.1 Scientific meaning

The selected matcher produces a **completion shell**. It is not a declaration
that every shell edge is a physical truth link. On the donor's independent
natural-population audit, the shell covers about 90.57% of HLT tokens, 99.23%
of HLT scalar pT, and 99.88% of leading HLT particles. Under the registered
harsh synthetic known-answer corruption its full-shell correctness is about
81.3%. The HC threshold recovers a proxy-qualified high-purity core at only
about 63--66% token coverage.

HCWDL deliberately uses the full shell as a high-coverage offline-directed
completion. Allowed claims must say `completion shell`, `offline-directed
repair`, or `fixed-skeleton endpoint`. They must not say truth match, exact
reconstruction association, perfect unsmearing, or 90% physical purity.

### 4.2 Particle population and native indices

For each already-paired ROOT row:

- HLT population: the first at most 200 visible `scoutpfcand` constituents in
  stored order;
- offline scoring population: regular charged `cpfcandlt` candidates followed
  by `npfcand` neutrals;
- appended lost tracks are excluded from assignment;
- internal compact neutral index zero starts at `n_cpfcands`;
- persisted native neutral index zero starts at `n_cpfcands + n_lts`.

The port must use the repository's existing `ParticleSet` decoder and must
preserve native indices. It may not introduce the donor's research NPZ
particle cache into production. Every particle must have positive finite pT
and energy, one of five exclusive categories, charge in `{-1,0,+1}`, and
finite-filled track values with explicit validity. Invalid candidates are
excluded or fail according to the frozen input contract; they are never
silently coerced into neutral hadrons.

### 4.3 Edge construction

For HLT token `i` and offline candidate `j`, build:

```text
deta       = eta_h[i] - eta_o[j]
dphi       = wrapped(phi_h[i] - phi_o[j])
deltaR     = hypot(deta, dphi)
log_pt     = log(pt_h[i] / pt_o[j])
log_energy = log(E_h[i] / E_o[j])
rank_delta = descending_percentile_rank_h[i]
             - descending_percentile_rank_o[j]
pid_transition    = 6 * pid_h[i] + pid_o[j]
charge_transition = 3 * (charge_h[i] + 1) + charge_o[j] + 1
```

Stable descending ranks and wrapped phi are mandatory. The primary gate is:

```text
deltaR <= 0.30
abs(log_pt) <= 4.0
abs(log_energy) <= 4.0
valid five-category identities
charges in {-1,0,+1}
```

PID equality and charge equality are score evidence, not hard gates. The
donor's full train-role feasibility audit found an exact-PID ceiling of about
76.72%, so strict equality cannot meet the high-coverage hypothesis. Gate
constants are scientific configuration and cannot be tuned from endpoint
classification.

### 4.4 Empirical score

The scorer loads train-only empirical LLR tables for `deltaR`, log-pT response,
log-energy response, PID transition, charge transition, and rank delta. It
applies the frozen nonnegative meta-weights and intercept from
`empirical_models.json`:

```text
score(i,j) = intercept + sum_k weight_k * LLR_k(feature_k(i,j))
```

Continuous features use frozen quantile bins; transition features use frozen
categorical tables. Arbitrary extrapolation, rebinning, clipping changes, or
weight normalization is forbidden in the port.

### 4.5 Lexicographic global assignment

The primary solver uses rectangular Hungarian assignment with one private
dustbin per HLT row and two strict priorities:

1. maximize the number of gated real edges;
2. among maximum-cardinality assignments, maximize total empirical score.

Scores are quantized at `1e-6`. Allowed secondary scores are normalized into
`[-1,+1]` within a jet. A real-edge cardinality unit of
`2 * min(n_hlt,n_offline) + 1` guarantees that one more gated edge dominates
every possible aggregate secondary-score change. Forbidden edges can never
win. Private dustbins cannot be replaced with a shared column.

The solver intentionally prefers a weak but gated completion edge over a
dustbin. Therefore the entire shell has completion semantics. A score-threshold
Hungarian assignment, greedy nearest neighbor, forced all-pairs solution,
Sinkhorn plan, hard-anchor lock, or contextual GNN is a distinct control and
cannot publish a primary HCWDL assignment artifact.

### 4.6 Anchor context and independent consensus

Ultra-tight mutual geometry/response pairs (`deltaR <= 0.0015`, absolute log
responses `<= 0.7`) provide a robust per-jet center/scale and diagnostic anchor
status. They are not hard-locked because donor ablation showed that locking can
reduce residual maximum cardinality.

A second assignment uses the same gate and lexicographic solver but a compact
independent geometry-plus-response score. Agreement between primary and
independent assignments becomes a confidence diagnostic. The selected runtime
does not load Torch, the contextual GNN, or optimal transport.

### 4.7 Post-assignment confidence

Confidence is computed only after global exclusivity. The ordered 18-vector is:

```text
score, row_margin, column_margin,
-deltaR/0.02, -abs(log_pt), -abs(log_energy),
pid_equal, charge_equal, mutual_geometry, solver_consensus,
anchor, log1p(anchor_count),
abs(centered_deta), abs(centered_dphi),
abs(centered_log_pt), abs(centered_log_energy),
log1p(row_degree), log1p(column_degree)
```

The frozen calibrator standardizes this vector, applies a logistic model, then
piecewise-linear isotonic calibration. Its output `q` is calibrated only to
the donor's registered synthetic corruption families. It is a repair-confidence
coordinate, not a proven real-particle correctness probability.

Frozen donor resources and semantic content hashes are:

```text
selected_matcher.json
  ea7dde63b66f9dc07d9f7532a320d560e83a20885c2580478627d68fee1a68d3
empirical_models.json
  b09d4ff84049f9646d3521e00cf6838d69ef62e0876535a52ad7981dba29b6bb
final_confidence_calibration.json
  7db644933ceb6541abd2e8869dccf8874d84753b46199cf2ab334082c1c3f53f
```

The port must validate parsed content, contracts, schema versions, feature
order, table sizes/monotonicity, finite coefficients/scales, isotonic
monotonicity, parent hashes, and exact selected constants before inference.
Path existence is not validation.

### 4.8 Cross-fitting

The donor resource contains `holdout_0` through `holdout_3` empirical scorers,
trained from development source folds 0--3, plus
`full_development_for_audit`. Train sources assigned to folds 0--3 use their
corresponding held-out scorer. Donor audit fold 4 uses the full-development
scorer, which did not fit on fold 4. Validation and post-lock final test use
the frozen full-development scorer.

The confidence calibrator is a single frozen train-role calibrator, not a
per-fold calibrator. This limitation is recorded; it is never refit on
validation or final test. Every shard stores the active scorer identity so a
consumer cannot silently mix folds.

### 4.9 One-time assignment storage

Assignments are computed once per selected jet, parallelized by source file,
then reloaded for every repair, teacher, and student. Training never rematches.
The dense per-visible-token payload is:

```text
entries                 int64
offsets                 uint64
native_offline_index    int16   (-1 means dustbin)
confidence_u16          uint16
```

Confidence is encoded as `round(q * 65535)` and reconstructed by exact division
by 65535. `uint16` is used instead of float16 because it has equal storage cost,
deterministic quantization, and preserves thresholds near one more reliably.
The shell is defined by nonnegative native index, not nonzero confidence.

Every shard and manifest binds source, split, role, row selection, source fold,
selected matcher config, empirical scorer, confidence calibration, native
index convention, selected candidate gate, logical array hashes, and byte
hashes. Validation requires exactly one dense row per visible HLT token, no
duplicate nonnegative offline index within a jet, correct lost-track offsets,
confidence in range, and exact selected-jet coverage. A deterministic sample
is recomputed from ROOT before authorization.

The assignment lock also requires `dustbin_fraction < 0.10` separately on the
complete selected train role and complete selected validation role. The
denominator is every visible HLT token, and dustbin means persisted native
index `-1`. This is the user's minimum high-coverage prerequisite, not a
tunable validation metric. Failure stops HCWDL before training and requires a
new matching plan; it cannot be repaired by hiding tokens, changing the
denominator, or choosing a confidence threshold. Per-category,
scalar-pT-weighted, leading-token, and jet-level dustbin distributions are
reported but cannot replace this token-count requirement.

The estimated full mapped-population payload is about 0.93 GB uncompressed;
the pilot is approximately one tenth of that. Final-test assignments are
constructed only after the execution lock.

## 5. Primary repair and endpoint qualification

The matching shell is plausible completion, not demonstrated physical truth.
The user selected `HIGHCOV_SHELL_EXACT/v1` as HCWDL's primary repair family.
This is fixed before implementation and is not chosen by validation.

### 5.1 Primary `HIGHCOV_SHELL_EXACT/v1`

Every non-dustbin shell assignment participates. Let `q_i` be reconstructed
from the persisted `confidence_u16`. The v1 confidence warp is:

```text
gamma(q_i) = 2.0 - 1.3 q_i
a_i(alpha) = 0                         when alpha = 0
             alpha ** gamma(q_i)       when 0 < alpha < 1
             1                         when alpha = 1
```

The explicit endpoint branches avoid numerical ambiguity at zero and one.
`gamma` ranges from 2.0 for the least-confident shell edge to 0.7 for the most
confident. Representative strengths are:

| Domain | `q=0` | `q=0.5` | `q=1` |
|---|---:|---:|---:|
| D0 | 0% | 0% | 0% |
| D25 | 6.25% | 15.4% | 37.9% |
| D50 | 25.0% | 39.2% | 61.6% |
| D75 | 56.25% | 67.8% | 81.8% |
| D100 | 100% | 100% | 100% |

Thus low-confidence assignments remain mostly HLT at D25, enter gradually at
D50/D75, and reach the exact assigned offline endpoint at D100. At D100 about
90% of HLT tokens and about 99% of HLT scalar pT are expected to be exact
assigned offline particles; true dustbins remain HLT. D100 is therefore an
exact **shell endpoint**, not the native offline jet and not 100% HLT-token
coverage.

Because the shell tail is uncertain, D100 may be worse than a more conservative
endpoint. That is a valid scientific result and does not redefine this v1
family after inspection.

### 5.2 Diagnostic `HIGHCOV_SHELL_SOFT/v1`

All shell assignments participate with

```text
a_i(alpha) = alpha * q_i ** gamma
```

The frozen diagnostic gamma is one. At nominal alpha one, uncertain
pairs remain partly HLT. This is a confidence-weighted completion view, not an
exact offline endpoint.

### 5.3 Diagnostic `HIGHCOV_HC_EXACT/v1`

Only assignments at or above the frozen HC confidence threshold participate.
They use the primary endpoint-preserving warp. Below-threshold shell assignments
remain byte-identical HLT. This is the cleaner correspondence endpoint but has
substantially lower token coverage.

The current `SELECTIVE_FULL_PARTICLE_ENDPOINT/v1` fitted-strict repair remains
the reference.

Before the full ladder campaign specification may be created, a separate
canonical same-capacity ParT endpoint-qualification campaign trains
identical-seed T0, fitted-strict, shell-soft, shell-exact, HC-exact, and TOFF
teachers. It reports validation metrics and the T0-to-TOFF gap recovered by
each endpoint. It does not automatically select another repair family.

The qualification task always completes for finite valid results. If
shell-exact performs poorly, launching HCWDL still requires a conscious user
decision. Switching to shell-soft or HC-exact requires a new version of this
plan, repair contract, and campaign identity; it cannot be accomplished by
editing a recipe file. Once a full ladder spec exists, every registered rung
runs regardless of performance.

## 6. Exact all-field repair

Every high-coverage family extends the existing 21-field mixed-type endpoint
implementation rather than replacing it. The primary implementation consumes
raw HLT/offline endpoints before CMSSW scaling and reconstructs model inputs
through the same canonical transform.

- p4 and continuous fields interpolate with `a_i`.
- Phi follows the wrapped shortest displacement.
- Quality, identity channels 1--6, and lost-inner-hits use deterministic,
  identity-bound nested switches with probability `a_i`.
- Existing validity groups switch coherently with the same `a_i`.
- Charged/neutral applicability changes remain coupled to identity switching.
- Dustbins and excluded shell tokens remain byte-identical HLT.
- `alpha=0` is exact HLT for every family.
- Every primary shell assignment is exact projected offline at alpha one.
- Token order, count, mask, padding, and HLT skeleton remain unchanged.

The exact field policy is:

| Channels | Fields | Policy |
|---|---|---|
| 0 | quality | deterministic discrete group |
| 1--6 | charge and five PID flags | one atomic identity group |
| 7 | phirel | wrapped-angle interpolation |
| 8--10 | etarel, abseta, log-pT | continuous interpolation |
| 11--19 | track/btag/log-energy quantities | continuous subject to grouped validity/applicability |
| 20 | lostInnerHits | deterministic discrete group |

Validity groups remain `quality`, `identity`, `relative_kinematics`,
`scale_kinematics`, `track_fit`, `track_dz`, `track_dxy`, and `track_btag`.
When validity differs between endpoints, the whole validity group follows a
deterministic switch at `a_i` rather than interpolating missing values. When
identity changes charged applicability, track values, quality, and
lost-inner-hits switch coherently with identity.

The discrete random coordinate is derived from SHA-256 of repair contract,
repair seed, canonical jet identity, HLT token index, and group name. It is
fixed across alpha values, epochs, batch layouts, and workers. Consequently
discrete repair is nested: once a group switches at one alpha it cannot switch
back at a larger alpha.

## 7. Frozen ladder graph

Shared roots:

```text
M0     ordinary CE-only HLT ParT
D100   fixed Shell Exact maximum-privilege ParT, trained from labels
TOFF   native-offline oracle ParT
```

Cold downward track:

```text
D100 -> D75c -> D50c -> D25c -> D0c
```

Warm downward track:

```text
D100 -> D75w -> D50w -> D25w -> D0w
```

Cold deployable ascent:

```text
D0c             -> M1c
M1c  + D25c     -> M2c
M2c  + D50c     -> M3c
M3c  + D75c     -> M4c
M4c  + D100     -> M5c
M5c  + TOFF     -> M6c
```

Warm deployable ascent:

```text
D0w             -> M1w
M1w  + D25w     -> M2w
M2w  + D50w     -> M3w
M3w  + D75w     -> M4w
M4w  + D100     -> M5w
M5w  + TOFF     -> M6w
```

`M0`, `D0`, and `M1` are distinct. M1 is a new born-again HLT model taught by
D0 only. M0 is not mixed into the primary M1 loss. Each ascent uses teachers
from its own cold or warm downward chain.

D100 is shared because it has no predecessor and therefore no warm/cold
initialization distinction. M0 and TOFF are also shared controls/oracles.

## 8. Cold and warm initialization

Cold children use a deterministic fresh ParT initialization at every rung.

Warm children load the immediate predecessor's selected model weights, then
reset optimizer, scheduler, scaler, update count, sampler position, and
training RNG for a new run. Warm initialization never carries optimizer
momentum and never removes the declared KD loss.

All models have the same HLT-style 21-input/15-output ParT capacity except the
native-offline TOFF input adapter. Teachers are frozen in evaluation mode.

## 9. Training budget and observation

Every pilot or production primary ladder model trains for exactly 60 complete
natural train-role passes. Smoke mirrors the graph with its bounded two-update
execution budget. Pilot/production updates are:

```text
ceil(train_rows / effective_batch_size) * 60
```

Validation runs after every complete pass, including pass 60. There is no
performance early stopping in the pilot or primary production ladder. Exact
resume checkpoints are published at every validation boundary and on Slurm
preemption. Full training continues after the selected checkpoint is first
observed.

The selected checkpoint maximizes validation macro OVR AUC, then minimizes
validation CE, then maximizes mean log QCD rejection at 50% signal efficiency,
then chooses the earliest update. Exact hexadecimal selection floats are
recorded. This checkpoint is both propagated to the next warm rung and used as
the frozen teacher checkpoint.

Final-epoch and selected checkpoints plus all 60 validation rows are retained.
No per-update model checkpoint is published.

## 10. Optimization recipe lock

The registered PMARD KD follow-up `pmard_kd_followup_b8a493547de8bd7e`
resolved the primary dual-teacher decision. A versioned `HCWDL_RECIPE/v3`
lock must exist before endpoint or ladder training. It binds:

- effective batch size and accumulation;
- cold and warm peak learning rates;
- warmup passes/fraction, minimum LR, cosine policy, weight decay;
- single-teacher CE/KD coefficients and domain-routed HLT/privileged temperatures;
- dual-teacher CE/predecessor/privileged coefficients and independent
  temperatures;
- class weights, AMP dtype, seed domains, and 60-pass budget;
- parent T100/schedule sweep reports and the deterministic selection rule.

The primary dual-teacher rungs are now fixed at 25% CE, 40% predecessor HLT
KD, and 35% privileged KD; privileged temperature is two and the dual-teacher
peak learning rate is `3e-4`. Pilot/production still use 60 passes, validate
every pass, and select by macro AUC first. The 40-pass fixed-LR evidence row
achieved CE `0.667893`, accuracy `0.790030`, and AUC `0.938643`; its 60-pass
counterpart remained competitive and improved rejection. The 60-pass maximum
therefore remains fixed while selection may choose an earlier checkpoint.

The complete primary recipe is now fixed: sole privileged teachers use 25% CE
plus 75% KD at temperature two; the sole HLT D0 teacher for M1 uses the same
coefficients at temperature one; HLT predecessors in dual-teacher nodes use
temperature one. Every primary peak learning rate is `3e-4`; effective and
microbatch size are both 256 with no accumulation; AdamW uses betas
`(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.01`, and no gradient clipping;
the schedule is five-percent warmup followed by cosine decay to five percent
of peak. Coefficients remain constant. BF16 model execution, FP32 loss math,
the square-root inverse-frequency class-weight rule, and the label-only warm
confirmation control are mandatory.

This is an enforced `primary_ladder` recipe profile, not an implicit default.
A lower-CE, stronger-privilege, different-temperature, reduced-warm-LR, or
accumulated-microbatch experiment must declare a separate
`registered_ablation` recipe and campaign identity. Missing or placeholder
values fail closed.

## 11. Teacher-target and view reuse

For each node, every required teacher is evaluated once over train identities.
Identity-keyed FP32 logits are retained in process RAM and replayed for all 60
passes. The student stream is then HLT-only unless the student itself is a
privileged-domain `D` model.

Each privileged `D` node constructs its repaired train and validation views
once in process RAM and replays the authenticated epoch sampler. It never
persists a repaired dataset. After teacher targets are computed, unused frozen
teachers and temporary views are released from GPU. Memory estimates are
checked against both a fixed cache cap and 75% of the Slurm allocation before
allocation.

Durable logits are optional only when one teacher checkpoint has multiple
registered consumers and the measured storage/computation tradeoff justifies
them. Any durable target cache is compact, FP32, identity ordered, hashed, and
deleted after its last authorized consumer. It never contains particle inputs.

## 12. Loss semantics

The primary ladder uses logits only. Representation KD, feature matching, and
layer freezing are excluded from the primary graph and may be registered as
post-ladder ablations.

Single-teacher nodes use class-weighted CE plus forward KL from their sole
teacher. Dual-teacher nodes use class-weighted CE plus independent forward-KL
terms from the HLT predecessor and privileged teacher. Each KL includes its
own `temperature**2` correction. CE, logits, softmax/log-softmax, KL, and
reductions execute in FP32 under BF16 ParT execution.

## 13. Seeds and multiplicity

The pilot executes both complete ladders with one screening seed. It then runs
five-seed confirmation for the predeclared compact set:

- M0;
- D0c and D0w;
- M1c and M1w;
- M6c and M6w;
- the best intermediate cold rung and best intermediate warm rung selected by
  the frozen AUC checkpoint/finalist rule;
- required same-generation self-KD controls.

The production campaign mode is implemented for both complete ladders. Live
submission remains unauthorized until the pilot, resource measurements, and a
separate production authorization lock explicitly decide whether to run both
tracks or a declared subset. Removing a track changes the production campaign
identity; it cannot be done by skipping registered jobs.

## 14. Primary comparisons

The ordered questions are:

1. Does M1 improve over M0?
2. Does each ascent rung improve over its immediate predecessor?
3. Does M6 improve over M1 and M0?
4. Does the complete warm system improve over the complete cold system?
5. How much of the D100-to-TOFF gap remains?
6. How much privileged-domain advantage reaches an HLT-only M6?

Primary reporting includes macro OVR AUC and multiclass CE. Mean log QCD
rejection is co-primary for interpretation. Accuracy, balanced accuracy,
top-label ECE, Brier, per-class OVR AUC, and per-class QCD rejection, including
Xbb/QCD and Xcc/QCD, are required secondary metrics.

## 15. Selection and final-test access

All selection uses validation. The finalist selector ranks highest macro AUC,
then lower CE, then higher mean log rejection, then stable graph ID. It freezes
a limited test wave before final-test assignments exist.

The pilot/production final-test wave contains only:

- M0;
- selected M6 cold and warm graphs;
- at most one predeclared best intermediate per track;
- D100 and TOFF oracle diagnostics;
- required confirmation/null controls.

Finalist and execution locks bind exact graph/report/checkpoint hashes. Final
inference atomically claims the test execution. The test is not reopened to
select a different epoch, recipe, repair family, rung, or track.

## 16. Campaign DAG

```text
source/split/audit/data lock
  -> highcov resource validation
  -> pilot row selection
  -> one-time train/validation assignment shards + manifest + lock
  -> Weaver parity and RAM/storage miniature
  -> recipe lock
  -> endpoint qualification -> Shell Exact qualification lock
  -> shared M0, D100, TOFF
  -> cold and warm down ladders (parallel tracks, sequential within track)
  -> cold and warm up ladders (parallel tracks, sequential within track)
  -> screen aggregation and confirmation lock
  -> five-seed confirmation
  -> finalist lock -> execution lock
  -> final-test row selection and one-time assignment cache
  -> sealed final evaluation -> aggregate report
```

Endpoint disappointment and rung disappointment do not prevent later
registered tasks. Dependencies express artifact availability only.

## 17. Modes and resources

- Smoke exercises every distinct code path with bounded rows and two updates;
  it does not access final test.
- Pilot uses 300k/100k/100k and measured resources.
- Production uses all realized role rows, requires a genuine Tigris miniature,
  clean pushed source, storage/RAM/time evidence, full dry run, and explicit
  authorization.
- Cold and warm branches may run concurrently; arrays are uncapped unless a
  measured cluster policy is explicitly recorded.
- Every Slurm worker uses the repository's absolute project path, isolated
  environment, `exec python`, and `USR1` checkpoint signal.

## 18. Versioned artifacts

Implementation must define new contracts for:

- high-coverage matcher resources and validation report;
- dense assignment shard and manifest;
- repair recipe and Shell Exact endpoint-qualification lock;
- HCWDL optimization recipe lock;
- ladder registry/specification and submission ledger;
- node training reports/resume checkpoints with AUC selection;
- screen aggregate, confirmation selection, finalist, execution, and final
  report.

Old PMARD v1 assignment, repair, report, resume, and campaign artifacts remain
readable only under their original consumers. New semantics require new names
and versions rather than silently broadening old contracts.

## 19. Required tests

- donor matcher score/assignment/confidence parity and resource tampering;
- maximum-cardinality-before-score, private dustbins, permutation handling,
  native lost-track offsets, and one-to-one integrity;
- dense assignment round trip, quantization, source-fold lineage, corruption,
  and exact role coverage;
- alpha-zero identity, shell-soft math, shell-exact endpoint, HC exclusion,
  all 21 fields, validity groups, discrete nesting, and p4 physicality;
- exact 60 epoch boundaries and AUC/CE/logR/earliest checkpoint ties;
- cold fresh initialization, warm weight inheritance, and optimizer reset;
- exact uninterrupted/resumed equivalence at a ladder node;
- correct teacher graph at every D/M rung and no M0 teacher in M1;
- complete cold and warm DAGs regardless of poor synthetic metrics;
- one-time assignment and one-time target/view construction;
- smoke bounds, production evidence gates, exact-ID recovery/cancellation,
  and final-test isolation;
- installed-Weaver parity and a genuine Tigris production-worker miniature.

## 20. Implementation boundary

This plan authorizes local implementation and verification. It does not
authorize a live pilot or production submission. A live pilot requires a clean
pushed commit, completed recipe lock, endpoint-qualification evidence, dry run,
measured resources, and explicit user launch instruction. Production requires
the additional evidence and authorization in Section 17.

## 21. Normative implementation detail

Sections 1--20 freeze the scientific design and its compact campaign overview.
Sections 21 onward are the normative implementation elaboration. They remove
degrees of freedom that an implementer could otherwise fill in inconsistently.
They do not replace the fixed decisions above: Shell Exact v1 remains primary,
both complete ladders run, every pilot/production primary node trains for 60
passes, validation is every pass, and checkpoint choice begins with highest
validation macro AUC.

### 21.1 Exact primary-node registry

HCWDL contains 23 primary training nodes: three shared roots, eight downward
nodes, and twelve deployable ascent nodes. All 23 are registered before the
first training job starts. `fresh` means a deterministic fresh ParT
initialization. `load X` means load only X's selected model parameters and
reset all training state. A teacher domain names the input view used to
evaluate the frozen teacher, not the student's input.

| Node | Student input | Initialization | Teacher 1 and domain | Teacher 2 and domain | Loss | Deployable |
|---|---|---|---|---|---|---|
| M0 | HLT | fresh | none | none | CE | yes |
| D100 | D100 Shell Exact | fresh | none | none | CE | no |
| TOFF | native offline | fresh | none | none | CE | no |
| D75c | D75 Shell Exact | fresh | D100 on D100 | none | CE + KD | no |
| D50c | D50 Shell Exact | fresh | D75c on D75 | none | CE + KD | no |
| D25c | D25 Shell Exact | fresh | D50c on D50 | none | CE + KD | no |
| D0c | HLT | fresh | D25c on D25 | none | CE + KD | teacher-only role |
| D75w | D75 Shell Exact | load D100 | D100 on D100 | none | CE + KD | no |
| D50w | D50 Shell Exact | load D75w | D75w on D75 | none | CE + KD | no |
| D25w | D25 Shell Exact | load D50w | D50w on D50 | none | CE + KD | no |
| D0w | HLT | load D25w | D25w on D25 | none | CE + KD | teacher-only role |
| M1c | HLT | fresh | D0c on HLT | none | CE + KD | yes |
| M2c | HLT | fresh | M1c on HLT | D25c on D25 | CE + two KD terms | yes |
| M3c | HLT | fresh | M2c on HLT | D50c on D50 | CE + two KD terms | yes |
| M4c | HLT | fresh | M3c on HLT | D75c on D75 | CE + two KD terms | yes |
| M5c | HLT | fresh | M4c on HLT | D100 on D100 | CE + two KD terms | yes |
| M6c | HLT | fresh | M5c on HLT | TOFF on native offline | CE + two KD terms | yes |
| M1w | HLT | load D0w | D0w on HLT | none | CE + KD | yes |
| M2w | HLT | load M1w | M1w on HLT | D25w on D25 | CE + two KD terms | yes |
| M3w | HLT | load M2w | M2w on HLT | D50w on D50 | CE + two KD terms | yes |
| M4w | HLT | load M3w | M3w on HLT | D75w on D75 | CE + two KD terms | yes |
| M5w | HLT | load M4w | M4w on HLT | D100 on D100 | CE + two KD terms | yes |
| M6w | HLT | load M5w | M5w on HLT | TOFF on native offline | CE + two KD terms | yes |

D0 is mathematically HLT-deployable because alpha zero is exact HLT, but its
scientific role is the last downward-domain teacher. M1 remains a separately
optimized student and cannot be aliased to D0 in code, paths, artifacts, or
reports. M0 is neither an initialization parent nor a teacher for M1.

D100, M0, and TOFF are shared. Every other teacher comes from the same cold or
warm track as its child. A cross-track teacher is invalid even if its
architecture and alpha happen to match.

### 21.2 Exact domain registry

| Domain | Particle view |
|---|---|
| HLT or D0 | authenticated HLT view; matcher not needed by the consumer |
| D25 | `HIGHCOV_SHELL_EXACT/v1` at alpha `0.25` |
| D50 | `HIGHCOV_SHELL_EXACT/v1` at alpha `0.50` |
| D75 | `HIGHCOV_SHELL_EXACT/v1` at alpha `0.75` |
| D100 | `HIGHCOV_SHELL_EXACT/v1` at alpha `1.00` |
| native offline | canonical native offline ParticleSet, not projected onto the HLT skeleton |

Alpha is stored as a canonical decimal and `float.hex()` value. A worker loads
it from the validated node spec; it cannot infer alpha from a node name or use
an environment override.

### 21.3 Common model contract

Every M and D node uses the same canonical 21-input, 15-output Particle
Transformer capacity, normalization, padding convention, token limit, pooling,
classifier head, and label order as the authenticated HLT baseline. A D model
never receives confidence, alpha, match indices, source coordinates, or offline
metadata as auxiliary inputs. Only its declared particle view differs.

TOFF uses the same ParT capacity and 15-output head with the canonical native
offline input adapter. Any unavoidable adapter difference is reported rather
than concealed by a claim that all input representations are identical.

The architecture config has a content hash bound into every node spec. A
parameter-name, tensor-shape, token-limit, transform, or label-order mismatch
causes warm initialization and teacher loading to fail closed.

## 22. Initialization and resume contract

### 22.1 Cold initialization

Every cold child starts from deterministic fresh parameters derived from:

```text
campaign seed domain + graph node ID + replicate seed + architecture hash
```

It does not load its teacher. Teacher information enters only through the
declared KD loss. Different cold rungs intentionally receive distinct initial
parameters; a shared accidental initialization is not permitted.

### 22.2 Warm initialization

Every warm child loads the immediate predecessor's selected model parameters
named in Section 21.1. It creates a new optimizer, schedule, AMP scaler,
sampler, update counter, validation history, best-state tracker, and RNG
streams. Optimizer moments, scheduler phase, scaler state, and dataloader
cursor are never inherited.

Warm start therefore tests continuation in parameter space under a new
objective. It is not checkpoint resume. The KD teacher stays frozen and the KD
term remains present even when the initialization parent and teacher are the
same checkpoint.

### 22.3 Exact resume

Resume is a separate mechanism. A rolling resume checkpoint restores model,
optimizer, scheduler, scaler, epoch/update, sampler cursor, RNG states,
validation history, interval-loss accumulators, cache identities, parent
hashes, and selected-best state. Resume cannot substitute a new recipe,
teacher, assignment manifest, repair family, or source commit.

Uninterrupted and resumed execution must be bitwise identical where the used
kernels are deterministic and within a documented numerical tolerance for any
installed-Weaver kernel that cannot be made deterministic.

## 23. Exact training and checkpoint protocol

### 23.1 Natural-population passes

If `N_train` is the exact selected train count and `B_eff` the locked effective
batch size:

```text
updates_per_pass = ceil(N_train / B_eff)
total_updates    = 60 * updates_per_pass
```

The final partial batch is consumed. Every selected train identity appears
exactly once per pass. Order changes deterministically by pass. Class
oversampling, replacement sampling, balanced batches, hidden row truncation,
and dropping the final batch are forbidden in the primary graph.

### 23.2 Every-pass validation

Validation occurs after passes 1 through 60, producing exactly 60 records. It
uses the complete selected natural validation role, with no resampling. M
models and D0 validate on HLT; D25/D50/D75/D100 validate on their own domains;
TOFF validates on native offline.

There is no performance early stopping. All 60 pilot/production passes run for
finite valid results. A lower AUC or higher CE is scientific evidence, not an
execution error. Smoke is an explicitly non-scientific two-update path test.

Each validation record stores pass, update, learning rate, interval-mean loss
components, CE, all metrics, elapsed timing, and CPU/GPU memory peaks. The
learning rate is the rate applied to the immediately preceding optimizer
update. This is required to diagnose the non-monotonic behavior already seen
in longer pilot schedules.

### 23.3 AUC-first selected checkpoint

The selected checkpoint is chosen across all 60 validation records by this
total order:

1. larger macro one-vs-rest AUC;
2. smaller multiclass cross entropy;
3. larger macro mean log QCD rejection at 50% signal efficiency;
4. earlier optimizer update;
5. lexicographically smaller canonical checkpoint identity.

Comparisons use unrounded FP64 report values. Every selection artifact records
the decimal and hexadecimal floating-point values, full ordered candidate
table, selected pass/update, model hash, and report hash. Console-rounded
values cannot participate. NaN in a required metric fails the node.

Only the selected checkpoint may initialize a later warm rung or produce
teacher targets. The pass-60 checkpoint is retained for diagnosis but never
substituted automatically.

### 23.4 Retention

Every node retains one atomically replaced rolling resume checkpoint, the
selected model-only checkpoint, the pass-60 model-only checkpoint if different,
all 60 validation rows, compact interval-loss history, immutable node spec,
selection report, and completed training report. Per-update checkpoint
publication is prohibited.

## 24. Recipe as a hard execution boundary

### 24.1 Values resolved for the primary pilot

The complete primary optimization recipe is resolved below. Operational RAM,
walltime, and storage requests remain subject to the genuine Tigris miniature,
but they cannot change scientific batching, optimization, temperatures, loss
weights, duration, validation cadence, or checkpoint selection. Launch code
rejects every placeholder.

### 24.2 Required `HCWDL_RECIPE/v3` contents

The immutable recipe binds:

- effective batch, microbatch, and gradient accumulation;
- separate cold-root, cold-child, and warm-child peak learning rates;
- optimizer, betas, epsilon, weight decay, and gradient clipping;
- warmup length, minimum LR, schedule function, and pass landmarks;
- single-teacher CE/KD coefficients and domain-routed HLT/privileged temperatures;
- dual-teacher CE/predecessor/privileged coefficients and two temperatures;
- constant or fully specified scheduled coefficient behavior;
- class weights and their authenticated train-count parents;
- BF16/FP32 policy, dropout, augmentation, and seed domains;
- 60-pass budget, every-pass validation, and AUC-first selector;
- evidence report hashes, candidate table, and deterministic choice rule;
- mode resource overrides that do not change scientific behavior.

Loss coefficients are finite, nonnegative, and sum to one within a declared
tolerance. Temperatures and learning rates are finite and positive. A locked
recipe contains no `null`, `TBD`, fallback, environment override, or implicit
default.

### 24.3 Locked complete primary recipe

```text
single teacher:  CE 0.25, KD 0.75
dual teacher:    CE 0.25, predecessor KD 0.40, privileged KD 0.35
single privileged temperature: 2
single HLT temperature:         1
predecessor temperature: 1
privileged temperature:  2
dual-teacher peak LR:     3e-4
all other peak LRs:       3e-4
effective/microbatch:     256/256 with accumulation 1
AdamW:                    betas 0.9/0.999, eps 1e-8, decay 0.01, no clipping
schedule:                 5% warmup, cosine to 5% of peak
class weights:            sqrt inverse authenticated train frequency
maximum passes:           60
validation:               every pass
checkpoint selector:      macro AUC, CE, log rejection, earliest
```

For a single-teacher node, runtime selects temperature by teacher domain:
privileged D teachers use two, while D0 teaching HLT-only M1 uses the HLT
predecessor temperature one. Less CE, another temperature, a reduced warm LR,
or another teacher mixture is a registered ablation rather than an in-place
modification of the primary.

### 24.4 Recipe resolution

The builder validates sweep specs, source roles, architecture, baseline and
teacher checkpoints, validation population, checkpoint selector, and completed
reports. It publishes the full candidate table, including execution failures
and valid poor rows, then applies the predeclared selection order. If evidence
does not determine a required value, it stops for explicit user authorization;
it never invents or silently falls back.

Changing a locked recipe after a ladder spec exists creates a new campaign
identity.

## 25. Loss and teacher-target semantics

### 25.1 Logit KD definition

For student logits `z_s`, teacher logits `z_t`, label `y`, and temperature
`tau`:

```text
CE = weighted_cross_entropy(z_s, y)
KD = tau^2 * KL(softmax(z_t / tau) || softmax(z_s / tau))
```

Teacher probabilities are targets and student log probabilities are the
modeled distribution. CE, logits entering the loss, softmax, log-softmax, KL,
coefficient multiplication, and reductions run in FP32 under BF16 ParT forward
execution.

### 25.2 Single-teacher nodes

D75, D50, D25, D0, and M1 use:

```text
L = lambda_CE * CE + lambda_teacher * KD_teacher
```

The frozen teacher evaluates the same jet identity on the richer domain named
in Section 21.1. D0 and M1 both consume HLT but are not duplicate experiments:
D0 learns from D25; M1 learns from the already compressed D0 function.

### 25.3 Dual-teacher nodes

M2 through M6 use:

```text
L = lambda_CE * CE
  + lambda_pred * KD(predecessor HLT teacher)
  + lambda_priv * KD(privileged-domain teacher)
```

The predecessor always sees HLT. The privileged teacher sees D25, D50, D75,
D100, or native offline according to the rung. The two KD terms use separately
locked temperatures and are never combined by averaging teacher logits or
probabilities before KL.

### 25.4 Labels and balance

The same authenticated jet label supplies CE in every domain. The training
population stays natural. If the recipe uses class-weighted CE, weights are
computed once from authenticated train-role counts, stored in the recipe, and
shared by comparable nodes. KD is not class-reweighted unless a separately
versioned ablation says so.

### 25.5 Explicit exclusions

Primary HCWDL excludes representation KD, embedding/Gram/attention losses,
layer freezing, feature adapters, teacher ensembling, hard pseudo-labels, and
online rematching. Those can be valuable later ablations, but adding one to a
primary node creates a new graph and campaign identity.

## 26. One-time assignments, views, and targets

### 26.1 Assignment reuse

Train and validation assignments are completed and authorized before the first
privileged teacher. Every D view and privileged teacher lookup consumes that
same dense assignment manifest. No epoch, node, seed, or ladder track reruns
the matcher. Test assignments are separately constructed once after the
execution lock.

### 26.2 Privileged-view RAM cache

A D-node worker streams each selected ROOT row once, joins the dense
assignment, constructs its declared repaired domain in canonical identity
order, and stores model-ready arrays in process RAM. That cache is replayed for
all 60 training passes and all 60 validation passes in the process. It is not
written as a repaired dataset.

The cache binds row identities, source hashes, assignment manifest, repair
contract, alpha, transform contract, tensor shapes/dtypes, and logical hashes.
Before allocation, the worker estimates required bytes and refuses the RAM
path if it exceeds the smaller of the configured cache limit and 75% of the
Slurm memory request. A declared streaming fallback may rebuild views from the
stored assignments if needed; it cannot rematch or change samples.

### 26.3 Teacher-logit RAM cache

Each frozen teacher is evaluated once per train identity. Its 15 FP32 logits
are stored in canonical class order and keyed by canonical jet identity. Raw
logits, rather than temperature-specific probabilities, are cached so the
locked loss can apply its own temperature.

For downward nodes the teacher uses its richer D domain. For upward nodes the
predecessor uses HLT and the privileged teacher uses its specified domain. For
M6, TOFF uses native offline particles. After targets exist, teacher objects
and unused temporary views leave GPU. An ordinary M student then uses only HLT
inputs plus joined logits.

A missing, duplicate, reordered, nonfinite, wrong-class-order, or wrong-parent
target fails closed.

### 26.4 Durable logit exception

RAM is the default. A compact durable logit cache is allowed when one selected
teacher has multiple registered consumers or measured preemption cost makes it
clearly preferable. It contains jet identities and FP32 logits only, never
particle arrays. It receives a versioned manifest, parent/checkpoint hashes,
logical hash, atomic publication, last-consumer declaration, and cleanup
record.

### 26.5 Deterministic replay

Sampler order depends only on canonical identity indices, replicate seed, and
pass number. ROOT streaming versus RAM cannot change ordering, partial batches,
augmentations, or teacher joins. Cache construction uses an isolated RNG domain
and cannot advance the scientific training RNG.

## 27. Matcher port and repository integration

### 27.1 Frozen donor

The immutable donor reference is commit `64be1a8`. At minimum, these semantics
are ported or independently reproduced with parity:

```text
src/highcov/final_matcher.py
src/highcov/assignment.py
src/highcov/repair.py
configs/selected_matcher.json
artifacts/models/empirical_models.json
artifacts/models/final_confidence_calibration.json
tests/test_assignment.py
tests/test_foundation.py
tests/test_repair.py
```

The donor research repository is not a runtime dependency. Resources are
packaged under a versioned directory in this repository and load without an
absolute Windows path. Every reused donor source/resource and donor commit is
recorded in `docs/LEGACY_SOURCE_MAP.md`.

The donor repair module is behavioral reference, not the final HCWDL policy:
the repository's existing all-21-field repair implementation remains the owner
of validity, identity, and endpoint semantics.

### 27.2 Planned responsibility boundaries

Names can receive a mechanical adjustment before code lands, but these module
boundaries are fixed:

| Planned path | Responsibility |
|---|---|
| `src/hlt_classification/scouting/highcov_resources.py` | load and validate scorer, calibrator, and selected config |
| `src/hlt_classification/scouting/highcov_assignment.py` | gate, scores, lexicographic solver, diagnostics |
| `src/hlt_classification/scouting/highcov_matcher.py` | repository ParticleSet-facing matcher and native-index result |
| `src/hlt_classification/scouting/highcov_cache.py` | dense shards, manifest, coverage, and sampled recomputation |
| `src/hlt_classification/scouting/repair.py` | three versioned high-coverage repair families |
| `src/hlt_classification/scouting/ladder.py` | node, domain, initialization, loss, and graph registry |
| `src/hlt_classification/scouting/ladder_contracts.py` | versioned specs, reports, locks, and validation |
| `src/hlt_classification/scouting/ladder_workflow.py` | task command construction and artifact consumption |
| `src/hlt_classification/scouting/ladder_campaign.py` | smoke, pilot, and production DAG/resource registry |

CLIs remain thin argument/delegation layers. Slurm scripts only establish the
authenticated environment and `exec` the Python task runner.

### 27.3 Port order

1. Package and validate the three donor resources and selected config.
2. Port scalar edge features and compare exact fixtures.
3. Port empirical and consensus score matrices and their finite masks.
4. Port lexicographic assignment and compact/native index conversion.
5. Port all 18 diagnostics and calibrated confidence.
6. Connect the matcher to the repository `ParticleSet` decoder.
7. Implement dense shards, manifest, and validators.
8. Add deterministic per-source assignment workers and merge.
9. Add sampled ROOT recomputation and complete selected-role count audit.
10. Connect authorized assignments to high-coverage repair.
11. Prove training code cannot invoke the matcher after assignment lock.

Each step receives focused parity tests before its output is consumed by the
next step.

### 27.4 Parity acceptance

Golden fixtures cover unequal counts, no gated edge, all dustbins, ties,
permutations, charged/neutral transitions, lost-track offsets, single-particle
jets, and high multiplicity. The port must reproduce:

- identical gate masks;
- empirical and consensus scores within a recorded FP64 tolerance;
- identical compact and native assignment indices;
- identical maximum cardinality;
- all 18 diagnostic values within tolerance;
- calibrated confidence within tolerance;
- exact semantic hashes for all selected resources.

It also reproduces the donor audit summaries within declared sampling and
floating-point tolerance. Those summaries are regression checks, not new truth
claims for the Scouting population.

## 28. Repair implementation and endpoint qualification

### 28.1 Family registry

The repair dispatcher gains exact entries for:

```text
HIGHCOV_SHELL_EXACT/v1   primary HCWDL family
HIGHCOV_SHELL_SOFT/v1    endpoint diagnostic
HIGHCOV_HC_EXACT/v1      high-confidence-core diagnostic
```

Each declares its assignment schema, confidence transform, alpha endpoint,
participating matches, and required offline arrays. Unknown families or schema
mismatches fail closed.

### 28.2 Shell Exact mechanics

`a_i(alpha)` is computed in FP64 from reconstructed uint16 confidence. Alpha
zero and one are explicit branches. At alpha one, continuous values are
assigned directly from the offline endpoint so exactness does not depend on an
interpolation rounding path. Discrete groups select the offline endpoint by
construction.

At D100, each non-dustbin shell token equals its projected offline 21-field
endpoint under the canonical transform. Every dustbin, padded token, mask bit,
and token position equals HLT. Visible count is exactly the HLT count.

### 28.3 Qualification matrix

Before a ladder spec is created, identical architecture, recipe, roles, and
seed train these label-only teachers:

| ID | View |
|---|---|
| T0 | HLT |
| TFS | existing fitted-strict endpoint |
| THC | high-confidence Shell Exact endpoint |
| TSOFT | full-shell Shell Soft alpha one |
| TSHELL | full-shell Shell Exact D100 |
| TOFF | native offline |

The report includes all required metrics, T0-to-TOFF gap recovery, and TSHELL
strata by matched-token fraction, matched-pT fraction, mean confidence, and
dustbin fraction.

This is qualification, not data-driven endpoint selection. A finite TSHELL
result worse than T0 still completes. Switching the primary family requires a
new plan/repair/spec version and explicit user choice before the ladder spec.

### 28.4 Qualification lock

The artifact is called `shell_endpoint_qualification_lock`. It proves the
primary endpoint ran on authenticated pilot roles, endpoint invariants passed,
comparison and resource reports exist, the user saw or explicitly waived the
diagnostic, and the plan still names Shell Exact v1. It never encodes “best
endpoint wins.”

## 29. Exact row selection

### 29.1 Source roles

HCWDL reuses the authenticated whole-file-disjoint 60/20/20 split. A source
file cannot contribute to multiple roles. The manifest records exact realized
class counts, mapped entries, file lists, and source hashes.

### 29.2 Pilot

Pilot selection is exactly 300,000 train, 100,000 validation, and 100,000
final-test jets when the immutable roles contain sufficient valid mapped rows.
Within each role, per-class counts preserve natural proportions through a
deterministic largest-remainder allocation with an exact total. Within each
class, stable identity hashing selects rows; file-prefix order is forbidden.

If a class cannot fill its allocation, selection fails with a shortfall report.
Another class is not silently substituted. All 23 nodes use the identical row
selection.

### 29.3 Smoke

Smoke uses no more than 4,096 train and 4,096 validation identities selected by
the same proportional identity-hash rule. It cannot access final test. Every
smoke assignment, target, and input cache is bounded to those identities. A
separate bounded cache miniature exercises RAM behavior; smoke coverage cannot
authorize a full role.

### 29.4 Production

Production consumes every valid mapped identity in each realized role. It does
not replace actual counts with nominal 3M/1M/1M values. The authoritative
count-only audit requires scanned mapped jets equal expected mapped jets and
exact visible/assigned category conservation before authorization.

## 30. Seeds, repeats, and controls

### 30.1 Screening

The pilot executes all 23 primary nodes with one predeclared screening seed.
That run establishes graph-wide feasibility and localizes effects; one seed is
not used to claim statistical certainty.

### 30.2 Five-seed confirmation

After screen aggregation, five predeclared replicate seeds execute:

- M0;
- D0c and D0w;
- M1c and M1w;
- M6c and M6w;
- the best intermediate cold ascent node and best intermediate warm ascent
  node chosen by the frozen validation rule;
- required same-generation self-KD/null controls.

The screening seed is included among the five only if declared before
confirmation. Every confirmation node keeps the identical 60-pass, every-pass
validation, recipe, domain, and selector. Attractive unregistered nodes cannot
be added after inspecting the screen.

### 30.3 Minimum null controls

The confirmation registry includes at least:

- fresh CE-only M0 for every confirmation seed;
- an M1-capacity self-KD student taught by M0 with the same single-teacher
  recipe;
- a predecessor-only counterpart to M6, using a predeclared normalized control
  recipe rather than silently dropping loss mass;
- a label-only warm-continuation control if recipe evidence indicates warm LR
  sensitivity.

Control coefficients and initialization are frozen in the recipe. Controls
are separate graph nodes and cannot overwrite primary artifacts.

### 30.4 Production multiplicity

Production code can materialize both full ladders. Actual submission requires a
post-pilot authorization that names tracks, nodes, and seeds. Dropping a track
changes the production campaign identity; it is not represented by deliberately
unsatisfiable dependencies.

## 31. Metrics and scientific estimands

### 31.1 Required metrics

Every node reports:

- multiclass cross entropy;
- overall accuracy and balanced accuracy as distinct values;
- macro and per-class one-vs-rest AUC;
- macro mean log QCD rejection at 50% signal efficiency;
- per-class QCD false-positive rate and rejection at 50% signal efficiency,
  explicitly including Xbb/QCD and Xcc/QCD;
- top-label 15-bin ECE and multiclass Brier score;
- confusion matrix and natural class counts;
- checkpoint-selection inputs and all 60 validation records.

Always-QCD accuracy is included as context for an imbalanced population.
Overall accuracy is never labeled balanced accuracy.

### 31.2 Gap recovery

For a metric where larger is better:

```text
recovered_fraction(X; lower, upper)
  = (metric_X - metric_lower) / (metric_upper - metric_lower)
```

For CE, signs reverse so positive recovery means improvement. A denominator at
or below a declared numerical tolerance yields `undefined`, not zero.

Reports include endpoint recovery from T0 to TOFF, downward retention at every
D rung, M1 bottom-rung transfer, each ascent increment, final M6 transfer, and
paired warm-minus-cold differences.

### 31.3 Ordered questions

1. Does M1 improve over M0?
2. Does each ascent rung improve over its immediate predecessor?
3. Does M6 improve over M1 and M0?
4. Does the warm system improve over the cold system?
5. How much of the D100-to-TOFF gap remains because Shell Exact is still a
   fixed-HLT-skeleton completion?
6. How much privileged advantage reaches HLT-only M6?

All comparisons are reported regardless of sign. No minimum performance gain
is an execution-success criterion.

## 32. Selection and final-test sealing

### 32.1 Screen selector

All screen and confirmation decisions use validation only. Within a declared
candidate set, the order is higher macro AUC, lower CE, higher mean log
rejection, then stable graph ID. The full candidate table and hexadecimal
comparison values are preserved.

### 32.2 Finalist set

Before test assignments exist, the finalist lock freezes:

- M0;
- M6c and M6w;
- no more than one predeclared intermediate ascent node per track;
- D100 and TOFF oracle diagnostics;
- required confirmation/null controls.

A weak M6 is not replaced with an unregistered attractive model. Exact graph,
checkpoint, report, recipe, assignment, and source hashes enter the lock.

### 32.3 Execution claim

The execution lock binds the test row identities, finalist lock, matcher
resources, repair, assignments, recipe, architecture, and pushed source commit.
Final inference atomically claims that lock once.

The test is not reopened to choose an epoch, coefficient, temperature, repair,
rung, seed, or track. A technical retry runs the identical claim; it cannot
change the candidate.

## 33. Campaign phases and dependency graph

### 33.1 Phases

**A. Port and local parity:** package resources, port matcher, dense assignment,
repair, ladder graph, and contracts. No cluster data are rewritten.

**B. Real-worker miniature:** exercise production assignment, cache,
teacher-target, D-node, M-node, validation, resume, and report workers on
bounded real Scouting rows. Measure RAM, storage, throughput, and walltime.

**C. Recipe and endpoint qualification:** publish the immutable recipe, build
pilot train/validation assignments once, run the endpoint matrix, and publish
the explicit Shell Exact qualification lock.

**D. Primary pilot:** train roots, both sequential down tracks, and both
sequential up tracks. Cold and warm tracks can run concurrently when their
parents exist.

**E. Confirmation and sealed test:** aggregate, freeze the five-seed registry,
confirm, select finalists, create the execution lock, build test assignments
once, and perform one test wave.

**F. Production decision:** use pilot science and measured resources to create
a distinct authorization. Pilot success does not implicitly authorize it.

### 33.2 Dependency graph

```text
source audit -> split manifest -> data lock
  -> matcher resource validation
  -> pilot row selection
  -> train/validation assignments -> manifest -> assignment lock
  -> real-worker cache miniature -> resource report
  -> recipe evidence -> recipe lock
  -> endpoint qualification matrix -> Shell Exact qualification lock
  -> roots: M0 || D100 || TOFF
       D100 -> D75c -> D50c -> D25c -> D0c -> M1c -> ... -> M6c
       D100 -> D75w -> D50w -> D25w -> D0w -> M1w -> ... -> M6w
  -> screen aggregate -> confirmation registry lock
  -> five-seed confirmation -> finalist lock -> execution lock
  -> test rows -> one-time test assignments -> sealed evaluation
  -> aggregate report
```

Dependencies express artifact availability, never a performance threshold.
Endpoint disappointment, teacher disappointment, or a negative rung gain
cannot suppress registered descendants.

### 33.3 Job granularity and arrays

Assignment workers are arrays by source file or deterministic source shard.
Each training node is a separate checkpointable job. Confirmation replicates
may be arrays keyed by immutable registry index. Locks and aggregation are CPU
tasks.

HCWDL leaves array concurrency uncapped by default. A measured filesystem,
GPU, RAM, or site limit may impose a recorded cap, but a cap cannot change row
selection or scientific meaning.

## 34. Slurm, memory, and storage plan

### 34.1 Worker environment

Every worker:

- uses account `reu-aisocial` and partition `tigris`;
- receives absolute `PROJECT_DIR` and campaign-spec paths;
- activates `atlas_kd_tigris`;
- sets `PYTHONNOUSERSITE=1`;
- prepends `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`;
- finishes its shell with `exec python -s ...`;
- uses `--signal=B:USR1@120` for checkpointable training;
- writes only below its campaign root or declared temporary directory;
- records exact job and array IDs in the ledger.

### 34.2 Resource classes

Separate measured resource classes cover CPU assignment shards, CPU merge and
audit, GPU roots, GPU single-teacher nodes, GPU dual-teacher nodes, CPU locks
and aggregation, and GPU sealed evaluation.

Pilot D-view workers may request 300 GB or more if measurement justifies it and
the selected node class supports it. Requests include headroom. A production
view that cannot fit safely uses declared sharding or assignment-backed
streaming; it does not persist an undocumented reconstructed dataset.

### 34.3 Storage

Durable artifacts are restricted to manifests, dense assignments, optional
compact logits, selected/final model checkpoints, one rolling resume state,
reports, locks, ledgers, and logs. Repaired datasets and epoch copies are
forbidden.

A prelaunch estimator uses actual visible-token counts to calculate assignment
bytes and sums worst-case concurrent cache/checkpoint storage with headroom.
Transient caches are removed only after validation and their last declared
consumer. Scientific reports and selected checkpoints remain recoverable.

### 34.4 Walltime and preemption

Walltimes come from genuine miniature throughput with a documented safety
factor. Full-role assignment audits receive independent allocations. Assignment
shards restart independently and reuse already validated shards. All 60-pass
training nodes publish exact resume state at validation boundaries and USR1.

## 35. Versioned artifacts and proposed layout

### 35.1 Contract families

Implementation introduces separately versioned contracts for:

- `HIGHCOV_MATCHER_RESOURCES/v1` and validation report;
- `HIGHCOV_DENSE_ASSIGNMENT_SHARD/v2`;
- `HIGHCOV_DENSE_ASSIGNMENT_MANIFEST/v2` and authorization lock;
- `HIGHCOV_REPAIR/v1`;
- `HCWDL_RECIPE/v3` and recipe lock;
- `HCWDL_NODE_SPEC/v1` and graph registry;
- `HCWDL_TRAINING_REPORT/v1`, checkpoint selection, and resume;
- `HCWDL_CAMPAIGN_SPEC/v2`, `HCWDL_COMMAND_PLAN/v1`, submission ledger,
  submission authorization v3, and monitor report;
- endpoint qualification, confirmation registry, finalist, execution, and
  aggregate-report contracts.

Reusable JSON contains contract name, schema version, canonical content hash,
parent hashes, source/split lineage, producer commit, and semantic config.
Binary artifacts have byte hashes and logical hashes when serialization bytes
are not scientific identity. Path existence alone never authorizes reuse.

Old PMARD artifacts remain under old consumers. New code does not silently
accept old meanings under a broadened schema.

### 35.2 Proposed tree

```text
campaign_root/
  campaign_spec.json
  source/{source_audit,split_manifest,row_selection}.json
  matcher/
    resources_validation.json
    train/*.npz
    validation/*.npz
    assignment_manifest.json
    assignment_audit.json
  locks/
    data_lock.json
    assignment_lock.json
    recipe_lock.json
    shell_endpoint_qualification_lock.json
    confirmation_registry_lock.json
    finalist_lock.json
    execution_lock.json
  qualification/<qualifier>/...
  training/
    roots/{M0,D100,TOFF}/...
    cold/{D75,D50,D25,D0,M1,M2,M3,M4,M5,M6}/...
    warm/{D75,D50,D25,D0,M1,M2,M3,M4,M5,M6}/...
    confirmation/<seed>/<node>/...
  final/
    matcher/test/*.npz
    evaluations/<finalist>/...
    aggregate_report.json
  recovery/{monitor_report,resume_ledger}.json
  submission_ledger.json
```

Exact paths become contractual with implementation. Node identity comes from
the registry, never from directory-name parsing.

## 36. CLI, failure, and recovery surface

### 36.1 Thin commands

The implementation provides these thin command surfaces (a mechanical rename
must be reflected by a plan amendment before code review):

```text
scripts/validate_highcov_resources.py
scripts/build_highcov_assignment_shard.py
scripts/finalize_highcov_assignments.py
scripts/build_hcwdl_recipe.py
scripts/create_hcwdl_campaign.py
scripts/run_hcwdl_task.py
scripts/train_hcwdl_node.py
scripts/select_hcwdl_checkpoint.py
scripts/aggregate_hcwdl_campaign.py
scripts/submit_hcwdl_campaign.py
scripts/monitor_hcwdl_campaign.py
scripts/resume_hcwdl_campaign.py
scripts/cancel_hcwdl_campaign.py
sbatch/run_hcwdl_task.sh
```

The task runner dispatches immutable registry tasks for resource validation,
assignment, qualification, each training node, locks, aggregation, and sealed
evaluation. It does not construct scientific configuration from shell
environment variables.

Commands support non-mutating validation/dry-run modes where appropriate.
Subprocess arguments are converted to strings before launch. Every consumer
validates exact repair, alpha, matcher, resource, recipe, teacher, graph, and
source parents.

### 36.2 Scientific versus execution failure

Bad performance is a completed scientific result. Missing inputs, corrupt
hashes, incompatible schemas, invalid particle identity, nonfinite required
tensors/metrics, incomplete scans, forbidden test access, or mismatched
checkpoint lineage are execution failures and fail closed.

### 36.3 Resume and imported prefixes

Recovery validates completed artifacts, reuses completed assignment shards and
completed prefix nodes, resumes interrupted nodes, and submits missing
descendants with exact dependencies. Imported prefix artifacts keep original
producer, path, and hash lineage and gain an explicit import record; they are
not made “new” by copying.

### 36.4 Exact cancellation

Cancellation consumes exact IDs from the campaign ledger, never broad job-name
patterns. A recovery ledger records old and replacement IDs without deleting
the original submission history.

## 37. Required test matrix

### 37.1 Matcher and assignments

- donor feature, score, assignment, native-index, diagnostic, and confidence
  parity;
- resource hash, feature-order, bins, coefficients, scales, and isotonic
  tampering rejection;
- maximum-cardinality-before-score and private-dustbin proof;
- ties, permutations, rectangular/empty cases, and high multiplicity;
- source-fold scorer identity and train-fold leakage rejection;
- lost-track neutral offsets and one-to-one assignment;
- uint16 round trip and confidence endpoint behavior;
- exact row/token/category conservation;
- perfect observed coverage over 9 of 10 expected jets must fail;
- a train or validation role with `dustbin_fraction >= 0.10` must fail the
  assignment authorization, while `dustbin_fraction < 0.10` must be calculated
  from complete-role token counts rather than averaged jet fractions;
- sampled ROOT recomputation and final-test pre-lock denial.

### 37.2 Repair

- exact alpha-zero HLT identity;
- Shell Exact warp values and monotonicity in alpha/confidence;
- exact offline endpoint for every assigned D100 token;
- exact HLT identity for dustbins and padding;
- Shell Soft and HC Exact diagnostic behavior;
- all 21 fields, wrapped phi, p4 physicality, order, masks, and counts;
- identity/quality/track applicability coupling;
- deterministic switches nested across domains and invariant to batch, epoch,
  worker, and resume.

### 37.3 Training

- exact 60 passes and 60 validations;
- final partial batch retained and each identity once per pass;
- deterministic AUC/CE/log-rejection/earliest tie handling;
- BF16 model execution with FP32 CE/KD/reductions;
- one-time teacher and view construction;
- HLT-only M stream after targets exist;
- cache/stream equivalence and target-join rejection;
- exact warm reset and true-resume restoration;
- USR1 reaches Python and produces resumable state.

### 37.4 Graph and campaign

- all 23 primary nodes and exact Section 21.1 edges;
- M0 absent from M1 initialization and loss;
- fresh cold starts and selected-weight warm starts;
- correct teacher input domains and no cross-track leakage;
- descendants remain registered for deliberately poor finite metrics;
- every smoke command has row bounds;
- default uncapped arrays and recorded operational caps;
- exact-ID recovery/cancellation;
- final-test isolation and one execution claim.

### 37.5 Acceptance ladder

Local acceptance requires focused tests, the complete suite, and
`git diff --check`. Runtime acceptance additionally requires donor fixture
parity, installed-Weaver parity, the full smoke DAG, a genuine Tigris miniature
using production workers, a measured resource report, and a full pilot dry run
from an exact pushed commit.

## 38. Implementation blocks and definition of done

### 38.1 Blocks

1. **Contracts and resources:** package donor resources, validators, semantic
   hashes, legacy source map, and parity fixtures.
2. **Matcher and dense cache:** port matcher, connect ParticleSet, build shards,
   manifest, count audit, and recomputation audit.
3. **Repair families:** implement Shell Exact/Soft/HC Exact through the existing
   21-field repair path.
4. **Ladder engine:** implement the registry, cold/warm semantics, two-teacher
   loss, every-pass validation, AUC selection, RAM caches, and resume.
5. **Orchestration:** implement modes, DAG, locks, CLI, Slurm, resources,
   monitoring, recovery, cancellation, and aggregate reporting.
6. **Real acceptance:** Weaver parity, smoke, Tigris miniature, resource
   measurements, recipe resolution, and full pilot dry run.

### 38.2 Completion means runnable

Implementation is complete only when no donor repository is needed at runtime;
all contracts validate; every node and edge is executable in smoke and pilot
specs; one-time assignment and RAM reuse are demonstrated; the genuine Tigris
miniature passes; the pilot dry run emits exact commands, dependencies,
resources, and paths without submission; and `docs/HANDOFF.md` records evidence
and launch blockers.

Files existing without those behaviors do not satisfy this plan.

## 39. Explicit unresolved-value ledger

| Value | Resolution evidence | Must be frozen before |
|---|---|---|
| pilot/production RAM and time | genuine Tigris miniature | live submission |
| production tracks/seeds | pilot results and resource report | production spec |

The following are fixed, not unresolved: single-teacher CE/KD `0.25/0.75`;
primary dual-teacher CE/predecessor/privileged weights `0.25/0.40/0.35`;
privileged and HLT temperatures `2/1`; every peak LR `3e-4`; effective and
microbatch 256 with accumulation one; the five-percent warmup/cosine/five-percent
minimum schedule; AdamW betas/epsilon/decay and disabled clipping; Shell Exact v1; domains
0/0.25/0.50/0.75/1; complete cold and warm pilot ladders; 60 passes per
pilot/production primary node; validation every pass; macro-AUC-first checkpoint
selection; one-time matching; 300k/100k/100k pilot roles; and HLT-only
deployable inference.

## 40. Claims, limitations, and authorization boundary

The shell is offline-directed completion, not proven truth matching. D100 is
exact only for shell assignments and retains HLT dustbins, count, and order.
Confidence is calibrated to donor synthetic corruption families and is not a
guaranteed real-particle probability. The calibrator is not per-fold
cross-fitted.

A D100-to-TOFF gap can arise from dustbins, wrong shell edges, fixed HLT
skeleton, excluded lost tracks, feature projection, or optimization. Failure to
transfer can arise from teacher weakness, incompatible targets, capacity,
optimization, or information unavailable at HLT. Warm versus cold answers an
initialization-system question only under the locked recipe.

Reports must preserve those qualifications and never describe an M model as
using offline information at inference.

This plan authorizes local implementation and verification, not live
submission. A live pilot requires the completed implementation, clean exact
pushed commit, validated assignments, measured resources, immutable recipe,
completed Shell Exact qualification with explicit user awareness, full dry
run, and explicit user launch instruction. Production additionally requires
completed pilot evidence and a distinct production authorization.

# HCWDL TRI100 Four-Spine Full-Cardinality Bottleneck Matching Implementation Plan

Status: proposed implementation authority. This document freezes the
scientific and execution semantics for a controlled rerun of the single-GH200
HCWDL TRI100 four-spine LOGIT campaign with one changed variable: the
HLT-to-offline particle pairing rule. No production campaign described here is
implemented or authorized merely by the existence of this plan.

## 1. Objective

The established HCWDL path uses a high-coverage correspondence matcher that
may abstain when a candidate pair fails its physical-plausibility gate. This
study asks a deliberately different question:

> If every particle on the smaller side of the paired HLT/offline jet is
> assigned one-to-one, and the assignment makes the worst selected angular
> separation as small as mathematically possible before improving the second
> worst, third worst, and so on, does the four-spine LOGIT ladder preserve more
> offline performance as it approaches exact HLT?

This is a controlled **full-cardinality pairing** experiment. It is not a new
claim that every forced pair is a physical detector-level correspondence.

The intended comparison is:

```text
established four-spine campaign
    existing abstaining high-coverage matcher

versus

new four-spine campaign
    full-cardinality lexicographic bottleneck-delta-R pairing
```

All model, data, loss, seed, optimization, validation, checkpoint-selection,
resource, and reporting semantics remain fixed unless this plan explicitly
states otherwise.

## 2. Claim boundary and terminology

The new object must be called one of:

- full-cardinality bottleneck pairing;
- lexicographic delta-R pairing;
- forced full-cardinality pairing oracle.

It must not be called:

- a truth match;
- a proven physical correspondence;
- a high-purity match;
- a calibrated particle-match probability.

The established matching research plan distinguishes physical
correspondence, set transport, and forced-assignment controls. This study sits
in the forced-assignment/control category. Its scientific value is whether a
complete, geometry-first alignment is a better training coordinate system for
the HCWDL ladder, even when some selected pairs are not defensible physical
correspondences.

No pairing index, delta-R, matching diagnostic, homotopy coordinate, source
index, or offline feature becomes a deployable model input. Every trained
model continues to consume only the registered particle view, and the D000
endpoint remains exact HLT.

## 3. Registered pairing cardinality

For one paired jet, let:

```text
n_h = number of valid HLT particles
n_o = number of valid offline particles
k   = min(n_h, n_o)
```

The selected relation must contain exactly `k` one-to-one pairs.

Therefore:

- if `n_h <= n_o`, every HLT particle is paired and `n_o - n_h` offline
  particles remain unused;
- if `n_h > n_o`, every offline particle is paired and `n_h - n_o` HLT
  particles remain unpaired;
- if either side is empty, the relation is empty;
- an HLT particle and an offline particle may each appear in at most one pair;
- padded particles are never eligible;
- there is no delta-R, response, charge, category, anchor, or confidence gate.

The complete bipartite graph over the valid particles is the feasible edge
set. This guarantees that a cardinality-`k` assignment exists for every
finite paired jet.

The durable orientation remains HLT-to-offline: each valid HLT particle stores
one native offline index or `-1`. When `n_h <= n_o`, no valid HLT index may be
`-1`. When `n_h > n_o`, exactly `n_h - n_o` valid HLT indices must be `-1`.

## 4. Exact optimization objective

### 4.1 Canonical angular distance

For every valid candidate edge `(i, j)`:

```text
deta(i,j) = eta_hlt(i) - eta_offline(j)
dphi(i,j) = wrap_to_minus_pi_plus_pi(
                phi_hlt(i) - phi_offline(j)
            )
dr(i,j)   = sqrt(deta(i,j)^2 + dphi(i,j)^2)
```

The implementation must compute this from the authenticated particle
four-vectors used by the current matcher. It may not use unwrapped phi,
jet-axis distance in place of pairwise distance, or a learned score as the
primary cost.

For cross-platform deterministic identity, delta-R is canonicalized to an
integer before optimization:

```text
DR_QUANTUM = 1e-7
qdr(i,j)   = round_half_to_even(dr(i,j) / DR_QUANTUM)
```

The constant, rounding mode, input dtype, wrapped-phi convention, and resulting
integer matrix are versioned scientific semantics. They may not be altered
without a new matcher contract version.

### 4.2 Primary lexicographic bottleneck objective

For a feasible cardinality-`k` assignment `A`, form the vector of selected
canonical distances and sort it from largest to smallest:

```text
D(A) = sort_descending([qdr(i,j) for (i,j) in A])
```

The selected assignment is the assignment whose `D(A)` is lexicographically
smallest.

This means, in order:

1. minimize the worst selected delta-R;
2. subject to the best possible worst delta-R, minimize the second-worst;
3. subject to both of those, minimize the third-worst;
4. continue until all `k` selected distances are fixed.

This is stronger and more specific than minimizing mean delta-R, total
delta-R, or the maximum alone. A reduction in many easy-pair distances may
never compensate for a worse earlier element of `D(A)`.

Because the feasible assignment set is finite and nonempty, a global optimum
always exists. The production implementation must solve this exact finite
objective; a weighted approximation, ordinary sum-cost Hungarian assignment,
greedy nearest-neighbor rule, or threshold-only bottleneck solution is not
conforming.

### 4.3 Secondary tie breakers

Only after the complete primary vector `D(A)` is identical may other particle
information influence the result. The tie-break sequence is frozen as:

1. lexicographically minimize the descending vector of canonical absolute log
   transverse-momentum responses;
2. minimize the number of particle-category mismatches;
3. minimize the number of valid-charge mismatches;
4. lexicographically minimize the tuple of offline native indices ordered by
   HLT native index, treating `-1` as larger than every real index.

The pT-response canonicalization and validity rules must be versioned in the
matcher specification. Category and charge are never hard gates. No secondary
term may trade against any element of the primary delta-R vector.

### 4.4 Exact solver requirement

The implementation must provide both:

- a small exact reference enumerator for tests; and
- a production exact lexicographic bottleneck solver.

The production solver should use ranked canonical delta-R levels and repeated
cardinality-feasibility/constrained matching to resolve the primary vector,
then resolve secondary ties on the exact primary-optimal face. It must never
encode the hierarchy in floating-point weights.

The implementation is acceptable only if, for exhaustively enumerable
rectangular matrices, it exactly matches the reference assignment signature,
including duplicate costs, count imbalance, empty sides, and secondary ties.

## 5. Relationship to the established matcher

The current high-coverage matcher:

- applies physical-plausibility gates including finite delta-R and response
  limits;
- first maximizes the number of plausible accepted edges;
- then maximizes an empirical aggregate edge score;
- may abstain and stores calibrated correspondence confidence.

The new matcher:

- has no candidate gate over valid particles;
- always reaches maximum possible one-to-one cardinality;
- minimizes the ordered tail of selected delta-R;
- does not estimate physical-correspondence probability;
- keeps old artifacts immutable and creates a separately versioned assignment
  lineage.

The existing `lexicographic_assignment` helper is not the solver specified by
this plan. Its current meaning is lexicographic in cardinality and aggregate
score, not lexicographic in the ordered selected delta-R vector. It must not be
silently repurposed under its existing contract or solver label.

## 6. Assignment artifact and confidence semantics

The new assignment artifact requires its own versioned contracts. At minimum:

```text
HCWDL_FULL_CARDINALITY_BOTTLENECK_MATCHER_SPEC/v1
HCWDL_FULL_CARDINALITY_BOTTLENECK_ASSIGNMENT_SHARD/v1
HCWDL_FULL_CARDINALITY_BOTTLENECK_ASSIGNMENT_MANIFEST/v1
HCWDL_FULL_CARDINALITY_BOTTLENECK_ASSIGNMENT_LOCK/v1
HCWDL_FULL_CARDINALITY_BOTTLENECK_DIAGNOSTIC_REPORT/v1
```

Every reusable artifact records:

- content hash and schema version;
- source commit and semantic source hashes;
- raw-data inventory, split, selection, and row-identity parents;
- matcher-spec hash and solver identity;
- exact role and source-file coverage;
- cardinality and integrity summaries;
- final-test access state.

The durable assignment payload contains only the compact native-index mapping
and the row/source identity needed to join it safely. Dense candidate matrices
must never be persisted. Per-pair delta-R may be retained only in a bounded
audit sample; full-population delta-R distributions are stored as aggregate
counts/histograms, quantiles, and extrema.

The current `confidence_u16` artifact field means calibrated evidence of
physical correspondence. It must not be populated with `1.0` merely because a
pair was forced. The implementation must either:

- add a separate pairing-validity API for HCWDL-UB construction; or
- add a new explicitly neutral pairing-provenance field and a versioned
  adapter whose value is not interpreted as correspondence confidence.

The preferred implementation is a new pairing-validity path. Existing
high-coverage cache and confidence semantics remain byte- and
contract-compatible for old campaigns. No fake confidence is authorized.

## 7. Foundation and reuse boundary

### 7.1 Reusable immutable parents

The new setup may reuse, read-only and after exact authentication:

- the raw JetClass data inventory;
- the established full-data split manifest;
- the all-authenticated-mapped-row selection manifest;
- the label/class map and feature schema;
- the offline and HLT endpoint definitions;
- the established full-data recipe where this plan does not override it;
- installed-Weaver parity and prior worker evidence whose semantics are
  genuinely unchanged.

Role counts and exact identities must match the established full foundation.
The expected scale is the existing all-mapped population, approximately
2.78 million train rows and 0.96 million validation rows. The final test role
remains sealed and unused.

### 7.2 Artifacts that must be rebuilt

Changing particle pairing invalidates every descendant that depends on that
pairing. The new lineage must rebuild:

- assignment shards, manifests, audits, and assignment lock;
- assignment-dependent scale calibration;
- residual/coupling artifacts;
- balanced sidecars or equivalent compact view-construction metadata;
- foundation/source consumer locks that bind the new matcher;
- every downstream four-spine target bank and fit report.

No old assignment, coupling, sidecar, or intermediate U/D target bank may be
copied into the new campaign merely because its path exists.

### 7.3 U000 reuse correction and gate

Code inspection shows that the current U000 coordinate (`s=0, f=0`) returns
the native prepared offline P0 view before assignment-dependent shell or
support construction. Therefore U000 is assignment-independent under the
current registered builder.

The scientifically clean comparison shall reuse the same authenticated U000
selected checkpoint and complete T=2/T=1 probability bank as the established
four-spine campaign. This keeps the common starting model literally identical
and makes particle pairing the only causal change.

Reuse is fail-closed. Before campaign creation, an immutable U000 equivalence
lock must prove:

- identical train and validation row identities and order;
- identical P0 input tensors under the old and new assignment parents on all
  rows through either streaming hashes or an equivalent complete proof;
- identical labels, weights, recipe, model architecture, seed aliases,
  selected checkpoint, and probability-bank identities;
- no runtime sibling-worktree imports;
- no final-test access.

If any U000-relevant identity or tensor differs, campaign creation fails. It
must not silently retrain U000 or continue with a near match. A fresh-U000
variant would be a separately planned experiment because it would introduce a
second changed variable.

## 8. Controlled four-spine campaign

The branch graph is exactly the registered single-GH200 four-spine graph:

```text
DIRECT
U000 -> D000

COARSE
U000 -> U050 -> U100 -> D066 -> D033 -> D000

DENSE
U000 -> U033 -> U066 -> U100 -> D080 -> D060 -> D040 -> D020 -> D000

ULTRADENSE
U000 -> U020 -> U040 -> U060 -> U080 -> U100
     -> D090 -> D080 -> D070 -> D060 -> D050
     -> D040 -> D030 -> D020 -> D010 -> D000
```

There are 29 fresh downstream fits and 25 new reducers. U000 is the one shared
read-only anchor and does not count as a fresh fit. There are no probability
ensembles, M1 fits, branch selection, or final-test jobs.

U and D coordinates retain their established meanings. In particular:

- U changes support while keeping paired fields offline;
- D uses full HLT support and moves paired fields toward HLT;
- D000 is exact HLT and does not expose assignment indices to the model;
- unavoidable unpaired HLT particles remain native HLT tokens in
  assignment-dependent intermediate views according to the existing HCWDL-UB
  construction rules.

Every student is freshly initialized and uses only the immediately preceding
selected model as its teacher. There is no checkpoint continuation.

## 9. Frozen training semantics

Every new fit uses:

- the same full authenticated train and validation identities as the existing
  single-GH200 four-spine campaign;
- constant `25% CE + 75% LOGIT KD`;
- temperature `T=2`;
- global batch size 256 on one GH200 process;
- the same per-coordinate initialization, sampler, repair, view, and model
  seed aliases as the existing campaign;
- unweighted CE and validation every pass;
- exact macro-AUC checkpoint selection, then minimum CE, maximum logR50, and
  earliest update as deterministic tie breakers;
- restoration of the exact selected checkpoint;
- no rolling resume or partial optimizer/checkpoint continuation.

The optimization schedule also remains the existing four-spine schedule:

```text
passes  1-3:   linear warmup to 3e-4
passes  4-45:  hold at 3e-4
passes 46-60:  cosine decay to 1.5e-5
passes 61-100: constant 1.5e-5 refinement
```

Early stopping remains:

- maximum 100 passes;
- minimum 60 completed passes;
- patience 15 validation passes;
- meaningful patience reset only for `delta macro-AUC > 5e-5`;
- patience accumulation before pass 60;
- exact checkpoint selection independent of the patience threshold.

Recent P90/H45 optimization results are scientifically interesting but are
not adopted here. Changing both matching and learning-rate schedule would
destroy the intended one-variable comparison. A later factorial study may
cross matcher type with optimization schedule under a new plan.

Weak performance, extreme selected delta-R, or failure to improve over the
established matcher is a valid scientific result. Such outcomes never fail a
fit or prune a registered downstream rung.

## 10. Target materialization

The shared authenticated U000 bank is imported read-only through the new U000
equivalence/consumer lock. After each nonterminal new fit, one reducer runs the
selected checkpoint over complete train and validation identities:

- train probabilities are stored at T=2 for the next KD edge;
- validation probabilities are stored at T=1 for auditing;
- every bank has exactly one component and complete identity coverage;
- every bank binds its report, checkpoint, split, graph, recipe, view,
  assignment, and campaign lineage;
- target banks from the established matcher are never substituted at an
  assignment-dependent rung.

Particle views, complete edge matrices, hidden states, optimizer states, and
rolling-resume state are not durable.

## 11. Required matching diagnostics

### 11.1 Integrity and cardinality

For every jet and in aggregate, report and validate:

- `n_h`, `n_o`, and selected count `k`;
- exact equality `selected_count == min(n_h, n_o)`;
- HLT-side and offline-side coverage;
- 100% smaller-side coverage for every nonempty jet;
- unavoidable unpaired HLT count `max(n_h - n_o, 0)`;
- unused offline count `max(n_o - n_h, 0)`;
- in-bounds native indices;
- one-to-one injection with no duplicate offline index;
- finite candidate and selected quantities;
- deterministic recomputation agreement on an authenticated sample.

Any violation above is an integrity failure and fails closed.

### 11.2 Delta-R tail quality

Report selected-pair delta-R:

- mean and standard deviation as descriptive values;
- quantiles 50, 75, 90, 95, 99, 99.5, and 99.9%;
- global maximum;
- per-jet maximum quantiles and maximum;
- worst, second-worst, third-worst, fifth-worst, and tenth-worst rank profiles
  where defined;
- counts and fractions above delta-R thresholds 0.01, 0.02, 0.05, 0.10,
  0.20, 0.30, 0.50, and 1.00.

These are reportable scientific observations, not pass/fail thresholds.

### 11.3 Sliced diagnostics

The same coverage and delta-R-tail diagnostics are sliced by:

- jet class;
- jet pT and eta bins;
- HLT and offline multiplicity bins;
- count imbalance `n_h - n_o`;
- HLT particle pT rank/quantile;
- particle category;
- valid charge state.

### 11.4 Comparison with established matching

On identical rows, report:

- established and new HLT/offline coverage;
- number and fraction of identical selected pairs;
- established accepted pairs retained or displaced;
- delta-R distribution for newly forced pairs;
- delta-R changes for HLT particles paired by both strategies;
- per-jet changes in worst selected delta-R;
- behavior in the count-imbalanced and low-confidence slices.

The report must keep established calibrated confidence separate from the new
pairing validity. It may describe the old confidence distribution of retained
and displaced edges but may not manufacture a new confidence calibration.

## 12. Model-level reporting

The primary validation aggregate contains:

- shared U000;
- all 29 established-matcher rows as they become durably available;
- all 29 new-matcher rows;
- completed and selected pass counts;
- accuracy, macro one-vs-rest AUC, and macro R50;
- AUC and linear-R50 recovery with the same M0CE60=0%, U000=100% convention;
- per-class QCD background rejection and recovery, including Hbb, Hcc, and
  Hqq;
- branch-local changes at every edge;
- matched-rung new-minus-established differences;
- the complete matching diagnostic summary and its lock hash.

The new campaign must not depend on completion of the currently running
established four-spine campaign. It may launch once its own pairing foundation,
U000 equivalence lock, and execution gates are complete. Cross-campaign report
rows are filled only from authenticated durable reports that exist; missing
established rows are marked pending and never block the new ladder.

Final-test inference remains sealed.

## 13. Isolation and scheduling

The new experiment has its own:

- detached clean worktree pinned to an exact pushed commit;
- campaign and foundation roots;
- job prefix, proposed as `hcwsp4b`;
- matcher specification and assignment lock;
- command plan, journal, submission ledger, attestations, monitor, and
  restart-from-zero recovery lineage.

It must not cancel, hold, reprioritize, depend on, or write into the existing
TRI100 four-spine campaign or any TRI60, DX, RSET, RREL, CE5, SD5, D000 screen,
or other user campaign.

The four branch heads depend only on the new campaign preflight and the shared
U000 consumer lock, so they may run concurrently. Later jobs depend only on
the previous reducer in the same branch. Aggregate completion depends on all
four terminal D000 fits.

The training and reducer resource requests remain those of the established
single-GH200 four-spine campaign unless measured matcher-specific evidence
requires a versioned operational change:

- fit: one node, one task, 72 CPUs, 320 GiB, one GH200, three days;
- reducer: one node, one task, 72 CPUs, 192 GiB, one GH200, one day;
- CPU lock/report jobs: 4 CPUs and 32 GiB.

The full-assignment builder receives a separately measured CPU/RAM/walltime
profile before live publication. Resource tuning may change execution only; it
may not change assignment results.

## 14. Storage and memory policy

The implementation must remain compatible with limited research-compute
storage:

- complete bipartite delta-R/cost matrices exist only per jet in worker RAM;
- no dense particle-pair matrix is durable;
- compact native-index assignments are sharded and atomically published;
- full-population diagnostic distributions are reduced to bounded histograms,
  counters, and quantiles;
- only a bounded deterministic audit sample retains detailed per-pair values;
- training views are constructed in RAM and not cached durably;
- no rolling resume or optimizer-state archive is written;
- every task publishes an output-size inventory;
- preflight refuses live submission if free-space reserve or conservative
  durable-output projection is violated.

Temporary files are written only beneath the new campaign root, are enumerated
before deletion, and are removed after their immutable replacement is
validated.

## 15. Failure policy

Fail closed for:

- stale or dirty source checkout;
- missing or mismatched parent/content hashes;
- invalid row identity or role coverage;
- nonfinite geometry or required metrics;
- cardinality below `min(n_h, n_o)`;
- duplicate/out-of-bounds assignment indices;
- production/reference solver disagreement;
- nondeterministic recomputation;
- fake or overloaded correspondence confidence;
- U000 equivalence failure;
- forbidden final-test or offline deployable-input access;
- corrupt/incomplete target banks or checkpoints.

Do not fail or cancel later registered work because:

- selected delta-R is large;
- newly forced pairs look physically implausible;
- validation accuracy, AUC, R50, calibration, or recovery is poor;
- a denser branch loses to a coarser branch;
- the established matcher wins.

## 16. Implementation map

Reusable scientific semantics belong under `src/hlt_classification/`.
Proposed new modules are:

```text
src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_contracts.py
src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_matcher.py
src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_cache.py
src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_diagnostics.py
src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_foundation.py
src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_source.py
src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_campaign.py
src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_recovery.py
```

The established TRI100 graph/training/reducer machinery should be reused where
its scientific semantics are identical. Thin CLIs and Slurm workers belong in
`scripts/` and `sbatch/`. Existing high-coverage and four-spine contracts must
not be mutated to make old and new artifacts appear interchangeable.

Any donor code copied from another worktree or commit must be recorded with
its donor path and commit in `docs/LEGACY_SOURCE_MAP.md`.

## 17. Required tests

### 17.1 Solver unit tests

- square and rectangular cardinality;
- both count-imbalance directions;
- empty-side behavior;
- wrapped-phi boundary cases;
- unique and duplicated canonical delta-R values;
- a counterexample where minimum total delta-R differs from lexicographic
  bottleneck delta-R;
- a counterexample where minimum maximum alone does not minimize the second
  worst edge;
- pT/category/charge tie-break ordering;
- deterministic final native-index tie break;
- exhaustive equality with the reference enumerator for bounded random
  matrices;
- invariance to worker count and shard boundaries.

### 17.2 Artifact and foundation tests

- content-hash and parent-hash tamper rejection;
- complete role/source/entry coverage;
- atomic publication and exact sampled recomputation;
- no dense matrices or fake confidence in durable payloads;
- all assignment-dependent old artifacts rejected as parents;
- identical split/selection reuse accepted only by hash;
- complete U000 old/new P0-view equivalence proof;
- one changed assignment bit invalidates every dependent source lock.

### 17.3 Campaign tests

- exact four branch paths;
- exactly 29 fresh fits and 25 reducers;
- immediate-parent single-teacher edges only;
- C25/P75, T=2, batch 256, and single-GH200 semantics;
- exact 60-to-100 floor-tail/early-stop behavior;
- matched seed aliases with the established campaign;
- no M1, ensemble, final-test, rolling-resume, or DDP task;
- no dependency on the running established four-spine ledger;
- dry-run command-plan and recovery closure;
- weak scientific metrics still complete all rows.

Focused tests must pass before implementation handoff, followed by the
repository testing ladder in `docs/TESTING.md`.

## 18. Tigris production-readiness gates

No standalone scientific smoke campaign is required. Before full submission,
the production DAG includes non-scientific acceptance gates:

1. exact pushed commit in a clean detached worktree;
2. complete local focused tests;
3. installed-Weaver parity for all touched view construction;
4. a genuine Tigris production-worker matcher miniature on authenticated real
   rows, including brute-force-checkable low-multiplicity jets;
5. measured assignment-builder CPU, RAM, walltime, and output bytes;
6. full assignment-manifest dry run and resource projection;
7. complete U000 equivalence/consumer lock;
8. single-GH200 forward/backward acceptance using the new view source;
9. canonical full command plan and dry-run submission ledger;
10. exact live-creation and live-submission authorization phrases.

The matcher miniature is an integrity/resource gate, not a pilot whose
scientific metrics control whether the registered full campaign runs.

## 19. Implementation phases

### Phase A: contracts and exact solver

Implement the mathematical objective, exhaustive reference, production
solver, artifact schemas, and focused unit tests.

### Phase B: full-data assignment foundation

Build compact sharded assignments, diagnostics, audit, calibration/coupling
descendants, and an immutable new foundation/source lock from the established
split and selection.

### Phase C: U000 equivalence and source binding

Prove complete P0 independence from assignment and bind the established U000
checkpoint/probability bank as a read-only common anchor. Fail closed on any
difference.

### Phase D: four-spine campaign integration

Reuse the registered single-GH200 graph and training semantics under the new
assignment source. Add isolated campaign creation, workers, reducers,
aggregation, monitoring, and restart-from-zero recovery.

### Phase E: verification and queue handoff

Run focused tests, installed-Weaver parity, genuine Tigris matcher and
single-GPU acceptance, exact dry run, storage audit, and self-review. Publish a
runbook with exact commit-pinned creation, audit, submission, monitoring, and
recovery commands.

## 20. Completion criteria

Implementation is queue-ready only when all of the following are true:

- the exact full-cardinality objective is contractually frozen and tested;
- the production solver matches exhaustive reference results;
- full-data assignment/foundation artifacts authenticate completely;
- smaller-side coverage is exactly 100% on every nonempty jet;
- pairing diagnostics and old-versus-new comparison are durable;
- U000 identity/equivalence is proven and locked;
- the campaign graph has exactly 29 fresh fits and 25 reducers;
- every scientific and seed semantic matches the established single-GH200
  four-spine campaign except the declared pairing lineage;
- local tests, installed-Weaver parity, and genuine Tigris acceptance pass;
- the command plan and dry-run ledger audit cleanly;
- output-size projections preserve the required free-space reserve;
- the source commit is pushed and the detached Tigris worktree is clean;
- no existing campaign has been modified or operationally disturbed;
- final test remains sealed;
- `docs/HANDOFF.md` records evidence rather than anticipated success.

Only after these criteria and explicit user authorization may the full
campaign be submitted.

## 21. Non-goals

This campaign does not:

- claim forced pairs are physical truth correspondences;
- learn or calibrate a new match probability;
- tune a delta-R gate;
- change C25/P75, temperature, batch size, schedule, seeds, or branch graph;
- compare LOGIT with RSET/RREL;
- add ensembling, M1 compression, or final-test inference;
- select a preferred branch based on interim results;
- replace the established high-coverage matcher in other campaigns.

Those are separate scientific questions and require separate versioned plans.

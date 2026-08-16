# HCWDL Multi-Horizon Projection-Ensemble Full-Data Plan

Status: **implementation-authoritative additive full-data successor plan**.

Short name: **HCWDL-MHPE-FULL**. `MHPE` means **multi-horizon projection
ensemble**. The informal phrase "teacher lattice" refers to the same graph.
This is not AdaBoost: no row weight is changed from a previous model's error.

This plan is a focused successor to the
[unified-root balanced homotopy plan](HCWDL_UNIFIED_BALANCED_HOMOTOPY_IMPLEMENTATION_PLAN.md)
and the
[all-mapped full-data plan](HCWDL_UNIFIED_BALANCED_FULL_DATA_THREE_ARM_PLAN.md).
It preserves their unified 21-channel model, projected-offline `U000` root,
balanced atomic U coordinate, uniform deterministic D coordinate, exact HLT
endpoint, unweighted population loss, prepared-endpoint implementation,
all-mapped row meaning, and fail-closed lineage. It changes the training graph:
several historical teachers are projected independently into each new student
domain, and those same-domain specialists are ensembled before the next stage.

The campaign is additive. It does not mutate, resume, relabel, or reinterpret
an existing HCWDL-UJ, HCWDL-UB, FULL3, or FULLCOARSE3 execution. Existing
results are immutable contextual controls. The new campaign receives its own
contracts, root, command plan, ledger, reports, and recovery closure.

## 1. Scientific question

A sequential KD ladder has one causal path:

```text
U000 -> U050 -> U100 -> D066 -> D033 -> D000
```

Every student is an imperfect projection of its teacher. Information lost at
one edge cannot re-enter a descendant, so small errors can accumulate like a
telephone game. Earlier direct privileged-to-HLT KD did not show evidence of
damaging the HLT endpoint, even when its gain was modest. The stronger unified
`U000` root also removes native TOFF's architecture boundary.

HCWDL-MHPE-FULL asks whether teacher horizons contain complementary knowledge
after each has been independently projected into the same current input
domain. At a target coordinate, one fresh specialist is trained per declared
historical teacher. Only after those projections are complete are their
same-domain probabilities averaged. The ensemble then becomes one of the
teachers available at later coordinates.

The primary hypotheses are:

1. a uniform same-domain ensemble outperforms the mean and preferably the best
   individual specialist at one or more target coordinates;
2. long skip teachers preserve knowledge that the local predecessor ensemble
   has lost, while the local ensemble provides better domain alignment;
3. the exact-HLT `D000E` ensemble exceeds the paired CE-only HLT baseline and
   the best individual exact-HLT specialist;
4. a fresh exact-HLT `M1` can compress most of `D000E` using stronger
   same-domain KD.

These are scientific hypotheses, not execution gates. Every finite result,
including a harmful ensemble or a failed hypothesis, completes the graph and
is reported.

## 2. Reused full-data foundation

The intended execution reuses the completed authenticated all-mapped
HCWDL-UB-FULL3 foundation through a new immutable reuse lock. Reuse is
permitted only when the lock proves exact equality of:

- source commit and frozen semantic-source hashes;
- split manifest and all-mapped train/validation identity sets;
- assignment, residual-coupling, balanced-switch, endpoint, and resource
  locks;
- unified model factory, input projection, balanced U builder, uniform D
  builder, cache, optimizer, checkpoint, and resume semantics;
- the selected 20-pass CE-only `U000` report and checkpoint bytes;
- the paired exact-HLT `M0paired` report and checkpoint bytes;
- the compact identity-ordered `U000` target-bank lineage;
- final-test access count of zero.

Path existence is never reuse evidence. A missing or mismatched parent fails
closed. The campaign must not silently retrain a nominally similar root on a
different population. If no exact completed foundation exists, implementation
must stop and document that blocker rather than invent a replacement.

Conceptually, `U000` is the first model in the lattice. Operationally, it is
an imported authenticated root and is not a fresh fit owned by this campaign.
No native `TOFF` checkpoint supplies a primary target.

## 3. Population and access boundary

The campaign uses every authenticated mapped train and validation row from
the reused full-data foundation. Approximate descriptions such as
`2.6M/1M/1M` are never identities. The immutable spec stores exact integer
counts, ordered identity-set hashes, and source inventories derived from the
foundation.

Ordinary campaign work may read:

- all mapped training rows for optimization and teacher-target generation;
- all mapped validation rows for every-pass metrics, checkpoint selection,
  ensemble evaluation, and descriptive diagnostics.

Final-test rows remain sealed. No ordinary task selects, opens, assigns,
couples, projects, caches, or evaluates a test row. A separately versioned
finalist lock and explicit human execution lock are required before any test
access. Sealed evaluation consumes exact HLT inputs only.

## 4. Exact coordinates and endpoint meanings

All student views use the implemented unified-balanced carrier

```text
V_UB(s, f)

s = balanced structural progress from projected offline support to D100
f = uniform matched-field progress from offline values to HLT values
```

The campaign freezes the following exact rational coordinates:

| Name | Exact view | Meaning |
|---|---|---|
| `U000` | `V_UB(0, 0)` | projected-offline unified root |
| `U050` | `V_UB(1/2, 0)` | half structural information-mass progress |
| `U100` | `V_UB(1, 0)` | exact authenticated D100 input |
| `D066` | `V_UB(1, 1/3)` | exact two-thirds offline matched-field strength |
| `D033` | `V_UB(1, 2/3)` | exact one-third offline matched-field strength |
| `D000` | `V_UB(1, 1)` | byte-exact canonical HLT input |

`D066` is a symbolic node label for exact offline strength `2/3`; it does not
mean floating-point `0.66`. Likewise `D033` means exact `1/3`. Specs store
integer numerator/denominator pairs and exact float-hex encodings. U switches
sample the already frozen rung-grid-independent balanced sidecar. D fields use
the already frozen uniform deterministic repair, including nested immutable
discrete-group switches. Match confidence is not a D-coordinate input.

No endpoint, coupling, information-mass, switch, interpolation, PID, charge,
validity, track, ordering, mask, padding, or raw-length semantics change in
this plan.

## 5. Exact teacher lattice

### 5.1 Canonical node IDs

The graph uses explicit source-qualified specialist IDs:

```text
U050_from_U000

U100_from_U000
U100_from_U050
U100E

D066_from_U000
D066_from_U050
D066_from_U100E
D066E

D033_from_U000
D033_from_U050
D033_from_U100E
D033_from_D066E
D033E

D000_from_U000
D000_from_U050
D000_from_U100E
D000_from_D066E
D000_from_D033E
D000E

M1
```

An ID ending in `E` is an immutable probability-ensemble target artifact and
validation row, not a trainable model or checkpoint. Every `_from_` ID is one
fresh specialist checkpoint that consumes the target coordinate named before
`_from_` and receives KD from the source named after it.

`U050` is the human-readable alias of canonical checkpoint
`U050_from_U000`. Every machine-readable teacher edge, target manifest, and
report uses the full canonical ID. No second U050 checkpoint exists.

### 5.2 Dependency graph

```text
U000
  |
  +--KD--> U050_from_U000                         := U050

U000 --------KD--> U100_from_U000 --+
U050 --------KD--> U100_from_U050 --+--> U100E

U000 --------KD--> D066_from_U000 ---+
U050 --------KD--> D066_from_U050 ---+--> D066E
U100E -------KD--> D066_from_U100E --+

U000 --------KD--> D033_from_U000 ---+
U050 --------KD--> D033_from_U050 ---+
U100E -------KD--> D033_from_U100E --+--> D033E
D066E -------KD--> D033_from_D066E --+

U000 --------KD--> D000_from_U000 ---+
U050 --------KD--> D000_from_U050 ---+
U100E -------KD--> D000_from_U100E --+--> D000E
D066E -------KD--> D000_from_D066E --+
D033E -------KD--> D000_from_D033E --+

D000E -------KD--> M1
```

Every arrow is a KD edge. A teacher model is evaluated on its own declared
input domain, and its outputs are joined to the student solely by canonical
jet identity. An ensemble teacher supplies an authenticated identity-ordered
probability bank constructed from specialists that all consume the ensemble's
declared coordinate.

The graph contains 16 fresh fits:

```text
1 U050 specialist
+ 2 U100 specialists
+ 3 D066 specialists
+ 4 D033 specialists
+ 5 D000 specialists
+ 1 M1 compression student
= 16 fresh fits
```

Including imported `U000`, the conceptual study has 17 model checkpoints.
The four ensemble artifacts add no fits. Within each stage, specialists are
independent and may run concurrently; stages remain causally sequential.
Peak intended training concurrency is five GPUs at the D000 stage.

### 5.3 No hidden multi-teacher loss

Each specialist has exactly one teacher. The implementation must not replace
the declared specialists with one student receiving several KD terms. This is
the central scientific distinction:

```text
declared:  teacher_k -> target-domain specialist_k -> same-domain ensemble
forbidden: average incompatible source-domain teachers -> one specialist
```

Separate projection prevents one optimizer from resolving gradients from
several differently privileged domains before those targets have been made
expressible in the common student domain.

## 6. Specialist training protocol

Every non-M1 specialist uses the exact full-data `C25P75` recipe:

```text
fresh initialization
0.25 unweighted per-jet CE
0.75 single-teacher KD
KD temperature 2
20 complete natural-population passes
validation after every pass
macro AUC, then CE, then logR50, then earliest update selection
locked full-data AdamW, batching, warmup/cosine, and BF16-forward policy
FP32 CE, softmax, KL, coefficient multiplication, and population reduction
no performance early stopping
```

The existing evidence favors `C25P75` and, critically, pure KD has not been
the robust endpoint recipe. This first lattice deliberately fixes one recipe
so teacher horizon is the intended difference. Recipe sweeps are separate
future campaigns.

All specialists at the same target coordinate share initialization, sampler,
dropout, optimizer, validation-order, and repair seed aliases. Only the
teacher target differs. This pairing makes specialist differences attributable
to teacher horizon rather than random initialization. No checkpoint or
optimizer state is warm-started.

Twenty full-data passes are intentional. They expose the optimizer to more
jet presentations than 60 passes over the historical 300k population while
keeping the triangular graph tractable. Changing pass count, loss weights,
temperature, optimizer schedule, or pairing seeds is a contract version
change.

## 7. Ensemble semantics

### 7.1 Uniform project-then-ensemble target

Suppose target-coordinate specialists have selected FP32 logits `z_k(x)` in
canonical class order. At declared temperature `T`, the ensemble distribution
is

```text
p_E,T(x) = (1/K) * sum_k softmax(z_k(x) / T)
```

The component order is lexical by canonical specialist ID. Softmax uses a
max-subtracted FP32 input, components are accumulated in FP64 in canonical
order, division is FP64, and the published distribution is rounded once to
little-endian FP32. Every row must be finite, nonnegative, and sum to one
within the contract tolerance. Changing reduction order, weights, temperature,
class order, or rounding changes the ensemble contract.

Raw logits are never averaged. Model weights are never averaged. A metric
average is not an ensemble.

The primary ensemble weights are exactly uniform and may not depend on labels,
validation performance, teacher confidence, class, jet, or rung. A weak but
finite specialist remains included. Validation-driven pruning or learned
stacking would need a separate calibration population and a new contract.

### 7.2 Temperatures and stored targets

`U100E`, `D066E`, and `D033E` publish temperature-2 probability banks for
their cross-domain child specialists. Each also publishes or reproducibly
derives its temperature-1 probabilities for validation metrics.

`D000E` publishes:

- temperature-1 probabilities for same-domain M1 KD and validation metrics;
- temperature-2 probabilities as a diagnostic only, so comparisons with the
  earlier stages remain interpretable.

An ensemble target is consumed directly as the left distribution in

```text
T^2 * KL(p_E,T || softmax(student_logits / T))
```

It must not be passed through another temperature transform as though it were
an ordinary logit vector. The target kind and temperature are explicit in the
loss configuration and report.

### 7.3 Ensemble artifacts

Each ensemble has immutable train and validation shards plus a manifest and
lock. A shard binds:

- canonical jet identities and exact role;
- ordered component node IDs, report hashes, selected-checkpoint hashes, and
  component-logit hashes;
- target coordinate and view-contract hash;
- class order, temperature, uniform rational weights, numerical policy, dtype,
  logical array hash, and byte hash;
- split, row-selection, foundation-reuse, graph, recipe, and producer hashes;
- explicit `final_test_accessed: false`.

Train ensemble probabilities are KD targets. Validation probabilities support
ensemble metrics, component comparison, disagreement, and leave-one-out
diagnostics. They never select or rewrite a component checkpoint.

Compact logit/probability targets are authorized because they have multiple
consumers. Durable reconstructed particle views remain forbidden.

## 8. M1 ensemble compression

`D000E` combines five models that all consume byte-exact HLT. `M1` is a fresh
unified 21-channel exact-HLT model trained with:

```text
0.10 unweighted CE
0.90 D000E probability KD
temperature 1
20 complete passes
validation every pass
the same checkpoint-selection and optimizer policy as specialists
```

Temperature 1 is appropriate because teacher ensemble and student consume the
same exact HLT input and the arithmetic ensemble is already softened by model
disagreement. Ten-percent CE provides label correction and calibration without
recreating the repeated 25% CE injection that motivated the study.

This v1 plan registers no 95/5 or pure-KD M1. Such a compression sweep is a
small separately versioned follow-up. `M1` is the designated single-model
deployment candidate.

Every D000 specialist and the `D000E` ensemble are also technically HLT-only.
`D000E` requires five forward passes and is the uncompressed ensemble ceiling,
not the designated single-model result.

## 9. Controls and comparisons

No additional control fits are required beyond the lattice because each
stage contains its own direct and local specialists. The campaign imports by
exact hash:

- full-data `M0paired` as the CE-only HLT denominator;
- full-data `U000` as the privileged upper bound and root;
- completed same-population sequential `C25P75` factorized rows when they are
  explicitly bound at spec creation;
- native TOFF and architecture-factorial rows as contextual evidence only
  when their exact populations and lineages are disclosed.

The immutable contextual-report list is frozen in the campaign spec. Reports
are never discovered later by scanning directories.

For every ensemble stage, reporting must include:

1. every specialist's validation metrics and selected pass;
2. the uniform ensemble metrics;
3. ensemble minus mean-specialist and ensemble minus best-specialist deltas;
4. pairwise probability correlation, Jensen-Shannon divergence, classwise
   disagreement, and prediction disagreement;
5. every leave-one-specialist-out ensemble metric, computed for diagnosis
   without changing the primary graph;
6. local-predecessor specialist versus each longer-skip specialist;
7. runtime, GPU-hours, target-build time, and cache memory.

The ordered primary endpoint comparisons are:

```text
D000E - best D000 specialist
D000E - D000_from_D033E
D000E - M0paired
M1    - D000E                 (compression gap)
M1    - best D000 specialist
M1    - M0paired
```

Recovery of the `M0paired -> U000` macro-AUC, CE, and logR50 gaps is reported
without clipping. Macro OVR AUC is primary. CE, accuracy, balanced accuracy,
per-class AUC, Brier score, ECE, confusion matrix, logR50, R50, and complete
histories are mandatory companions.

One full-data seed is an exploratory result. It provides a cleaner statistical
signal than the 300k study but is not a substitute for paired replication.

### 9.1 Sealed final evaluation

Validation does not prune the registered HLT endpoint set. After ordinary
completion, the finalist lock registers exactly:

```text
M0paired
D000_from_U000
D000_from_U050
D000_from_U100E
D000_from_D066E
D000_from_D033E
D000E
M1
```

All eight consume exact HLT only. A separate explicit human execution lock is
still required before the sealed final-test job exists. That one job evaluates
the frozen eight-entry set on the same test identities, publishes the five
component predictions before forming `D000E` with the identical uniform
probability rule, and validates that the recomputed ensemble matches its
registered numerical policy. Test metrics do not change checkpoints,
components, weights, graph, or the designated single-model result.

## 10. Persistence and efficient execution

The implementation starts from a clean descendant containing the corrected
prepared-endpoint and all-mapped assignment-identity fixes documented in the
[ragged preprocessing guide](../HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md).

Required execution behavior:

- reconstruct each specialist's train and validation student views once in
  process-local RAM and replay them for all 20 passes;
- prepare each complete ragged source chunk a fixed number of times, never
  below a per-row loop;
- precompute or load each multi-consumer teacher target bank once;
- reuse the authenticated U000 train target bank under the new five-consumer
  authorization (`U050`, `U100`, `D066`, `D033`, `D000` specialists);
- publish U050 logits once for its four direct consumers;
- publish each ensemble probability bank once for its declared downstream
  consumers;
- never persist U/D particle datasets, epoch copies, model activations, or
  optimizer-independent RAM caches;
- retain exact checkpoint/resume state and request `USR1` 120 seconds before
  every GPU walltime;
- use the absolute source-pinned `${PROJECT_DIR}` worker path and required
  Tigris environment variables.

The scheduler shape is:

```text
authenticate foundation and publish reuse/graph locks
  -> U050 specialist
  -> two U100 specialists in parallel -> U100E
  -> three D066 specialists in parallel -> D066E
  -> four D033 specialists in parallel -> D033E
  -> five D000 specialists in parallel -> D000E
  -> M1
  -> validation aggregate and completion
```

A reducer never begins until every registered component report and target
shard for that ensemble validates. Poor metrics do not block it. Missing,
corrupt, nonfinite, cross-role, or cross-lineage inputs fail closed.

## 11. Contracts and implementation map

Implementation introduces a new additive family, provisionally:

```text
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FOUNDATION_REUSE_LOCK/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_SHARD/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_MANIFEST/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_LOCK/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_COMMAND_PLAN/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_STAGE_REPORT/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_AGGREGATE/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINALIST_LOCK/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_EXECUTION_LOCK/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_COMPLETE/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECOVERY_SPEC/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RESOURCE_RECOVERY_SPEC/v1
HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v1
```

Existing HIGHCOV assignment, residual coupling, balanced switch, uniform
repair, U000 target, generic submission-ledger, task-attestation, checkpoint,
resume, and metric contracts may be reused only by exact validated hash.

Preferred implementation is additive:

| Surface | Required work |
|---|---|
| new graph module | exact 16-fit triangular registry, dependency levels, teacher domains, seeds, and deployability |
| new ensemble-target module | uniform temperature-specific probability reduction, shards, manifests, locks, and identity joins |
| training adapter | explicit probability-target KL alongside existing logit KD without changing old reports |
| runner/workflow | one-time views/targets, specialist training, ensemble reducers, M1, reporting, and completion |
| campaign layer | foundation authentication/reuse lock, source-pinned spec/plan, dry run, live submission, and access boundary |
| recovery | exact failed/downstream closure, completed-stage preservation, source and resource recovery |
| Slurm | thin absolute-path workers and dependency-aware submission with stage-local parallelism |
| reporting | specialist, ensemble, leave-one-out, diversity, recovery, compression, and resource tables |

Old contract validators and campaign graphs must continue to accept only their
original identities and bytes.

## 12. Recovery and failure behavior

The source-pinned recovery path must exist before live submission. It binds
one immutable campaign spec, original ledger, immutable monitor, exact source
commit, foundation reuse lock, graph, recipe, and every completed artifact.
It schedules exactly the failed/downstream closure:

- a failed specialist invalidates only its stage reducer and later stages;
- a failed ensemble reducer preserves every completed specialist;
- a failed downstream stage never reruns a completed upstream stage;
- a source repair may change execution code only under a separately versioned,
  exact-file-allowlisted repair contract that proves scientific semantics are
  unchanged;
- a resource recovery may only increase CPUs, RAM, or walltime and may not
  change GPU class, rows, graph, loss, views, seeds, or targets;
- repeated recovery remains supported and preserves the complete lineage
  chain.

Cancellation uses exact campaign-bound job IDs from one ledger. Broad name
matching is forbidden. Finite poor science completes; invalid source, data
access, identity coverage, lineage, probability, loss, or checkpoint state
fails closed.

## 13. Required tests and audits

### Graph and loss

- exact 16 fresh fits, four ensemble artifacts, one imported U000 root, and
  every declared edge;
- exact dependency-level concurrency `1,2,3,4,5,1` after U050;
- every specialist has exactly one teacher and uses `C25P75/T=2`;
- M1 alone uses `C10P90/T=1` from `D000E`;
- paired target-coordinate seed aliases and cold initialization;
- no hidden grandparent, multi-teacher, warm-start, or joint-path edge;
- exact 20 passes, 20 validations, final partial batch, and checkpoint order.

### Probability ensembles

- hand-calculated two-, three-, four-, and five-component distributions;
- raw-logit averaging rejection;
- canonical component ordering, FP64 accumulation, FP32 publication, and
  worker/shard-order invariance;
- temperature-1 and temperature-2 behavior and no double-temperature bug;
- exact identity/class coverage, uniform weights, row sums, finiteness, and
  logical hashes;
- corrupt component, checkpoint drift, cross-role, duplicate/missing identity,
  class-order drift, and target-temperature rejection;
- analytic probability-target KL and gradient equivalence fixtures;
- deterministic ensemble, leave-one-out, and diversity metrics.

### Views, lineage, and access

- exact `U050`, `U100`, `D066`, `D033`, and `D000` rational coordinates;
- U100 byte equality to authenticated D100 and D000 byte equality to HLT,
  including raw lengths above 200;
- complete all-mapped train/validation coverage and zero ordinary test reads;
- exact foundation reuse and five-consumer U000 authorization;
- no label enters view construction or uniform ensemble weights;
- no durable repaired-particle dataset;
- one-time prepared chunk, view, and target construction counts;
- final-test denial before both locks.

### Campaign and operations

- topological absolute-path command plan and exact dependency IDs;
- dry-run nonmutation, clean-source drift rejection, and immutable spec paths;
- poor-science continuation and fail-closed invalid artifacts;
- USR1 checkpoint/resume including GPU RNG mapping;
- exact-ID cancellation and exact failed/downstream recovery;
- resource-only recovery preserves every scientific hash;
- aggregate registry, imported-context freeze, GPU-hours, and completion;
- all CLI help, contract identity, shell/static, Markdown-link, and
  `git diff --check` checks.

Before live submission, implementation must pass focused tests, the complete
repository suite, compilation, a bounded local synthetic end-to-end exercise,
all CLI/help and contract-version checks, a complete nonmutating full-data dry
run, source/recovery tests, and documentation review.

## 14. Explicit direct-to-full-data and no-smoke decision

At the user's explicit direction, HCWDL-MHPE-FULL will not require or submit:

- a new standalone Slurm smoke campaign;
- a new 300k pilot;
- a reduced-row GPU miniature as a prerequisite to the full-data graph.

Once implementation and the checks in Section 13 are complete, the next live
scientific execution is the all-mapped full-data campaign itself. This is a
deliberate campaign-specific decision, not a claim that the new graph passed a
smoke.

The implementation must publish an honest operational-evidence waiver that
binds:

- the completed production-worker FULL3 foundation and corrected preprocessing
  lineage;
- prior installed-Weaver parity and model/training-worker evidence;
- the exact new pushed source and semantic diff;
- unchanged U/D/view/model/checkpoint/resume worker hashes;
- focused probability-ensemble and probability-KD tests;
- the bounded local synthetic graph result and complete full-data dry run;
- the authenticated foundation reuse lock and resource profile;
- the user's decision to proceed without a new Slurm miniature or pilot;
- the residual risk introduced by the new ensemble reducer and DAG.

The waiver may not say that HCWDL-MHPE-FULL completed a smoke. Foundation
authentication, graph/recipe locking, target-consumer authorization, command-
plan validation, and resource checks form a fail-closed prefix of the real
campaign. They are not a second miniature campaign and do not train reduced
models.

A clean full commit, clean detached Tigris worktree, complete dry run,
authenticated reuse lock, recovery path, exact resource request, and explicit
live-submission authorization remain mandatory. This plan does not itself
authorize a push, Slurm submission, cancellation, or final-test access.

## 15. Definition of done

Implementation is queue-ready only when:

1. the new contracts and exact graph validate all 16 fresh fits and four
   ensemble stages;
2. probability-target KD is numerically and gradient tested without changing
   legacy logit-KD behavior;
3. the full-data foundation reuse lock authenticates exact rows, endpoints,
   U000/M0 checkpoints, targets, and source semantics;
4. all specialists, reducers, M1, reports, locks, ledgers, cancellation, and
   recovery tasks render in one complete nonmutating dry run;
5. no ordinary task can read final test or persist repaired particle views;
6. focused and full suites plus compilation/static/link/diff checks pass;
7. `docs/HANDOFF.md` records exact local evidence and remaining Tigris action;
8. the no-new-smoke operational waiver truthfully records carried evidence
   and residual risk;
9. the exact pushed commit and explicit live phrase are available for the
   all-mapped submission.

The final scientific result is not required to declare implementation done.
Implementation readiness means the declared campaign can execute and recover
without changing its science, not that its hypothesis has succeeded.

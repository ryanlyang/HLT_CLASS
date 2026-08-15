# HCWDL Unified-Root Balanced Homotopy Implementation Plan

Status: **implementation-authoritative successor plan; not yet implemented**.

Short name: **HCWDL-UB**. `U` denotes the structural-support path, `D`
denotes the fixed-support matched-field path, and `J` denotes their joint path.
`UB` distinguishes this successor from the immutable, currently executing
HCWDL-UJ v2 campaign.

This is a focused successor to the
[HCWDL structural-feature homotopy plan](HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md).
Everything not changed explicitly below retains the existing authenticated
HCWDL-UJ meaning: endpoint projection, residual-shell coupling, carrier
conservation, HLT-only deployment, cold starts, unweighted loss, identity-
joined FP32 logits, 60 passes, every-pass validation, macro-AUC-first
checkpoint selection, exact resume, and fail-closed lineage.

This plan does **not** modify or reinterpret the running HCWDL-UJ v2 campaign.
Its artifacts, contracts, and results remain immutable controls. HCWDL-UB uses
new versioned scientific identities and a new campaign root.

## 1. Decisions

HCWDL-UB makes four changes.

1. A freshly trained unified projected-offline model, `U000`, replaces native
   `TOFF` as the first teacher in both primary paths.
2. Structural edits retain real-particle atomicity but receive a balanced,
   rung-grid-independent deterministic switch coordinate. The coordinate is
   designed to spread information mass and edit types rather than place edits
   primarily by coupling cost.
3. The D axis uses a new uniform Shell Exact transition. Every matched slot
   has the same continuous offline strength at a given rung. Discrete groups
   make deterministic nested population-level switches independent of matcher
   confidence.
4. The first 300k pilot wave runs six predeclared CE/parent-KD/grandparent-KD
   recipes as six independent downstream campaigns that may execute
   concurrently. They share one immutable preparation/U000 foundation but
   never share mutable training state, output directories, ledgers, or
   recovery closures. This tests whether repeated 25% CE injections wash out
   inherited teacher information as the number of ladder generations
   increases without turning the comparison into one monolithic operational
   graph.

The default first graph retains the present v2 density. Rung density is a
hashed graph parameter, however, and may be changed before campaign creation
without rebuilding the structural coupling or switch coordinates.

The primary default paths are:

```text
factorized:
U000 -> U020 -> U040 -> U060 -> U080 -> U100
     -> D80F -> D60F -> D40F -> D20F -> D0F -> M1F

joint:
U000 -> J010 -> J020 -> J030 -> J040 -> J050
     -> J060 -> J070 -> J080 -> J090 -> J100 -> M1J
```

Every arrow is predecessor-logit KD. `U100` is also the factorized `D100`
endpoint; it is not trained twice. `F` means **factorized** and distinguishes
the U-then-D path from the joint J path. `D0F` and `J100` are exact HLT.
The two paths are instantiated once in each of the six recipe arms defined in
Section 9. All arms share one selected CE-trained U000 root.

Operationally, the study has two levels rather than one 151-fit submission:

```text
one shared foundation execution
  -> six separately specified and separately submitted arm campaigns
  -> one read-only cross-arm comparison aggregate
```

The foundation must complete and publish its selected U000 checkpoint and
foundation lock before any arm specification becomes executable. After that
join, the six arm campaigns have no dependencies on one another and may use
six independent GPU/Slurm frontiers.

## 2. Evidence and claim boundary

The completed 300k architecture-by-input factorial supplied the motivating
evidence:

| Model | Input | Architecture | Validation macro AUC |
|---|---|---|---:|
| `H_U` | exact HLT | unified 21-channel | 0.937727 |
| `H_S` | exact HLT | charged/noncharged split | 0.933479 |
| `O_U` | projected offline P0 | unified 21-channel | 0.948125 |
| `O_S` | projected offline P0 | charged/noncharged split | 0.945785 |

The unified architecture won on both input domains, while projected-offline
particles produced the large improvement. This supports using the unified P0
model as the ladder root and treating native TOFF as a contextual oracle.

The exact factorial aggregate, reports, checkpoints, source commit, split,
row selection, and recipe hashes must be authenticated before these results
are imported. The numeric table above is motivation, not sufficient artifact
lineage by itself.

Allowed interpretation:

- the unified 21-channel model is preferred for this campaign;
- offline particle information, rather than the charged/neutral split alone,
  explains the measured advantage;
- a unified `U000` root removes the first native-to-unified KD boundary.

Forbidden interpretation:

- P0 is tensor-identical to native TOFF;
- one exploratory seed proves universal architectural superiority;
- an HCWDL-UB result is paired with an old 300k result merely because its
  nominal count or node name is similar; exact row, checkpoint, seed, and
  parent hashes are still required.

## 3. Exact endpoint meanings

### 3.1 U000

`U000` consumes the existing P0 endpoint:

- at most the first 90 native charged/lost-track particles and first 60 native
  neutral particles visible to the registered native-offline adapter;
- the existing public projection of each endpoint into the unified 21-field
  Scouting schema;
- the exact projected offline p4 and stored endpoint-relative fields;
- a single maximum-200 unified Scouting Particle Transformer.

Unlike the earlier semantic-only `U000=P0` corner, HCWDL-UB registers and
trains `U000` itself with fresh, unweighted CE. Its selected checkpoint is the
sole first teacher of `U020`, `J010`, and the registered comparison roots.

`U000` is not deployable and is not renamed TOFF. Native TOFF remains an
optional imported reference and never supplies primary-path targets.

### 3.2 U100/D100

`U100` has the exact current D100 support and bytes:

- every assigned HLT slot contains the exact projected offline endpoint;
- every dustbin contains the exact observed HLT particle;
- HLT slot order, mask, padding, p4, all 21 fields, and raw HLT length metadata
  are byte-identical to the authenticated D100 builder.

### 3.3 D0/J100

`D0F`, the legacy-warp comparison endpoint, and `J100` are byte-identical to
canonical HLT inputs. Their subsequent M1 students also consume HLT only.
Only HLT-input nodes may enter finalist selection or sealed test evaluation.

## 4. The balanced atomic U coordinate

### 4.1 What is retained

HCWDL-UB reuses the authenticated residual-shell endpoint partition and
maximum-cardinality minimum-cost coupling from HCWDL-UJ:

```text
O -> R         real offline residual replaced by a real D100 residual
O -> padding   offline residual removed
padding -> R   D100 residual inserted
```

Each edit remains atomic. All 21 raw fields, p4, identity, charge, validity,
track state, mask state, and padding state change together. An active token is
always an exact observed endpoint record. The coupling remains an operational
carrier, not a new physical match.

The base edit identity, cost, endpoint membership, carrier location, and edit
mass remain reusable only when all existing parent hashes match. Repaired
particle views are never persisted.

### 4.2 Why the old switch coordinate changes

The v1 structural coordinate places edits through a role-global cost CDF.
That is deterministic and nested, but it can concentrate similar-cost edits
and cardinality changes into a small part of the path. A nominal U step can
therefore be globally reasonable while being abrupt for individual jets or
particular edit types.

The new coordinate deliberately stops treating coupling cost as an
activation priority. Coupling cost still contributes to information mass and
reporting, but a difficult residual edit is not automatically deferred to the
end of the U path.

### 4.3 Exact information mass

Each edit keeps a positive integer mass with the existing components:

```text
count floor
+ source/target scalar-pT share
+ source/target energy share
+ endpoint disruption cost
```

The disruption cost already contains bounded kinematic, identity/charge,
validity, track, and remaining-field components. The implementation publishes
those components separately so reports can show what each rung changed.

The mass formula, integer quantum, rounding, p4 floors, endpoint groups, and
cost table are content-hashed. Changing any of them creates a new switch
contract rather than silently altering HCWDL-UB v1.

### 4.4 Deterministic balanced placement

For each jet, edits are divided into label-free balance strata using:

```text
edit kind
source and target six-state particle categories, including unknown/dummy
source-to-target charged-applicability state
eight-bit validity-group change mask
```

Within one `(jet identity, stratum)`:

1. order edits by a domain-separated SHA-256 hash of the immutable edit key;
2. lay the ordered edits around a unit circle with arc length proportional to
   integer information mass;
3. rotate the circle by a second domain-separated hash of the jet identity and
   stratum;
4. place each edit at the rotated midpoint of its mass arc;
5. quantize that coordinate to uint16 with declared round-half-up arithmetic.

For an edit with integer mass `m`, preceding mass `c`, total stratum mass `M`,
and exact hash-derived phase `phi`, its conceptual coordinate is:

```text
u = (phi + (2*c + m)/(2*M)) mod 1
```

The implementation uses integer rational arithmetic before the final uint16
conversion. Empty strata emit no values. A one-edit stratum is placed by its
hashed phase rather than forcing every singleton to switch at the midpoint.

This construction has four useful properties:

- the coordinate is deterministic and independent of epoch, batch, worker,
  checkpoint resume, and registered rung grid;
- edits of the same operational type are spread across the whole path;
- within each jet and stratum, mass arcs are spread as evenly as indivisible
  edits permit; the combined per-jet balance is approximate and is reported;
- changing from 20-point to 10-point rungs only samples the same frozen
  coordinate more densely.

At structural progress `s`, an edit is active when its stored exact coordinate
is at or below `s`. The endpoints are explicit overrides: U000 applies no
edits and U100 applies every edit. Switches are nested and never reverse.

A jet with one dominant edit must still change discretely. The scheduler cannot
remove that physical granularity; it must report the largest single-edit mass
and realized per-jet jump rather than claim perfect smoothness.

### 4.5 Required U diagnostics

For every registered rung and role, report cumulative and incremental:

- edit count and information mass;
- substitution, insertion, and removal counts;
- scalar pT and energy affected;
- identity/category and charge changes;
- track-applicability and validity-group changes;
- active-particle counts;
- per-jet median, tail quantiles, and maximum switched mass;
- deviation from the nominal structural fraction.

These diagnostics never alter the frozen coordinate. Labels may be joined
only later for descriptive reporting and cannot enter construction.

## 5. Uniform deterministic D semantics

### 5.1 New repair family

HCWDL-UB introduces `HCWDL_UNIFORM_SHELL_EXACT/v1`. It does not modify or
relabel `HIGHCOV_SHELL_EXACT/v1`.

Let `alpha` be the offline fraction:

```text
D100: alpha = 1.0
D80:  alpha = 0.8
D60:  alpha = 0.6
D40:  alpha = 0.4
D20:  alpha = 0.2
D0:   alpha = 0.0
```

For every assigned matched slot, the continuous strength is exactly:

```text
a_i(alpha) = alpha
```

It no longer depends on matcher confidence. Confidence remains part of the
assignment provenance and diagnostics but is not a D-coordinate input.

### 5.2 Continuous fields

At an intermediate rung:

- p4 and ordinary continuous fields use the same shared linear strength;
- phi follows the wrapped shortest path;
- explicit alpha-one and alpha-zero branches copy exact offline and HLT
  endpoints respectively;
- validity/applicability disagreements follow the atomic group rules below;
- no jet axis or relative field is recomputed from the changing view.

“Linear” means a shared endpoint interpolation strength. It does not claim
equal Euclidean displacement in every transformed model channel.

### 5.3 Discrete and validity groups

Discrete values are never interpolated. Each mismatched semantic group gets
one deterministic switch variate derived from:

```text
uniform-repair contract hash
campaign discrete seed
canonical jet identity
immutable target HLT slot
semantic group name
```

The first 64 SHA-256 bits define an exact uniform integer. At offline fraction
`alpha=n/d`, the group uses the offline endpoint exactly when:

```text
hash_u64 * d < n * 2**64
```

and otherwise uses the HLT endpoint exactly. Endpoints override the comparison.
No RNG is sampled while building a batch.

Consequently, at D80 approximately 80% of mismatched groups retain their
offline value and 20% have switched to HLT. At D60 the expected fractions are
60% and 40%. Since each group has one immutable threshold, a group switches
once as alpha falls and never switches back.

Semantic atomicity is mandatory:

- charge and all five PID flags switch together as the identity group;
- quality switches as one group;
- lost-inner-hits switches as one group;
- each registered validity group switches atomically;
- a track field cannot become applicable before the identity/validity state
  that gives it meaning;
- dependent track values select the coherent endpoint whenever applicability
  or validity differs.

Thus “80% chance” is a deterministic population interpretation, not a
per-epoch coin flip. Actual finite-population fractions and their deviations
from nominal are reported by group and rung.

### 5.4 Exact D invariants

The implementation must prove:

- every matched continuous slot uses the same alpha;
- changing confidence bytes cannot change a uniform-D tensor;
- changing identity, slot, group, contract, or seed changes only the declared
  switch stream;
- PID/charge one-hot and applicability invariants hold at every rung;
- all switches are nested across arbitrary registered alpha grids;
- D100 is byte-identical to U100/current D100;
- D0 is byte-identical to canonical HLT, including raw lengths above 200.

## 6. Unified generalized view

The successor builder remains two-dimensional:

```text
V_UB(s, f)

s = balanced atomic structural progress from P0 support to D100 support
f = uniform matched-field progress from offline to HLT
alpha = 1 - f
```

Its exact corners are:

```text
V_UB(0,0) = U000/P0 input
V_UB(1,0) = exact D100
V_UB(1,1) = exact HLT
```

The factorized path samples `V_UB(s,0)` followed by `V_UB(1,f)`. The joint
path samples `V_UB(t,t)`. Structural residual tokens remain exact endpoints;
only assigned matched slots use uniform D interpolation.

For the default step-20 factorized graph:

```text
U020 = V_UB(.20,0)       J010 = V_UB(.10,.10)
...
U100 = V_UB(1,0)         J100 = V_UB(1,1)
D80F = V_UB(1,.20)
...
D0F  = V_UB(1,1)
```

Factorized and joint paths each contain ten homotopy students plus one M1.
They are edge/update matched, not exactly FLOP matched.

## 7. Rung-grid policy

The scientific coordinates are continuous and independent of graph density.
The campaign spec freezes one exact rational table before any training or
validation result is read.

The first HCWDL-UB campaign defaults to the existing v2 density:

```text
factorized U/D step: 20 percentage points per axis
joint step:          10 percentage points on both axes
```

A later 10-point or 5-point U/D graph reuses the same coupling and balanced
switch sidecars if every parent hash matches. It receives a new graph/spec
identity and cannot be substituted into an active campaign. The implementation
must not bake node names or a particular grid into the view builder.

## 8. Paired comparison with the old behavior

The running HCWDL-UJ v2 result is imported as contextual evidence. Because a
new campaign may use more rows, cross-campaign differences are not treated as
paired effects.

HCWDL-UB therefore registers two compact same-population ablations.

### 8.1 Structural scheduler ablation

From the same selected `U000` checkpoint, train:

```text
primary: U020 -> U040 -> U060 -> U080 -> U100
control: U020_legacycdf -> ... -> U100_legacycdf
```

The control uses the immutable HCWDL-UJ v1 role-global cost-CDF switch
algorithm but otherwise uses the same base coupling, root, views, optimizer,
updates, and paired transition seeds. When the selected population differs
from the old 300k campaign, its control sidecar is recomputed over the new
role using the frozen v1 algorithm and published under new exact lineage; an
old sidecar is never relabeled. Both terminal students consume identical exact
D100 tensors. Their difference isolates the structural switch schedule from
the old native-TOFF adapter boundary.

### 8.2 D-coordinate ablation

From the same selected primary U100 checkpoint, train:

```text
primary: D80F -> D60F -> D40F -> D20F -> D0F -> M1F
control: D80F_legacywarp -> ... -> D0F_legacywarp -> M1F_legacywarp
```

The control uses immutable confidence-warped `HIGHCOV_SHELL_EXACT/v1`; the
primary uses uniform Shell Exact. Rung coordinates, inputs at both endpoints,
teacher root, cold initializations, seeds, updates, loss, and selection are
paired. This is the direct test of whether uniform motion makes adjacent KD
cleaner.

### 8.3 Direct and baseline controls

The campaign also registers:

- `M0paired`: fresh CE-only exact-HLT baseline;
- `D100direct`: fresh exact-D100 student taught directly by U000;
- imported native TOFF and the four architecture-factorial cells as contextual
  rows when exact hashes are available.

No legacy joint path is required for the first campaign. The existing U/J
campaign provides contextual joint evidence, while the new primary J path
tests the complete balanced/uniform geometry.

The paired legacy-U and legacy-D branches run only in the reference
`C25P75` arm. Repeating them in every recipe arm would not improve the primary
CE/KD comparison and would add fifty-five unnecessary fits.

With the default grid and all six recipe arms, the exact overall study
registry is 151 fits, distributed across one shared foundation execution and
six independent child campaigns:

```text
2 shared roots/baselines: U000, M0paired
6 * (11 primary factorized + 11 primary joint + 1 D100direct)
5 reference-arm legacy-CDF U nodes
6 reference-arm legacy-warp D/M1 nodes
= 151 fits
```

Changing rung density changes this count and requires a new graph contract.
No individual arm ledger claims ownership of all 151 fits. The cross-arm
sweep index proves that the union of the shared foundation and six immutable
arm registries is exactly this set.

## 9. Six-arm KD/CE wave

### 9.1 Exact recipe arms

The first 300k execution contains six scientifically and operationally
separate arm campaigns that bind one completed shared foundation and one
read-only sweep index. They use the same source commit, role identities, U000
checkpoint, coupling, balanced switches, coordinates, architecture,
optimizer, updates, and transition-index seeds. Their sole primary difference
is the declared loss row:

| Arm | CE | Parent KD | Grandparent KD | Purpose |
|---|---:|---:|---:|---|
| `C25P75` | 0.25 | 0.75 | 0.00 | current reference recipe |
| `C10P90` | 0.10 | 0.90 | 0.00 | high-KD single-parent recipe |
| `C05P95` | 0.05 | 0.95 | 0.00 | very-high-KD single-parent recipe |
| `C10P75G15` | 0.10 | 0.75 | 0.15 | two-generation skip recipe |
| `C05P80G15` | 0.05 | 0.80 | 0.15 | stronger two-generation recipe |
| `C00P100` | 0.00 | 1.00 | 0.00 | pure-parent-KD boundary diagnostic |

Weights sum to one exactly. No coefficient changes with validation metrics,
training pass, rung, confidence, class, or batch.

Every arm traverses the complete factorized and joint chains. The study is
deliberately not a one-edge loss sweep: its target is cumulative generational
retention, which cannot be diagnosed from one teacher/student transition.

As a reporting diagnostic only, idealized ancestry coefficients obey:

```text
r_0 = 1
r_1 = first-edge parent weight
r_n = w_parent*r_(n-1) + w_grandparent*r_(n-2)
```

This is not asserted to equal learned information retention at temperature 2;
it makes the intended difference between recipes explicit and is reported
beside the measured teacher/student metrics.

For a homotopy transition with both teachers, the loss is:

```text
L = w_ce * CE(y, student)
  + w_parent * T^2 * KL(parent_T || student_T)
  + w_grandparent * T^2 * KL(grandparent_T || student_T)

T = 2
```

CE, both teacher softmaxes, KL terms, coefficients, and population means use
FP32. Parent and grandparent KL terms are computed and reported separately;
an implementation may optimize their algebra only if logits, loss, and
gradients remain exactly equivalent to the declared expression.

`C00P100` has no training-label gradient. Labels remain available for
validation metrics and checkpoint selection, and the report proves that the
serialized CE contribution was exactly zero. It is a diagnostic, not assumed
to be the preferred production recipe.

### 9.2 Grandparent routing

For transition one, no grandparent exists. A grandparent arm transfers its
grandparent allocation to the parent without changing CE:

```text
C10P75G15 first edge -> 0.10 CE + 0.90 parent KD
C05P80G15 first edge -> 0.05 CE + 0.95 parent KD
```

This first-edge rule applies independently to `U020`, `J010`, and
`D100direct` in those arms.

From transition two onward:

- the parent is the immediate predecessor;
- the grandparent is the parent's immediate predecessor;
- each teacher is evaluated on its own declared input view;
- both logit tables are joined to the student solely by canonical jet
  identity;
- no teacher weights or hidden representations enter the student.

The routing continues across the factorized U-to-D corner. For example, the
first factorized D student receives U100 as parent and U080 as grandparent in
the default grid. This is intentional: it is the exact two-generation skip,
not a same-view shortcut. Joint routing follows the same graph rule.

No teacher crosses recipe arms. A `C10P75G15` node can consume only
`C10P75G15` predecessors.

### 9.3 Shared foundation and six independent campaigns

`U000` and `M0paired` are trained once in a separately authenticated shared
foundation execution. The foundation also owns the exact source/split/row
selection, assignments, residual coupling, balanced-switch manifests,
coordinate table, endpoint/resource gates, selected U000 checkpoint, and the
multi-consumer U000 train-logit cache. It publishes one immutable foundation
lock that binds all of those artifacts and proves that no child-owned path is
writable beneath the foundation root.

After that foundation lock exists, the six recipe arms are created and
submitted as six genuinely independent campaigns. Each arm has its own:

- immutable arm campaign spec and command plan;
- campaign root and training/checkpoint namespace;
- Slurm submission ledger and task attestations;
- monitor and exact-ID cancellation surface;
- failed/downstream recovery closure and resource-recovery history;
- arm-local validation aggregate and completion artifact.

An arm may read the foundation lock and its named immutable children. It may
not write into the foundation root or another arm's root. Shared artifacts are
accepted only by content hash and complete parent lineage; path existence is
never sufficient. No teacher, target cache, checkpoint, rolling state,
optimizer state, or report crosses from one recipe arm to another.

Canonical node identity is `(arm_id, node_id)`, rendered as paths such as
`C10P75G15/D60F`; bare node names are forbidden in cross-arm artifacts and
Slurm task attestations. Shared roots use the reserved `shared/` namespace.

U000 is a multi-consumer teacher. Its selected-checkpoint train logits are
materialized once as an identity-ordered FP32 target cache and shared by all
first edges. Within an arm, any checkpoint needed by both a child and a
grandchild receives an authenticated compact identity-logit cache. Particle
views are not duplicated durably.

The six arm submissions have the same foundation-lock prerequisite but no
inter-arm dependency. Scheduler capacity, not an artificial array throttle,
controls their concurrency. Within an arm, the factorized and joint branches
may begin concurrently where their own dependencies permit, while every
ladder remains sequential along its KD edges. A failure in one arm does not
cancel, hold, or invalidate a valid sibling arm. Exact recovery remains
arm-local unless a shared immutable parent is corrupt, in which case every
dependent arm fails closed.

`HCWDL_UNIFIED_BALANCED_RECIPE_SWEEP/v1` is a read-only meta-index, not a
seventh training campaign and not a combined submission ledger. It binds the
foundation lock and the exact six arm-spec hashes. The cross-arm aggregate is
published only after all six arm completion artifacts exist; it reads those
immutable aggregates and never discovers arms dynamically from directories.

This separation intentionally accepts a small amount of repeated execution:
the same student coordinate may be reconstructed once in RAM by each of six
jobs on six nodes. That is six linear prepared-endpoint passes in aggregate,
not one shared pass, but it avoids cross-job mutable caches and keeps failure,
resume, and scientific lineage simple. It is expected to be hidden largely by
node-level parallelism. Durable reconstructed U/D/J particle datasets remain
forbidden.

### 9.4 Fixed M1 recipe

The final exact-HLT-to-exact-HLT M1 edge is deliberately held fixed in all six
arms:

```text
0.25 CE + 0.75 immediate-parent KD, temperature 1
grandparent weight = 0
```

This keeps the six-way comparison focused on information retention through
the changing U/D/J views. It also reports the pre-M1 exact-HLT nodes `D0F` and
`J100`, so an M1 transformation cannot conceal which homotopy recipe carried
the most knowledge. A later M1-specific weight sweep is a separate study.

### 9.5 Recipe selection and density follow-up

The six-arm wave is exploratory validation science. For the primary
factorized path, recipes are ordered by:

1. highest validation macro OVR AUC at `D0F`;
2. highest validation macro OVR AUC at `J100`;
3. highest validation macro OVR AUC at `M1F`;
4. lowest validation CE at `D0F`;
5. lexical arm ID.

All metrics and full rung curves are reported; the total order only makes the
next action reproducible. The two highest-ranked recipes may subsequently be
registered in a denser 10-point factorized follow-up to test whether reduced
generational dilution makes density beneficial. That follow-up is not
auto-submitted and receives a new graph/spec identity and explicit
authorization.

### 9.6 Shared training protocol

Unless a later predeclared optimization study changes it, every fit uses the
existing optimizer and unweighted population-reduction recipe:

```text
fresh initialization
arm-specific unweighted CE/parent/grandparent coefficients from Section 9.1
temperature 2 for every homotopy teacher
fixed 0.25 CE / 0.75 parent KD at temperature 1 for final M1 edges
peak LR 3e-4 with existing warmup/cosine policy
effective batch 256
60 complete passes
validation after every pass
macro-AUC, then CE, then logR50, then earliest-update selection
BF16 model forward and FP32 CE/KD/reduction
```

`U000` and `M0paired` use CE only. Every teacher is evaluated on its own input
domain and joined to the student solely by canonical jet identity.
Student views and multi-consumer logits are built once per job or materialized
only through compact authenticated target contracts. No student loads teacher
weights.

The U000 cache contains no particle inputs and binds the exact U000 checkpoint,
P0 domain, row identities, class order, and inference policy.

Primary/control nodes at the same transition share initialization, sampler,
dropout, optimizer, and validation-order seeds. Poor finite metrics complete
the graph.

### 9.7 Prepared-endpoint execution baseline

HCWDL-UB implementation starts from commit `f698889` or a clean descendant
that retains the corrected HCWDL ragged-preprocessing path described in
[`HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md`](../HCWDL_RAGGED_PREPROCESSING_PERFORMANCE_GUIDE.md).
`f698889` is the execution baseline, not the final campaign source identity;
every executable foundation and arm spec binds the eventual exact clean
descendant commit.

The following are implementation invariants rather than optional
optimizations:

- every source chunk constructs `PreparedOfflineEndpoints` and
  `PreparedHltEndpoints` a fixed number of times per registered endpoint
  representation, normally once;
- `build_p0_inputs`, residual partition construction, balanced-U,
  uniform-D, joint `V_UB`, calibration, and exhaustive audit paths receive
  those prepared objects explicitly;
- a helper that converts a complete ragged chunk must never be called below a
  per-row loop;
- bounded chunk workers retain at most one in-flight chunk per worker and emit
  results in canonical submission order;
- train and validation student views are each constructed once per job in
  process-local RAM and replayed for every pass;
- parent and grandparent logits are computed once per job, except for compact
  authenticated multi-consumer target caches explicitly authorized above;
- every worker reports separate train-view, validation-view, teacher-target,
  and optimizer-training phase boundaries.

Six independent arms therefore repeat only the corrected linear view build.
They must not reintroduce the former quadratic whole-chunk conversion, rebuild
views per epoch, or persist repaired particle datasets in an attempt to share
RAM across Slurm nodes. Peak-memory reasoning includes every in-flight source
chunk and the simultaneous train/validation caches. The requested worker
count may not exceed `SLURM_CPUS_PER_TASK`.

## 10. First 300k population and final-test policy

The first HCWDL-UB wave is locked to:

```text
train:       300,000 jets
validation:  100,000 jets
final test:  100,000 sealed jets
```

These are the exact authenticated role populations from the canonical
unweighted HCWDL 300k parent used by the existing pilot family. HCWDL-UB must
reuse their ordered canonical jet identities, split manifest, and row-
selection lineage; it must not draw a new 300k/100k/100k sample merely with
the same counts. Campaign creation binds the exact parent artifact hashes,
prints all three counts and identity-set hashes, and requires acknowledgement
before publication. A missing or mismatched parent fails closed rather than
falling back to reselection.

This population choice is deliberate. HCWDL-UB already changes the root
teacher, U scheduler, D coordinate, joint path, and CE/KD recipe. Holding the
rows fixed makes differences attributable to those changes, permits direct
same-population controls where seeds and all other lineage also match, and
keeps the six-arm study fast enough to complete. Scaling is deferred: after
this wave, only the predeclared best one or two recipes may be registered for
a 1M-or-larger follow-up under a new campaign/spec identity and a new explicit
authorization.

The implementation may remain count-parameterized for that later follow-up,
but the first executable graph and dry run accept only the exact counts above.

Rules:

- train rows optimize every registered model;
- validation rows select checkpoints and compare all ladder/ablation nodes;
- final-test rows are selected and sealed but are not read by coupling,
  training, checkpoint selection, or ordinary aggregation;
- after a validation-only finalist lock and a separately authorized execution
  lock, final test evaluates only `M0paired` and the selected HLT-only
  `(arm_id, node_id)` candidates among `D0F`, `J100`, `M1F`, and `M1J`;
- no offline P0/U/D construction or residual coupling is required on final
  test merely to evaluate an HLT-only finalist;
- test metrics cannot choose a different checkpoint, path, rung grid, or
  repair family.

Existing assignment and residual-coupling base artifacts are reused when and
only when their complete train/validation identity sets and parent hashes
match this fixed population. HCWDL-UB still publishes new balanced-switch
sidecars, endpoint/graph locks, views, checkpoints, targets, and reports under
its own contracts. A lineage mismatch requires a new separately authorized
campaign design; the first wave does not silently rebuild a different row
population. Test assignments are not generated unless a separately authorized
test diagnostic genuinely needs them.

## 11. Contracts and persistent artifacts

Implementation introduces new identities rather than broadening the running
campaign's contracts:

```text
HCWDL_BALANCED_STRUCTURAL_SWITCH_CONFIG/v1
HCWDL_BALANCED_STRUCTURAL_SWITCH_SIDECAR/v1
HCWDL_BALANCED_STRUCTURAL_SWITCH_MANIFEST/v1
HCWDL_UNIFORM_SHELL_EXACT/v1
HCWDL_UNIFIED_BALANCED_COORDINATE/v1
HCWDL_UNIFIED_BALANCED_NODE_SPEC/v1
HCWDL_UNIFIED_BALANCED_GRAPH/v1
HCWDL_UNIFIED_BALANCED_RECIPE/v1
HCWDL_UNIFIED_BALANCED_RECIPE_ARM/v1
HCWDL_UNIFIED_BALANCED_RECIPE_SWEEP/v1
HCWDL_UNIFIED_BALANCED_RECIPE_SWEEP_AGGREGATE/v1
HCWDL_UNIFIED_BALANCED_ENDPOINT_LOCK/v1
HCWDL_UNIFIED_BALANCED_FOUNDATION_SPEC/v2
HCWDL_UNIFIED_BALANCED_FOUNDATION_COMMAND_PLAN/v1
HCWDL_UNIFIED_BALANCED_FOUNDATION_LOCK/v1
HCWDL_UNIFIED_BALANCED_ARM_CAMPAIGN_SPEC/v1
HCWDL_UNIFIED_BALANCED_ARM_COMMAND_PLAN/v1
HCWDL_UNIFIED_BALANCED_TRAINING_REPORT/v1
HCWDL_UNIFIED_BALANCED_ARM_AGGREGATE/v1
HCWDL_UNIFIED_BALANCED_ARM_CAMPAIGN_COMPLETE/v1
HCWDL_UNIFIED_BALANCED_FINALIST_LOCK/v1
HCWDL_UNIFIED_BALANCED_FINAL_EVALUATION/v1
HCWDL_UNIFIED_BALANCED_CAMPAIGN_COMPLETE/v1
HCWDL_UNIFIED_BALANCED_RECOVERY_SPEC/v1
HCWDL_UNIFIED_BALANCED_RESOURCE_RECOVERY_SPEC/v1
HCWDL_UNIFIED_BALANCED_OPERATIONAL_EVIDENCE_WAIVER/v2
```

The old residual coupling base shards may be referenced by hash. New balanced
switch sidecars are separate immutable children; old sidecars are never
rewritten. Uniform repair calls shared endpoint projection/interpolation
primitives but has its own repair identity and tests.

The foundation spec owns shared data/preparation and the two shared fits. An
arm spec names exactly one recipe arm and binds the completed foundation lock;
it cannot contain another arm. The recipe-sweep artifact is the immutable
six-arm meta-index described in Section 9.3. Generic submission ledgers,
monitors, and task attestations may be reused only with an explicit scope key
that distinguishes `foundation` from one exact `arm_id`.

Durable storage remains limited to compact assignments/couplings/switches,
identity-ordered logits where multiple consumers justify them, checkpoints,
reports, locks, and ledgers. Reconstructed particle datasets and epoch copies
are forbidden.

## 12. Implementation map

Preferred changes are additive:

| Surface | Work |
|---|---|
| `repair.py` | expose a shared endpoint evaluator and add uniform strength/discrete switching without changing Shell Exact v1 |
| `hcwdl_upper_coupling.py` | retain base coupling; add exact balanced circular switch placement and diagnostics |
| `hcwdl_homotopy.py` | add `V_UB(s,f)` using balanced U and uniform D while preserving old `V(s,f)` |
| new graph module | register the shared U000 root, six isolated recipe arms, primary/joint paths, and reference-arm legacy controls |
| new contracts module | build and validate every HCWDL-UB v1 artifact |
| runner/workflow | reuse RAM caching, identity targets, generic training, resume, and runtime reports |
| campaign/recovery | lock the first 300k/100k/100k role identities, sealed test, dry run, exact-ID cancellation, and failed-closure recovery while keeping later scale-up separately versionable |
| reporting | compare balanced versus legacy U and uniform versus warped D with paired seeds |

Thin CLIs should create, dry-run, submit, monitor, cancel, recover, aggregate,
lock finalists, and execute the sealed test. Existing HCWDL-UJ CLIs continue
to address only their original contracts.

## 13. Required tests and audits

Focused tests must prove:

### U switching

- exact reuse/rejection of residual-coupling base lineage;
- hand-calculated strata, hashes, phase rotation, mass arcs, rational
  coordinates, uint16 rounding, and endpoint overrides;
- singleton, empty, substitution-only, insertion-only, removal-only, and
  highly asymmetric jets;
- input permutation, batch, shard, worker, and resume invariance;
- grid independence: 5-, 10-, and 20-point grids sample identical stored
  coordinates;
- strict nestedness and U000/U100 endpoint equality;
- no label, validation metric, or final-test dependency;
- realized balance reports reconcile exact edit/mass/pT/energy totals.

### Uniform D

- every continuous matched slot uses exactly alpha, independent of confidence;
- hand-calculated wrapped-angle and p4 interpolation;
- exact rational hash comparisons for PID/charge, quality, lost-inner-hits,
  and every validity group;
- one-hot identity, charge, track-applicability, and validity coherence;
- switches occur once, are nested, and are invariant to compact carrier order;
- arbitrary rational rung grids retain exact endpoints;
- D100 equality to U100 and D0/J100 equality to canonical HLT, including raw
  lengths over 200;
- old `HIGHCOV_SHELL_EXACT/v1` bytes remain unchanged.

### Graph, training, and access

- exact default 151-fit meta-registry, six child registries, teachers, domains,
  losses, temperatures,
  seed aliases, and dependencies;
- exact partition of that registry into two foundation fits, six disjoint arm
  registries, and the eleven reference-arm legacy fits, with no duplicate or
  unowned `(arm_id, node_id)`;
- six separately validated arm specs, command plans, ledgers, roots, monitors,
  aggregates, completion artifacts, cancellation sets, and recovery closures;
- read-only sweep indexing of exactly six predeclared arm-spec hashes, with no
  directory discovery or combined mutable ledger;
- all six loss tables sum exactly to one; first-edge grandparent reallocation,
  later parent/grandparent routing, pure-KD zero CE, and fixed M1 routing;
- no cross-arm teacher or artifact reuse beyond declared shared immutable
  parents;
- all arms share U000 and paired stochastic seeds and can execute concurrently;
- same-input pairing for both legacy ablations;
- U000 is CE-trained and is every primary first teacher;
- no native TOFF primary target cache exists;
- 60 validations, final partial batch, checkpoint order, USR1, and exact resume;
- one fixed number of ragged conversions per endpoint representation per
  source chunk; prepared-versus-reference byte equality; one-worker versus
  allocated-worker identity/order equality; rejection of worker counts above
  the Slurm CPU allocation;
- one train-view build, one validation-view build, and one target-build phase
  per job regardless of pass count, with mandatory phase timing;
- full train/validation identity coverage and zero ordinary final-test reads;
- finalist and execution locks precede any test access;
- poor finite science completes; corrupt lineage and nonfinite values fail.

Before a live campaign: run focused tests, the complete repository suite,
Python compilation, all new CLI help checks, contract-version checks, a bounded
synthetic end-to-end, Markdown-link validation, and `git diff --check`.

## 14. Explicit no-new-smoke decision

At the user's direction, HCWDL-UB and its six recipe arms will **not** run
another standalone Slurm smoke campaign before the first 300k
execution. The completed HCWDL-UJ
v1 production-worker smoke and subsequent v2 infrastructure/runtime evidence
may be carried forward only through an honest operational-evidence waiver that
binds:

- the completed old smoke and campaign-completion hashes;
- the exact new source commit and semantic diff;
- unchanged model factory, training engine, cache, resume, Slurm worker, and
  reporting primitives by hash;
- successful new local/synthetic U and D tests;
- installed-Weaver parity when its referenced model code is unchanged, or a
  fresh parity artifact otherwise;
- the exact human decision that no separate HCWDL-UB smoke was run;
- the requested resource envelope and its evidence/assumptions.

The waiver must never say that HCWDL-UB passed a smoke. It carries operational
evidence forward and records the residual risk. It also binds the corrected
prepared-endpoint execution baseline and the focused conversion-count,
prepared/reference equality, and ordered-worker regressions required by
Section 9.7.

One waiver binds the exact shared foundation and the six arm-graph templates;
it is not copied into six independently worded exceptions. The shared
foundation begins with fail-closed non-training gates: parent
authentication, coupling/switch construction, full-role conservation,
bounded real-data view construction, endpoint equality, peak-resource
estimation, graph/recipe locking, and a complete command-plan check. Training
jobs cannot start until those gates pass. This is the real campaign prefix,
not a second campaign or a collection of miniature model fits.

One clean pushed source, a complete nonmutating dry run, source-pinned and
resource-only recovery paths, and explicit live submission authorization are
still required. A measured timeout/OOM uses the versioned resource-recovery
path; it does not mutate scientific semantics or trigger an improvised smoke.

## 15. Reporting and comparisons

Validation macro OVR AUC is primary. CE, accuracy, balanced accuracy, per-class
AUC, mean log QCD rejection, R50, Brier score, ECE, confusion matrix, selected
pass, full history, and GPU-hours are mandatory companions.

The ordered questions are:

1. Does CE-trained U000 equal or exceed the authenticated native TOFF context?
2. Does balanced U reach a stronger exact-D100 U100 than the paired legacy-CDF
   U path and direct U000-to-D100 control?
3. Starting from the same U100, does uniform D retain more AUC than the paired
   confidence-warped D path at each identical nominal rung and exact HLT?
4. Does the balanced/uniform joint path outperform or underperform the
   factorized path at exact HLT?
5. Which HLT-only endpoint or M1 retains the largest validation fraction of
   the `M0paired -> U000` gap?
6. Does increasing parent KD or adding a grandparent skip reduce cumulative
   teacher-information loss, and does that make a later denser ladder more
   competitive?

Every KD rung reports its separate CE, parent-KD, and grandparent-KD losses,
the two teacher/student KL divergences, agreement with each teacher, and the
idealized ancestry coefficient. A zero-weight term remains serialized as
zero rather than disappearing from the report schema.

The aggregate reports both `M0paired -> U000` recovery and contextual legacy
`M0 -> TOFF` recovery where exact denominators exist. Recovery is never clipped.
Comparisons across different data populations are labeled contextual. Only
same-population, same-input, seed-paired rows support direct treatment claims.

## 16. Implementation blocks and definition of done

1. Add versioned balanced-switch and uniform-repair contracts plus exact unit
   fixtures; prove old contract bytes unchanged.
2. Implement balanced sidecars and audits over reusable base coupling.
3. Implement `V_UB`, endpoint equality, and arbitrary-grid tests.
4. Register the shared foundation plus six independent arm graphs and the
   read-only sweep index; integrate dual-teacher targets, prepared one-time
   views/logits, training, reporting, arm isolation, concurrent submission,
   and exact arm-local resume.
5. Add exact 300k/100k/100k campaign creation, sealed-test evaluation,
   recovery, Slurm commands, and the explicit operational-evidence waiver.
6. Complete focused/full tests, synthetic end-to-end, CLI/static/link checks,
   full dry run, and documentation evidence.

Implementation is complete only when the entire default graph is executable
from a clean source-pinned worktree, every artifact validates by content and
parent hash, no final-test access is possible before both locks, old HCWDL-UJ
behavior remains reproducible, recovery exists before launch, and the exact
300k/100k/100k identities and counts are locked in the foundation and every
arm campaign spec.

This document authorizes implementation when requested. It does not authorize
job submission, cancellation, remote push, mutation of the running campaign,
or final-test access.

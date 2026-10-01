# HCWDL TRI100 Four-Spine Full-Cardinality Salience Matching Implementation Plan

Status: implemented locally; live candidate foundations, validation selection,
and winner-only production remain gated on remote acceptance and explicit
submission authorization.

This document freezes the scientific and execution semantics for a new,
isolated full-cardinality HLT/offline pairing study. It does not reinterpret,
replace, or mutate the existing high-coverage, bottleneck, non-persistent, or
persistent-support campaigns. The existence of this plan does not authorize a
live production submission.

## 1. Scientific question

The existing full-cardinality bottleneck matcher gives every particle on the
smaller side a one-to-one partner and lexicographically minimizes the worst
selected delta-R, then the second worst, and so on. That objective treats the
worst selected edge as more important than every improvement to easier edges.
It can therefore disturb several close, high-impact matches to improve one
soft peripheral particle whose effect on classification may be negligible.

This study asks:

> Does a full-cardinality assignment that preferentially preserves close
> matches for high-pT, jet-core particles produce a stronger and smoother
> classifier-relevant HLT/offline coordinate than worst-edge optimization?

The hypothesis is not that salience weighting discovers physical truth. The
hypothesis is that, for an HLT-only classifier trained through offline-rich
intermediate views, preserving the most classifier-relevant constituent
structure may be more useful than equalizing the assignment tail.

## 2. Controlled comparison and authority

The primary causal comparison is:

```text
persistent-support full-cardinality bottleneck campaign
    lexicographic worst-delta-R assignment

versus

persistent-support full-cardinality salience campaign
    bounded salience-weighted angular-closeness assignment
```

The new campaign keeps fixed, unless this document says otherwise:

- the authenticated train and validation populations;
- the full smaller-side cardinality constraint;
- the persistent-HLT-support U/D view semantics;
- raw endpoint identity handling and pairing-validity provenance;
- the four branch paths;
- immediate-parent-only C25/P75 logit KD at temperature 2;
- model architecture, batch size, seed aliases, optimizer, validation,
  checkpoint selection, and single-GH200 execution;
- the 100-pass floor-tail schedule and early stopping;
- storage, recovery, and final-test restrictions.

The changed scientific variable is the assignment objective, including its
preregistered three-candidate salience family and validation-only selection.
The selection multiplicity must be visible in every downstream report. The
selected matcher must never be presented as though it had been specified as a
single candidate before validation.

When documents conflict for this study, this plan has authority over older
bottleneck and persistent-support plans only for artifacts carrying the new
salience contract family. It has no authority to redefine old artifacts.

## 3. Claim boundary

Permitted names are:

- full-cardinality salience pairing;
- salience-weighted angular pairing;
- classifier-utility-oriented forced alignment control.

The result must not be called:

- truth matching;
- a calibrated physical-correspondence matcher;
- high-purity matching;
- a detector reconstruction association probability.

Every selected pair remains a forced alignment edge. Large-distance and
identity-disagreeing pairs are allowed because exact smaller-side coverage is
the registered control. Pairing indices, salience, delta-R, ranks, source
indices, and offline features never become deployable model inputs. D000
remains byte-identical canonical HLT.

## 4. Population and exact cardinality

For one paired jet, let:

```text
n_h = number of valid visible HLT particles
n_o = number of valid visible offline particles
k   = min(n_h, n_o)
```

The assignment contains exactly `k` one-to-one real pairs:

- if `n_h <= n_o`, every HLT particle is paired and `n_o - n_h` offline
  particles are unused;
- if `n_h > n_o`, every offline particle is paired and `n_h - n_o` HLT
  particles remain unmatched;
- if either side is empty, the relation is empty;
- each real HLT and offline index appears at most once;
- padded or nonfinite particles are ineligible;
- there is no delta-R, response, charge, PID, rank, or confidence gate.

The feasible graph is complete bipartite over valid particles. The durable
orientation remains one native offline index or `-1` per valid HLT particle.
Exactly 100% of the smaller side is covered for every nonempty jet.

Validity has the existing full-cardinality meaning. Finite-binary zero-hot or
multi-hot particle identity does not remove a visible particle. Category and
charge may break exact objective ties but never control eligibility.

## 5. Canonical kinematics

All kinematics are computed in float64 from the authenticated endpoint
four-vectors already used by the bottleneck matcher:

```text
pt = sqrt(px^2 + py^2)

dphi(a,b) = wrap_to_half_open_minus_pi_plus_pi(phi_a - phi_b)
dr(a,b)   = sqrt((eta_a - eta_b)^2 + dphi(a,b)^2)
```

Pairwise HLT/offline delta-R uses the two particle coordinates. Particle
centrality uses the angular distance from the particle to the axis formed by
the sum of the valid visible four-vectors in its own endpoint collection.
Thus “central” means central within the reconstructed jet, not central in
detector pseudorapidity. A nonempty endpoint with nonfinite kinematics or a
nonpositive finite scalar-pT sum fails closed.

The canonical quantization is:

```text
DR_QUANTUM       = 1e-7
PT_SHARE_SCALE   = 1_000_000
SALIENCE_FLOOR   = 10_000
PAIR_DR_CAP      = 0.8
ROUNDING_MODE    = IEEE-754 roundTiesToEven
```

For particle `i` in one endpoint:

```text
x_i       = pt_i / sum_j(pt_j)
qshare_i  = round_half_to_even(PT_SHARE_SCALE * x_i)
qaxis_i   = round_half_to_even(dr_to_own_jet_axis_i / DR_QUANTUM)
QCAP      = round_half_to_even(PAIR_DR_CAP / DR_QUANTUM)
qcore_i   = max(QCAP - min(qaxis_i, QCAP), 0)
```

Native particle order is the reduction order for the endpoint pT sum. The
dtype, ordering, wrap convention, quantization, and constants are scientific
semantics and require a new contract version if changed.

## 6. Preregistered salience candidates

Exactly three salience candidates are eligible. All weights are positive
integers, so no constituent is ignored.

### 6.1 `SALIENCE_PT_LINEAR`

```text
w_i = SALIENCE_FLOOR + qshare_i
```

This is the least-assumptive candidate. Importance grows linearly with the
particle’s fraction of endpoint scalar pT and has no explicit core bonus.

### 6.2 `SALIENCE_PT_QUADRATIC`

```text
qpower_i = round_half_to_even(qshare_i^2 / PT_SHARE_SCALE)
w_i      = SALIENCE_FLOOR + qpower_i
```

This gives leading particles more influence while retaining the same floor.

### 6.3 `SALIENCE_PT_QUADRATIC_CORE25`

```text
base_i  = SALIENCE_FLOOR + qpower_i
bonus_i = round_half_to_even(base_i * qcore_i / (4 * QCAP))
w_i     = base_i + bonus_i
```

The core bonus is continuous, bounded, and at most 25% of the base weight.
It is deliberately mild because low-pT or peripheral displaced charged tracks
can carry important heavy-flavor information. No impact parameter, jet label,
classifier gradient, attention value, or learned importance enters salience.

The candidate registry order is exactly:

```text
SALIENCE_PT_LINEAR
SALIENCE_PT_QUADRATIC
SALIENCE_PT_QUADRATIC_CORE25
```

No additional exponent, floor, cap, centrality coefficient, or learned
salience candidate may be added after inspecting screen metrics.

## 7. Exact assignment objective

For candidate edge `(i,j)`, define:

```text
qdr_ij       = round_half_to_even(dr(i,j) / DR_QUANTUM)
qclose_ij    = max(QCAP - min(qdr_ij, QCAP), 0)
pair_weight  = w_hlt_i + w_offline_j
utility_ij   = pair_weight * qclose_ij
```

For every cardinality-`k` assignment `A`, the primary objective is:

```text
maximize  sum_(i,j in A) utility_ij
```

This bounded angular-closeness utility has the intended behavior:

- improving a close high-salience pair is worth more than improving a close
  low-salience pair by the same amount;
- a very poor soft-particle edge cannot dominate the assignment merely by
  being the worst edge;
- selecting an important particle from the larger side is rewarded when it
  has a geometrically useful partner;
- all required smaller-side particles remain matched regardless of utility;
- distances at or beyond 0.8 contribute zero primary utility but remain
  visible in diagnostics and lower-order tie breaking.

The objective is a global assignment, not a greedy walk through particles in
pT order. A high-salience decision may reserve a partner for another
high-salience particle when that improves total registered utility.

### 7.1 Exact secondary hierarchy

Only assignments with identical total primary utility proceed to:

1. minimize total uncapped canonical `qdr`;
2. maximize total selected salience from the larger endpoint;
3. minimize total canonical absolute log-pT response;
4. minimize raw particle-category mismatch count;
5. minimize valid-charge mismatch count;
6. lexicographically minimize native offline indices in HLT native order,
   with `-1` after every real index.

No secondary term may trade against one integer unit of primary utility. In
particular, the old sorted worst-edge vector is diagnostic only and is not a
tie breaker.

## 8. Exact solver and reference implementation

The implementation must provide:

- an exhaustive reference enumerator for jets with both sides at most eight;
- an arbitrary-precision integer production assignment solver;
- exact agreement of mapping and complete objective signature between the two
  on all bounded fixtures;
- rectangular handling in both count-imbalance directions;
- deterministic native-index tie resolution.

The primary and secondary hierarchy may use mixed-radix integer encoding only
after every lower-priority bound is proved. Floating weighted sums are not
allowed. The existing integer Hungarian implementation may be factored into a
neutral reusable helper, but its bottleneck objective and contracts must not
be repurposed.

Dense cost matrices exist only per jet in process memory. They are never
persisted.

## 9. Matching-only diagnostics

Each candidate and the existing bottleneck control must report, separately
for train and validation:

- exact jets, endpoint particle counts, selected pairs, and smaller-side
  coverage;
- unmatched HLT and unused offline counts;
- unmatched scalar-pT fraction and unmatched leading-particle counts on the
  larger side;
- ordinary and pT-weighted selected delta-R mean and quantiles;
- delta-R for leading-pT ranks 1, 2, 3, 5, 10, and greater than 10;
- fraction of scalar pT paired within delta-R 0.02, 0.05, 0.10, 0.20, 0.30,
  0.50, 0.80, and 1.00;
- per-jet worst-edge distributions, explicitly labeled diagnostic rather than
  objective;
- top-1, top-3, and top-5 maximum and mean delta-R;
- charged, neutral, classified, zero-hot, multi-hot, and lost-track slices;
- displaced-track slices using preregistered absolute and significance bins;
- pT-response, category-transition, and charge-transition summaries;
- pair overlap and common-pair delta-R changes relative to bottleneck;
- exact recomputation on a bounded authenticated sample.

The expected trade is better high-salience and pT-weighted geometry with a
possibly worse unweighted maximum. A worse maximum is not a failure.

## 10. Validation-only endpoint screen

Matching diagnostics alone do not determine classifier utility. Before the
full ladder, run one matched CE-only endpoint fit for:

```text
BOTTLENECK_REFERENCE
SALIENCE_PT_LINEAR
SALIENCE_PT_QUADRATIC
SALIENCE_PT_QUADRATIC_CORE25
```

The endpoint is the persistent-support `U100` view: exact HLT cardinality,
matched slots carrying offline features, and unavoidable unmatched HLT slots
carrying native HLT. Historical `T100` or `D100` shorthand must map explicitly
to this registered view name in reports.

All four screen fits use:

- the complete authenticated train role;
- identical architecture and initialization seed;
- identical sampler, batch size 256, optimizer, and one-GH200 execution;
- CE only, with no KD teacher or checkpoint continuation;
- the registered 100-pass floor-tail schedule and early stopping;
- no rolling resume or optimizer-state publication.

### 10.1 Validation partition

The authenticated validation role is deterministically partitioned by class
and canonical row identity into:

```text
V_checkpoint = 75%
V_select     = 25%
```

The split algorithm, salt, class counts, row hashes, and manifest are frozen
before training. `V_checkpoint` performs per-pass checkpoint selection.
`V_select` is evaluated exactly once from each restored selected checkpoint
and is never visible to training, early stopping, or checkpoint selection.

The later production ladder returns to the established complete-validation
checkpoint policy after the matcher is frozen. The screen partition is only
the independent candidate-selection mechanism.

### 10.2 Winner selection

All three salience fits always run. The bottleneck reference is not eligible
to become the salience winner; it is a contextual control. Among the three
salience candidates:

1. find the maximum `V_select` macro one-vs-rest AUC;
2. retain candidates within `5e-5` absolute AUC of that maximum;
3. choose maximum `V_select` macro mean log rejection at 50% signal
   efficiency;
4. then maximum `V_select` accuracy;
5. then minimum pT-weighted validation delta-R;
6. then earliest candidate in the frozen registry.

The lock records all metrics, the complete rule application, and paired
bootstrap intervals, but confidence intervals do not alter this deterministic
selection. Weak performance cannot suppress the winner or cancel the later
registered campaign. Invalid lineage or nonfinite required metrics fails
closed.

## 11. Artifact and contract family

The new family requires distinct versioned contracts, at minimum:

```text
HCWDL_FULLCARD_SALIENCE_MATCHER_REGISTRY/v1
HCWDL_FULLCARD_SALIENCE_MATCHER_SPEC/v1
HCWDL_FULLCARD_SALIENCE_ASSIGNMENT_SHARD/v1
HCWDL_FULLCARD_SALIENCE_ASSIGNMENT_MANIFEST/v1
HCWDL_FULLCARD_SALIENCE_ASSIGNMENT_AUDIT/v1
HCWDL_FULLCARD_SALIENCE_DIAGNOSTIC_REPORT/v1
HCWDL_FULLCARD_SALIENCE_SCREEN_SPLIT/v1
HCWDL_FULLCARD_SALIENCE_SCREEN_REPORT/v1
HCWDL_FULLCARD_SALIENCE_SELECTION_LOCK/v1
HCWDL_FULLCARD_SALIENCE_FOUNDATION_SPEC/v1
HCWDL_FULLCARD_SALIENCE_FOUNDATION_LOCK/v1
HCWDL_TRI100_FOUR_SPINE_SALIENCE_PERSISTENT_HLT_*/v1
```

Every reusable artifact has a content hash, explicit parent hashes, source
commit, semantic source hashes, role and row coverage, final-test state, and
validation before reuse. Paths and timestamps are never scientific identity.

Assignment payloads contain compact native-index mappings and pairing-validity
metadata only. Salience is reproducible from authenticated inputs and need not
be repeated per epoch. It must not be mislabeled correspondence confidence.

## 12. Foundation rebuild boundary

The study may reuse read-only:

- raw data inventory;
- split and all-mapped-row selection manifests;
- endpoint schemas and labels;
- model architecture and training recipes where unchanged;
- established M0CE60 and pure-offline U000 reports as contextual references.

For each screen candidate it must rebuild or independently bind:

- assignment shards, manifests, audits, and candidate lock;
- assignment-dependent scale/coupling and balanced support metadata;
- endpoint screen views and reports.

After selection, the production lineage builds a winner-only foundation lock
and rejects every loser or bottleneck assignment as a view parent. No
assignment-dependent probability bank, coupling sidecar, or training report
from another matcher may be substituted.

The persistent-support anchor must be retrained because its particle values
depend on the selected mapping. Pure-offline U000 remains only the shared 100%
recovery oracle.

## 13. Persistent-support coordinate semantics

The selected salience assignment is consumed by the existing persistent
support policy:

1. every HLT skeleton slot exists from `SP4S_U000` onward;
2. matched skeleton slots initially carry matched offline endpoint features;
3. unmatched HLT skeleton slots always carry native HLT features;
4. offline-only source tails exist at U000 and are monotonically removed;
5. matched offline particles appear exactly once on their HLT slots;
6. U100 has exact HLT cardinality with matched slots still offline-rich;
7. D changes matched fields toward HLT on fixed support;
8. D000 is byte-identical canonical HLT.

The raw endpoint identity policy remains
`native_collection_applicability_atomic_raw_endpoint_v1`. Native collection
membership determines applicability without inventing a particle class;
finite-binary zero-hot and multi-hot identities remain raw. Nonfinite,
nonbinary, incompatible, or nondiscrete required values fail closed.

## 14. Full four-spine graph

The selected matcher launches a fresh anchor `SP4S_U000`, then the same four
immediate-parent paths:

```text
DIRECT
SP4S_U000 -> D000

COARSE
SP4S_U000 -> U050 -> U100 -> D066 -> D033 -> D000

DENSE
SP4S_U000 -> U033 -> U066 -> U100
           -> D080 -> D060 -> D040 -> D020 -> D000

ULTRADENSE
SP4S_U000 -> U020 -> U040 -> U060 -> U080 -> U100
           -> D090 -> D080 -> D070 -> D060 -> D050
           -> D040 -> D030 -> D020 -> D010 -> D000
```

The graph has 30 fresh fits including the anchor and 26 single-component
probability reducers. There are no ensembles, M1 fits, weight continuation,
branch pruning, or final-test tasks.

Every non-anchor fit uses only the immediately preceding selected checkpoint’s
complete T=2 probability bank. The four first rungs share only the new anchor
bank and may run concurrently.

## 15. Frozen training and reporting

The anchor is CE only and uses its matched persistent-support view. Every
downstream fit uses constant 25% CE plus 75% logit KD at T=2. All fits use:

- batch size 256 on one GH200 process;
- the existing coordinate-specific initialization, sampler, repair, view, and
  model seed aliases;
- passes 1-3 warmup to `3e-4`;
- passes 4-45 held at `3e-4`;
- passes 46-60 cosine decay to `1.5e-5`;
- passes 61-100 constant `1.5e-5` refinement;
- minimum 60 passes, maximum 100, patience 15, and meaningful AUC delta
  `5e-5`;
- exact best-checkpoint restoration under the established tie hierarchy;
- no rolling resume or partial checkpoint continuation.

Reporting uses `M0CE60 = 0%` and pure-offline U000 `= 100%` for recovery.
`SP4S_U000` is an experimental hybrid anchor, not the offline oracle. Reports
must show the complete candidate screen, selection multiplicity, bottleneck
comparison, four salience branches, and all Hbb/Hcc/Hqq class-rejection
metrics already registered by the surrounding campaign family.

## 16. Storage and memory

- Complete cost matrices live only per jet in RAM.
- No dense edge, salience, particle-view, ROOT, NPZ, or Parquet cache is
  durable.
- Compact assignment maps are sharded and atomically published.
- Histograms and bounded deterministic samples replace full pair diagnostics.
- Training views are materialized in process RAM and discarded after each
  job.
- Probability banks contain identity digests and float32 class probabilities,
  not particle features.
- No optimizer archive or rolling-resume state is written.
- Every job publishes an output inventory.
- Candidate and production roots have measured conservative storage bounds
  and a minimum 20-GiB free-space reserve.

Candidate artifacts cannot be deleted merely because their metrics lose. Any
later cleanup is a separately authorized operational action after immutable
selection and diagnostic reports are safely retained.

## 17. Failure policy

Fail closed for:

- stale or dirty source;
- missing or mismatched parent/content hashes;
- incomplete row/source coverage;
- nonfinite kinematics, salience, metrics, or model outputs;
- invalid cardinality, duplicate indices, or out-of-bounds indices;
- production/reference assignment disagreement;
- candidate registry or selection-rule drift;
- cross-matcher foundation, view, target, or report reuse;
- nondeterministic recomputation;
- forbidden final-test access or offline deployable input;
- hidden token truncation or nonexact D000;
- corrupt checkpoints or probability banks.

Do not fail, skip, or cancel registered scientific work because:

- bottleneck has a better worst edge;
- salience has visually bad low-pT pairs;
- any candidate has poor AUC, accuracy, R50, calibration, or recovery;
- bottleneck beats all three salience candidates;
- a coarse branch beats a dense branch.

## 18. Slurm isolation and recovery

The screen, winner foundation, and full campaign use new roots, job prefixes,
specifications, command plans, journals, ledgers, attestations, and monitors.
They have no Slurm dependency on and no write path into running bottleneck,
persistent-support, TRI100, TRI60, DX, RSET/RREL, fusion, or other campaigns.

Recovery is source-pinned and restart-from-zero. It is created only after all
subject jobs are terminal, inherits only fully authenticated completed tasks,
and retries everything else under a new recovery root. Cancellation uses only
exact IDs read from the bound ledger.

Matcher, screen, training, and reducer resources require genuine Tigris
measurement. Resource changes may improve execution but cannot alter numeric
assignment or training semantics.

## 19. Proposed implementation map

Reusable code belongs under `src/hlt_classification/scouting/`; CLIs and Slurm
workers remain thin. Expected new surfaces are:

```text
src/hlt_classification/scouting/hcwdl_fullcard_salience_contracts.py
src/hlt_classification/scouting/hcwdl_fullcard_salience_matcher.py
src/hlt_classification/scouting/hcwdl_fullcard_salience_diagnostics.py
src/hlt_classification/scouting/hcwdl_fullcard_salience_foundation.py
src/hlt_classification/scouting/hcwdl_fullcard_salience_screen.py
src/hlt_classification/scouting/hcwdl_tri100_spine4_salience_campaign.py
src/hlt_classification/scouting/hcwdl_tri100_spine4_salience_runner.py
src/hlt_classification/scouting/hcwdl_tri100_spine4_salience_recovery.py
```

Neutral integer-Hungarian, compact-cache, source-authentication, persistent-
support, training, probability, reporting, and exact-DAG utilities should be
reused where their contracts genuinely match. Existing bottleneck modules
must not be renamed or silently generalized in place.

No external donor is presently required. If implementation copies code from
another worktree or historical commit, record its exact path and commit in
`docs/LEGACY_SOURCE_MAP.md`.

## 20. Required tests

### 20.1 Salience and solver

- exact pT-share, jet-axis, core, floor, and round-ties-to-even fixtures;
- invariance to batch, worker, and source-shard partitioning;
- all three exact candidate weight vectors;
- empty, square, and both rectangular count directions;
- exhaustive reference equality with duplicate costs and native-index ties;
- counterexample where bottleneck and salience assignments differ;
- counterexample proving a low-salience worst edge cannot dominate several
  high-salience close edges;
- larger-side high-salience selection when geometrically useful;
- zero primary utility beyond the cap and deterministic secondary resolution;
- wrapped-phi and nonfinite failure cases.

### 20.2 Artifacts, screen, and selection

- tamper rejection for every parent and content hash;
- complete source/role/row coverage;
- no dense durable matrices or fake confidence;
- exact deterministic 75/25 stratified validation partition;
- V_select inaccessible to checkpoint selection;
- four matched screen fits and three eligible candidates;
- AUC band, logR50, accuracy, geometry, and registry tie resolution;
- poor candidate metrics still produce a winner and complete report;
- winner-only production foundation rejects loser assignments.

### 20.3 Persistent views and campaign

- persistent unmatched HLT slots and monotone U support;
- matched offline features on HLT slots at U000 and U100;
- raw zero-hot/multi-hot endpoint preservation;
- exact HLT cardinality at U100 and byte-exact HLT at D000;
- exact four branch paths, 30 fits, and 26 reducers;
- C25/P75, T=2, batch 256, one-GH200, and floor-tail semantics;
- no ensemble, M1, DDP, final-test, rolling-resume, or existing-job mutation;
- complete command-plan, monitor, attestation, and recovery closure.

Focused tests precede the complete ladder in `docs/TESTING.md`.

## 21. Production-readiness sequence

1. Implement and locally test contracts, salience, reference, and production
   solvers.
2. Run installed-Weaver parity for touched view/training surfaces.
3. Execute a genuine Tigris matcher miniature containing both count-imbalance
   directions, low-multiplicity exhaustive rows, high multiplicity, lost
   tracks, and zero-hot/multi-hot identities.
4. Measure CPU, RAM, walltime, and bytes for all three candidate assignments.
5. Build complete train/validation candidate assignments and diagnostics.
6. Materialize and authenticate the validation partition and four-fit screen
   dry run.
7. Run all four screen fits and publish the immutable selection lock.
8. Build the winner-only foundation and all-row persistent-support audit.
9. Run a genuine single-GH200 production forward/backward preflight from the
   selected source.
10. Materialize and audit the complete four-spine dry-run ledger.
11. Require the exact pushed commit, clean detached worktree, measured storage
    headroom, and separate explicit live-submission authorization.

No final-test assignment, view, probability, or inference artifact is created
by this sequence.

## 22. Implementation phases

### Phase A: contracts and exact mathematics

Implement the candidate registry, canonical salience, pair utility, exhaustive
reference, integer production solver, and focused counterexamples.

### Phase B: candidate assignment and diagnostics

Implement compact artifacts, full-data builders, matching-only comparison,
resource accounting, and exact candidate lineage.

### Phase C: endpoint screen

Implement the frozen validation partition, four matched CE-only endpoint
fits, evaluation, paired uncertainty, and deterministic selection lock.

### Phase D: winner foundation and persistent views

Build the winner-only assignment-dependent descendants, source lock, support
audit, and real-data preflight. Reuse persistent view semantics without
changing their endpoint meaning.

### Phase E: four-spine campaign

Implement the isolated 30-fit/26-reducer DAG, aggregation, monitoring,
restart-zero recovery, storage audit, and queue handoff.

## 23. Completion criteria

Implementation is queue-ready only when:

- all equations and constants above are represented in versioned artifacts;
- production assignments exactly match the exhaustive reference;
- every nonempty jet has exact smaller-side coverage;
- all three candidate diagnostics and screen rows are present;
- the immutable winner follows the preregistered rule;
- the winner-only foundation authenticates complete train/validation coverage;
- persistent-support and exact-D000 invariants pass;
- the campaign has exactly 30 fits and 26 reducers;
- storage remains within measured bounds without durable particle views;
- focused, adjacent, and repository-wide tests pass;
- installed-Weaver and genuine Tigris acceptance pass;
- the exact source commit is pushed and the worktree is clean;
- no existing campaign is changed or operationally disturbed;
- final test remains sealed;
- `docs/HANDOFF.md` records evidence, not anticipated success.

## 24. Non-goals

This study does not:

- claim forced pairs are physical correspondences;
- learn salience, matching scores, or correspondence confidence;
- tune against final test;
- add a matching gate or permit incomplete smaller-side coverage;
- alter the old bottleneck assignment or its artifacts;
- change representation KD, ensembles, compression, or M1;
- compare multi-GPU training or batch sizes;
- change the four-spine schedule, loss, seeds, or architecture;
- authorize automatic replacement of the repository’s canonical matcher.

A successful result supports salience-weighted forced alignment as a useful
training coordinate. It does not establish detector-level matching truth.

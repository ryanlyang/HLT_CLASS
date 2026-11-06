# Next supervision: Group B Q/V contract v1

Authority: the [implementation plan](../plans/LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md)
and [ideas catalog](../plans/LUKA_NEXT_SUPERVISION_CAMPAIGN_IDEAS.md).
This extends `luka_next_supervision_hydra`, not a separate execution stack.
Internal implementation donor: `342e5d3affba78f9d6add4cbb95d6264fc4b516b`.
Existing A worktrees, specifications, targets and live jobs are unchanged.

## Nine rows and native inference

| Order within B | ID | Target | Coefficient | Seed |
|---:|---|---|---:|---:|
| 1 | Q1 | Four OFFLINE question families,64 questions | .3 | 0 |
| 2 | V1 | OFFLINE K128 count+pT frequencies | .3 | 0 |
| 3 | Q2 | Q1 replication | .3 | 1 |
| 4 | V2 | V1 replication | .3 | 1 |
| 5 | Q5 | Same questions answered from HLT | .3 | 0 |
| 6 | V6 | Separately fitted HLT dictionary/frequencies | .3 | 0 |
| 7 | Q3 | Q1 with gentler loss | .1 | 0 |
| 8 | V7 | OFFLINE count frequencies only | .3 | 0 |
| 9 | Q6 | Composition/count families only,32 questions each | .3 | 0 |

Every row is from-scratch native HLT ParT8x128,17 inputs/11 classes,
batch128/full30 epochs, CE +5*the original frozen standardized U100 SmoothL1.
Original initialization/data-order seeds, seed1 offset1000003, U100 predictor
seed20261019, AdamW/LR and full-validation AUC selection are unchanged. New
heads use the isolated auxiliary CPU RNG and a three-epoch linear loss ramp.
No privileged values enter native attention or the classifier. Exports contain
only `mod.*` and must reproduce selected native probabilities/metrics on reload.
C0/C1 reuse the original authenticated same-seed references; no new controls
or teacher fits are silently added. The selected40-row order remains intact.

## Q physical targets and semantic inputs

Use the already authenticated TRAIN-only raw14-column transfer. No validation
or independent-test particles contribute targets, ranges, dictionaries or
support. The full source collection supplies scalar-pT fractions; angular
offsets use the paired HLT collection's summed-p4 axis. This differs explicitly
from P/G/V's source-parent-local descriptors. No particle correspondence is
required. Geometry stores FP64 fraction/eta/phi/displacement, measured charge/
category and separate validity bits. It keeps every real particle.

Displacement fields are absolute d0/dz in mm and reported absolute d0/dz
significance. Significance requires a valid value, valid error and positive
error. Invalid values are zero placeholders with validity false, not measured
zeros. Displacement questions require charged particles and valid measurements.
Shared category codes0..5 are measured categories;6 means any category.
Charge selectors0/1/2 mean -1/0/+1, and3 means any charge.

Questions have13 semantic columns: family, category, charge, field, radial
lower/upper bounds, scalar-pT fraction lower/upper bounds, displacement
threshold, minimum count, pair-separation lower/upper bounds, pair-charge
relationship. Field0..3 selects d0/dz/significances;4 means no displacement
condition. Pair-charge0/1/2 means any/same/opposite sign. Bounds are lower
inclusive/upper exclusive; displacement exceeds its threshold strictly.

- Composition: selected scalar-pT fraction, including category/charge,
  annulus and momentum conditions.
- Counts: whether at least m selected valid charged measurements exceed the
  chosen displacement threshold.
- Joint: scalar-pT fraction satisfying the joint measured conditions.
- Correlations: existence of two distinct selected particles in the same
  source jet satisfying deltaR and charge-relation conditions. There is no
  ancestry/common-vertex assertion. Wrapped deltaPhi is used.

Catalog construction is fixed before HLT fitting. SHA256-ranked TRAIN IDs
select at most32768 design jets; each contributes at most8 uniformly drawn
real particles. This bookkeeping is not a model input and uses no jet labels.
Radial and fraction grids use design quantiles plus explicit physical bounds;
displacement thresholds use valid charged design quantiles, with zero included.
Count thresholds are1,2,3,5,8,13,21,34. Pair-separation grids use twice the
radial quantiles and a full-support bound. Seeds and formulas are in the
hashed STRUCTURED_RECIPE. Category/charge combinations use measured global TRAIN
support, conditional on valid charged measurements for displacement questions;
this removes globally impossible combinations without pruning rare support or
conditioning a query on an individual anchor. Up to2048 distinct proposals per family are generated;
if the finite semantic space is smaller, enumerate it without duplicate padding.

Stream the **entire TRAIN population** through the bounded oracle to record
FP64 sums/squared sums and exact min/max of FP32 answers. Remove exactly
constant proposals; retain the first up to1024 nonconstant entries per family.
There is no prevalence cutoff: a single rare positive remains supported.
The design reservoir is never used to declare a rare query globally absent.
Question and pair chunks bound temporary arrays; no dataset-wide pair table
or jets-times-questions answer bank is stored.
The reduction uses fixed32-jet batches with at most8 outstanding results;
process count does not change the ordered reduction or sampling stream.

The1024 supported-entry goal is a design budget, not a quality admission gate.
Actual coverage is frozen, printed and included in the science dry review.
If fewer entries survive, retain all and disclose the shortfall. If an entire
family is constant, retain its fixed proposals with zero loss applicability
and report that the dataset supplies no nonconstant supervision for that
family. This explicit closure supersedes the catalog's unresolved suggestion
to revise the draft after observing short support. It never fabricates support,
changes another registered row or fails a fit for finite poor targets.

For each identity/epoch/replicate/family, PCG64 draws32 catalog questions;
four-family arms take16 per family, and Q6 takes32 from each of families0/1.
Use replacement only if that family's frozen catalog has fewer than32 entries.
Sampling is independent of that anchor's answers, locations, source and class.
Q1/Q3/Q5 use identical questions; Q6 shares the corresponding first16 and adds
coverage. Encoded inputs are31 semantic values/one-hot fields, not catalog IDs.
Numerical scaling is frozen in the catalog. Source controls share this OFFLINE
catalog and physical thresholds but compute answers from HLT measurements.

Question encoder31->128->64 and answer decoder192->256->128->1 use GELU and
no BatchNorm. FP32 BCE-with-logits accepts binary and fraction targets. Average
questions within each selected family, then families and jets; zero-applicable
families contribute zero without increasing the weight on other families.
This learns mean fractions/event probabilities, not a full uncertainty model.

## V dictionaries and target frequencies

Use the exact22 source-local P descriptors and TRAIN FP64 mean/std-floor1e-3
normalization. Invalid continuous placeholders remain fixed values accompanied
by validity bits. Feature blocks are kinematics0..4, composition5..11, and
tracking/validity12..21. Divide each block by the square root of three times
its full-TRAIN mean squared norm, floor1e-12, to give equal block energy.

The identity-ranked32768-jet/up-to8-particle reservoir uses a separate stream;
jet-class labels are unused. Fit sklearn MiniBatchKMeans, K128, k-means++,
n_init3, random_state2026110513, max_iter100, batch4096,
max_no_improvement20, reassignment_ratio0, compute_labels=False, with native
CPU libraries limited to one thread. Freeze centers/transforms before fitting.
Assignment uses direct FP64 squared distances in1024-particle chunks, taking
the first center on exact ties. Empty/duplicate centers are diagnostics.

Build two full-TRAIN K128 targets: counts normalized by real-particle count,
and raw physical scalar-pT sums normalized by total scalar-pT. No implicit
multiplicity target is added. Zero frequency bins are valid; invalid source
denominators/nonfinite required data fail. Banks bind ordered identities,
labels as metadata, source, dictionary and normalization hashes. HLT-source V6
fits its own HLT dictionary; its codes are not claimed to match OFFLINE codes.

Each target has a removable128->128->128 GELU head. V1 averages count and pT
soft cross-entropies; V7 uses count only. Sum over codes, then average heads
and jets, in FP32. Do not divide again by128. Frozen centers/banks are never
learnable student parameters or deployable inference memory.

## Artifacts, execution and interpretation

New envelopes are STRUCTURED_RECIPE/SPEC/TARGETS, QUESTION_GEOMETRY/CATALOG,
DICTIONARY and FREQUENCY_BANK under `LUKA_NEXT_SUPERVISION_*/v1`. The common
ACCEPTANCE/RESOURCE_CHECK remain execution v2; all bind exact source/recipe/
population/raw parents, content hashes and file checksums with atomic publication.
The current A recipe/artifacts remain their original versions and bytes.

Registration authenticates the Group A input spec in its own pinned worktree,
then creates a separate B output root/source snapshot. It does not wait for A's
scientific results or use them as targets. Preserve old source, jobs and claims.
Setup prepares both sources, dictionaries and compact banks once, verifies
installed-Weaver native/w5 parity and measures six distinct warmed full/partial
paths: Q-full OFFLINE, Q-HLT, Q-marginal, V-both OFFLINE, V-HLT, V-count-only.
Weight/seed variants reuse only support-identical execution measurements.
Q miniatures include the largest source-particle and valid-charged-track
tails as well as native HLT extremes, with unique rows and non-singleton partial
batches. This avoids measuring only native-large jets while missing costly
OFFLINE pair-query support.

Use one RTX A6000,6CPUs,32000MiB, four CPU-only fork oracle workers where needed.
Children never call CUDA. Full-TRAIN support reduction has bounded outstanding
results and records conservative parent-peak-plus-child-PSS preparation memory.
The one-time preparation elapsed time is reviewed separately, not charged to
every science fit. Per-fit measurement includes target loading/IPC/oracle,
all model/backward paths and first-TRAIN-batch diagnostics. Preserve the48h
fit ceiling,85% memory safety fraction, all anchors and full30 budget.

Setup is an explicit technical submission. After exact COMPLETED/0:0 and
authenticated acceptance, review all9 independent B fits in filtered global
order. Scope-specific authorization prevents an A token admitting a B plan.
There are no fit dependencies, holds, arrays/lanes/concurrency caps or nice
adjustments. No automatic retry, new reference fit or dependent report is added.

Saved probes include per-family/head losses, prevalence/prediction variation,
auxiliary jet-gradient norm, TRAIN prior and zero/within-class-shuffled-query
comparisons. These adaptive TRAIN probes are not independent target validation.
Q's fixed prior is the OFFLINE catalog mean, including in Q5; it is labeled
as such rather than claimed to be the optimal HLT-answer prior. V uses its
source-specific frequency mean. Neither prior enters the optimization target
or query sampling.
Partial results/progress reuse the native saved-metric views and report all
pending/failed/incomplete rows and matched C0/C1 deltas. Historical recovery
anchors are exploratory; original50k overlaps validation and no independent
test is introduced. Finite poor target quality/metrics never reject admission,
cancel a row or shorten its budget. Invalid lineage, data, nonfinite required
values, forbidden capabilities and resource violations fail closed.

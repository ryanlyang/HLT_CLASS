# JetClass2 dz-fix fixed-slot concatenation ladder: K=2, HLT x3 -> HLT x1

Status: staged implementation added on 2026-09-21 after the user selected K=2
and requested implementation with every job in debug. See the versioned
[campaign contract](../contracts/JETCLASS2_DZFIX_CONCAT_K2.md) for frozen
choices, tests and queue interface. Queue tooling must run fresh expanded-input
SPORC acceptance before releasing science; no real GPU acceptance is claimed.
Implementing this plan does not itself submit jobs.
This is a separate study and must not change any existing matching foundation,
fusion campaign, checkpoint, job, or immutable specification.

## 1. Scientific question

Can a single Particle Transformer trained on a concatenated rich-plus-HLT
particle set transfer more of its performance to HLT than the existing U/D
ladder, if support stays fixed and only the rich particle contents change?

The user has observed promising concatenation results in earlier experiments.
Those motivate this study; they do not establish performance on this dataset,
after the new overflow policy, or after distillation.

The user-selected path is:

```text
D100 -> D075 -> D050 -> D025 -> D000 -> HLT_X1_COMPRESSED
```

```text
D100: offline + original HLT + HLT fillers, exactly 3*N_HLT particles
D075/D050/D025: rich slots progressively move toward their HLT owners
D000: HLT x3, three identical raw copies of every HLT particle
HLT_X1_COMPRESSED: a separate ordinary HLT x1 student trained by logit KD
```

All ladder models process one set with one encoder/classifier. There is no
HLT/offline source feature, branch identifier, slot embedding, learned
two-encoder fusion, output ensemble, or context-withdrawal gate. Matching and
slot ownership are training-view construction metadata, never model inputs.

HLT x3 and HLT x1 are both deployable. Intermediate rich views are privileged
training inputs, not deployable classifiers. Replication adds no new measured
HLT information; whether it affects the learned classifier, and whether a
single-copy student retains its performance, are experimental questions.

## 2. Dataset and relation to existing work

Use the same authenticated September-18 dz-fixed JetClass2 Delphes snapshot
and exact TRAIN_500K population as the separate fusion-chain study:

- 500,000 training jets;
- 1,000,000 validation jets;
- 1,000,000 final-test jets, sealed throughout preparation and development.

The authoritative location and provenance reference is the
[dz-fix matching handoff](../JETCLASS2_DZFIX_SALIENCE_MATCHING_HANDOFF.md).
The documented dataset root on SPORC is:

```text
/home/ryreu/atlas/datasets/jetclass2_10M_20260918_dzfix_partial_v1/jetclass2
```

The confirmed Windows-local parent snapshot is:

```text
C:/Users/22rya/ComputerScience/CERN/data/jetclass2_10M_20260918_dzfix/jetclass2
```

It has 331 authenticated files; the SPORC partial snapshot has 186 of those.
They are not interchangeable file sets. The local inventory and partial-transfer
plan authenticate their relationship; see the
[K=2 capacity audit](../JETCLASS2_DZFIX_K2_CAPACITY_AUDIT.md).

Authenticate inventory, exact role membership, particle decoding, and parent
foundation hashes. Do not import the September-10 dataset or CMS checkpoints.
The input remains the common 17-feature, 11-class interface, but token capacity
and support semantics are new and require a separate contract.

References, not interchangeable execution specifications:

- [Existing fusion-chain plan](JETCLASS2_DZFIX_FUSION_CHAIN_500K_PLAN.md).
- [Salience matching guide](../HCWDL_FULLCARD_SALIENCE_MATCHING_CODE_GUIDE.md).
- [One-to-one matcher contract](../contracts/HCWDL_FULLCARD_SALIENCE_MATCHING.md).
- [Persistent-HLT view contract](../contracts/JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT.md).

The old no-truncation policy and one-to-one assignment format remain unchanged
for existing campaigns. This plan explicitly permits only the new rich-pool
overflow cropping described below. It does not permit truncating native HLT
inputs or silently reusing one-to-one assignments as capacity-two assignments.

## 3. Particle pools and fixed support

Let H be the native HLT collection, n = len(H), and R the full native offline
collection before the explicit capacity-overflow cropping. Thus:

```text
m = len(R) = N_offline
K = 2                         rich slots available per HLT owner
total active slots = (K+1)*n = 3*n
```

The user's ladder clarification supersedes the initial draft's use of the
persistent-HLT U000 multiset as R. D100 is offline plus HLT plus fillers, not
that hybrid U000 plus HLT. In particular, do not first add old unmatched-HLT
entries to R: native HLT support and all necessary fillers are provided by
this study's own fixed-slot construction. Keep each offline entry's native
identity internally; it may be retained and assigned at most once.

When m <= 2*n, D100 contains every offline particle once, every native HLT
particle once, and exactly 2*n-m HLT fillers. When m > 2*n, the declared
least-salience cropping retains 2*n offline particles and no fillers are
needed. "Full offline" therefore means uncropped except for these explicitly
registered overflow cases; it is not a promise that cropping never occurs.

For every native HLT particle h_i, reserve three slots:

| Slot | Rich start | Intermediate rung | HLT endpoint |
| --- | --- | --- | --- |
| Original | h_i | h_i | h_i |
| Rich 1 | assigned R particle, or h_i if vacant | assigned particle moves toward h_i; filler stays h_i | h_i |
| Rich 2 | assigned R particle, or h_i if vacant | assigned particle moves toward h_i; filler stays h_i | h_i |

Here K=2 means three total copies at deployment, not three additional copies.
Vacant rich slots contain real HLT fillers with active masks from the start.
They are not zero padding and do not appear halfway through training.
Ordinary batch padding outside the 3*n active tokens remains masked.

Example: n=30 and m=50 produces 30 original HLT tokens, 50 rich tokens, and
10 HLT fillers: 90 active tokens at every rung. At the endpoint these become
exactly three copies of each of the 30 HLT particles.

Use native HLT order followed by a deterministic within-owner slot order,
for example the interleaved sequence (h_0, r_01, r_02, h_1, r_11, r_12, ...).
Do not encode that order through positional/source features. Canonicalize the
exchangeable rich slots deterministically. At the endpoint, no rich ordering
or matching lookup is needed because the three entries for each owner agree.

## 4. Overflow: retain salient rich particles, never remove HLT support

The retained rich count is min(m, 2*n). If m exceeds 2*n, drop exactly
m - 2*n rich entries, ranked from least salient upward. Break equal integer
salience scores by a frozen native-provenance order. Then solve the assignment
on the retained set. This ordering is important: do not drop a salient but
distant particle just because it would make the geometric matching easier.

Compute rich salience using the full original R collection and HLT salience
using the original H collection. Freeze scores before cropping, filling, or
duplicating target slots. Do not renormalize salience after removing particles.
The salience formula is borrowed from the current matcher, but its application
to the capacity-two problem is a new scientific policy.

Never drop an original HLT slot, reject a jet for overflow, introduce a
variable deployment replication count, or increase K automatically. Only
offline entries can be cropped; native HLT support always remains. Low
salience is not a guarantee of low classification value.

The present source population is nonempty on both endpoints. An unexpected
empty HLT collection is a contract violation, not permission to silently drop
the row or infer a fallback from offline information.

## 5. Global salience-aware capacity-two assignment

Every retained rich entry must have one HLT owner; each owner can receive at
most two entries. An owner need not receive any rich entry: fillers guarantee
its three active slots regardless. There is no geometric veto that discards
additional retained entries. Forced matches are training coordinates, not
claims of true detector correspondence.

Represent each HLT owner by two exchangeable destination slots and solve one
global rectangular assignment of retained R entries into the 2*n destinations.
Do not use independent nearest neighbors followed by overflow repair.

The recommended extension preserves the current primary utility:

```text
qdr(i,j)   = canonical quantized deltaR(h_i, r_j)
closeness  = max(QCAP - min(qdr(i,j), QCAP), 0)
utility    = (salience_H[i] + salience_R[j]) * closeness
objective  = maximize the sum of utility over assigned rich entries
```

Use the existing float64 wrapped-phi convention, delta-R quantum 1e-7, and
primary closeness cap 0.8. Each of an owner's two destinations receives the
same original HLT salience, not a new score computed on duplicated particles.

The new contract must explicitly adapt the secondary hierarchy: uncapped
delta R, selected endpoint salience, absolute log-pT response, category
mismatch, valid charge mismatch, then canonical native-index ties. In this
orientation all retained rich entries are selected, so their total salience
is constant; a selected-HLT-slot salience term counts occupied destination
slots, not distinct owners. Freeze this interpretation and the canonical slot
permutation before implementing the solver; do not blindly reuse an old
one-to-one mapping validator or floating epsilon weights.

The implementation follows the recommended prior-screen-winner source of the
salience formula (linear, quadratic, or quadratic-core25), not a claim that it
is optimal for capacity two. Its immutable launch registration explicitly
records that choice; a different choice requires a new registration. A new
three-candidate performance screen, if wanted, is a separate registered
extension rather than an implicit addition to this study.

Regardless of formula choice, publish a fresh capacity-two matcher/foundation
and run its own exhaustive small-case, data, and resource acceptance. Existing
one-to-one maps are not needed to define the raw offline pool and are not the
new assignment. If importing the prior selected salience formula, authenticate
its source selection/foundation as provenance without using its maps as ours.
Labels, gradients, attention scores, and final-test information are forbidden
in salience, retention, assignment, and tie-breaking.

## 6. Fixed-slot progression and feature construction

Use the user-facing names D100, D075, D050, D025, and D000, with distinct
artifact IDs CONCAT_K2_D100 through CONCAT_K2_D000. The short names D50 and
D00 mean D050 and D000. D100 is the retained/padded rich concatenation; D000
is HLT x3. None is an alias for historical U000, U100, or D000 artifacts:
historical D000 has one native HLT copy, whereas this study's D000 has three.

Let a in [0,1] denote retained rich strength, with a=1 at D100 and a=0 at
D000. The exact registered rung fractions are:

| Rung | Rich strength a | HLT interpolation strength 1-a |
| --- | --- | --- |
| D100 | 1 | 0 |
| D075 | 3/4 | 1/4 |
| D050 | 1/2 | 1/2 |
| D025 | 1/4 | 3/4 |
| D000 | 0 | 1 |

These strengths control content, not how many offline tokens are kept at each
rung: cropping and assignments are fixed before the ladder starts. For an
assigned offline particle r_j owned by h_i:

```text
continuous p4(a) = a*p4(r_j) + (1-a)*p4(h_i)
```

Adapt the registered D-style mixed-type transformation: interpolate applicable
continuous measurements; switch charge/category and applicability/validity
groups atomically with deterministic nested thresholds. Do not linearly blend
one-hot identities or independently mix invalid value/error pairs. Hash domains
must include the new view contract and stable jet/slot identities. Assignment,
retention, fillers, and thresholds are fixed across rungs, epochs, and workers.
Native HLT originals and filler slots remain unchanged in raw particle space.

Rebuild all model features, axes, normalization, pair inputs, and masks from
the current complete supplied set under the frozen transform. The derived
features of an unchanged raw HLT token can therefore change at intermediate
rungs as the whole set changes. Do not preserve rich-start jet normalization
in a deployable endpoint. Equal copies mean equal raw particle contents; do
not silently replace their transform with repeated precomputed x1 features.

At a=0, bypass rich data and assignments entirely: a dedicated HLT-only builder
repeats each native HLT particle three times and constructs the full input.
Require byte-equivalence between this builder and the a=0 construction path,
including derived features and masks. A deployment artifact cannot require
offline counts, salience, crop decisions, parent maps, or stored rich slots.

Uniform duplication preserves information availability, not necessarily model
predictions. Do not assume a trained x3 model can be converted to x1 merely by
deleting duplicate tokens.

## 7. Training graph and final compression

The agreed graph structure is a fresh rich concatenation teacher, a sequence
of progressively less-rich fixed-slot students, an HLT x3 endpoint, and a
separately trained HLT x1 compression student. No U support-removal stage is
needed, and no model changes its view continuously within a fit in this design.

The user-selected ladder is fixed to quarter steps:

```text
CONCAT_K2_D100  (fresh CE teacher: offline + HLT + fillers)
  -> CONCAT_K2_D075
  -> CONCAT_K2_D050
  -> CONCAT_K2_D025
  -> CONCAT_K2_D000  (HLT x3)
       -> HLT_X1_COMPRESSED
```

This replaces the earlier provisional C100/80/60/40/20/0 proposal. It specifies
one starting CE fit, four ladder KD fits, and one final x1 compression fit;
controls are additional fits. Recording the graph is not live authorization.

Each arrow is ordinary logit KD from the selected immediate-parent model on
the same training rows. Recommended inherited recipe: fresh initialization,
25% CE / 75% temperature-2 KD, batch 256, the established H45/decay60/floor100
schedule with early stopping and best-checkpoint restoration. The KD kernel,
seed registry, optimizer, checkpoint tie rules, and
early-stopping semantics must be frozen in the new implementation contract.
Do not infer a new learning-rate or batch-size sweep from this plan.

The final student consumes only ordinary HLT x1; its teacher consumes HLT x3.
This is another distillation fit, not a change to the x3 checkpoint. Preserve
and report both models, including any loss in compression. There is no promise
that either ladder transfer or final compression will be lossless.

## 8. Controls and interpretation

The minimum proposed comparison panel is:

| Model | Purpose |
| --- | --- |
| HLT_X1_CE | ordinary deployable HLT baseline |
| HLT_X3_CE | isolate the effect of uniform repetition without privileged KD |
| OFFLINE_CE | same-population pure-offline reference |
| CONCAT_K2_D100_CE | measure the actual cropped/padded concatenation teacher |
| DIRECT_HLT_X3_KD | rich teacher -> x3 without intermediate rungs |
| LADDER_HLT_X3 | endpoint of the registered fixed-support ladder |
| HLT_X1_COMPRESSED | deployment-sized compression of LADDER_HLT_X3 |

Use identical authenticated jet membership, validation partitions, model
architecture family, and matched initialization/sampler seeds for directly
comparable fits. Record training compute and inference cost; the ladder is
not compute-matched to a direct fit. No seed ensemble is part of the first
test. Existing checkpoints are not automatically reusable merely because
their names resemble these controls.

Optional follow-ups, not part of the minimum panel: unconstrained raw
concatenation CE to isolate crop/filler effects; rich-teacher -> HLT x1 direct
KD; K=1/3 comparisons; multiple independent seeds. Do not add these silently.

The first rich-view comparison is CONCAT_K2_D100_CE versus OFFLINE_CE on the
same registered validation rows and role. D100 outperforming pure offline is
a hypothesis, not an acceptance threshold. If D100 underperforms, retain and
report the result, examine class-specific metrics and crop/filler diagnostics,
and discuss a separately registered follow-up. Do not attribute any deficit
to cropping alone, automatically change K, or cancel later registered fits.

Use the existing deterministic validation-role pattern as the recommended
default: 50% checkpoint selection, 25% diagnostics, 25% reporting. Early
stopping uses checkpoint only; comparison tables use report only. Since the
parent salience screen used this validation reservoir, report is not an
untouched final test. All displayed recovery values must compare models on
the same role and rows, never live checkpoint AUC against report-subset AUC.

Report accuracy, macro AUC, macro linear R50, per-class QCD rejection and
recovery relative to HLT_X1_CE=0% and OFFLINE_CE=100%. Label rich references
as privileged and x3/x1 as deployable. Also report direct-versus-ladder gains,
compression loss, selected/completed epochs, wall time, memory, and inference
cost. Weak finite results are valid results, not gate failures.

## 9. First step: audit K=2 before implementing training

The [local audit](../JETCLASS2_DZFIX_K2_CAPACITY_AUDIT.md) has now measured
6,858,920 selected jets across 234 authenticated files, conservatively excluding
the union of test files from both local and SPORC splits. Exactly 32,452 jets
(0.473136%) overflow; 99.526864% need no crop. The SPORC-overlap population has
3,593,995 audited jets and 0.492210% overflow. This is a broad outer-reservoir
census, not a claim to have evaluated the exact TRAIN_500K row profile.

Class dependence matters: overflow is 0.1234% for QCD, about 0.14-0.23% for
hadronic signal classes, but 11.8598% for X_mm and 6.5912% for X_tauhtaum.
Keep the declared overflow policy and per-class diagnostics; do not infer
negligible scientific impact from the population-weighted rate. Actual
salience-policy information loss and classifier retention remain to be tested.
The known foundation capacity of 320 alone could not establish this frequency.

The second audit pass checked actual particle-array lengths on the entire
census. Dropping exactly the excess particles with smallest raw pT gives a
median minimum lost-pT fraction of 0.4490% among overflow jets, but 2,767 jets
(0.040342% of all audited jets) must lose more than 10%. This lower bound does
not replace the frozen salience-policy audit or demonstrate classifier
retention. Full results and reproducible artifacts are in the linked audit.

Decision after reviewing the K=3 alternative: **retain K=2 for the first
experiment**. K=3 still overflows in 13,849/6,858,920 jets (0.201912%), compared
with 32,452 (0.473136%) for K=2, while adding a fourth token per HLT particle
to every jet. The user accepts the measured K=2 count-overflow tradeoff for
this study, including its class dependence; this is not a claim of harmless
cropping. Keep exactly three active slots per HLT particle, crop only excess
least-salient offline entries, and retain all original HLT particles and jets.
K=3 is not part of the initial campaign. Reconsideration, if motivated by
validation diagnostics, requires a distinct follow-up specification and never
uses sealed final-test results to tune this choice.

The intended quick check is whether violations of N_offline <= 2*N_HLT are
rare. The <= case fits without cropping; the > case is the overflow case we
hope is rare. Report both fractions (including equality separately) rather
than assume either frequency. This count audit does not need the new matcher
or the parent salience screen to finish; it needs authenticated input counts
and the exact train/validation membership only.

Run a bounded read-only audit over authenticated train and validation rows:

1. Count m, n, min(m,2*n), max(0,m-2*n), and max(0,2*n-m) per jet.
2. Report overflow frequency, lost-particle count/fraction, filler frequency,
   count-ratio quantiles, and maximum active/padded x3 length.
3. With the frozen proposed salience formula, report lost scalar-pT and lost
   salience fractions, both per-jet distributions and population-weighted
   totals. Class-wise summaries are diagnostics only, never selection inputs.
4. Audit assignment occupancy (0/1/2 per owner), delta-R tails, salience-weighted
   match quality, and the number of forced distant matches once maps exist.
5. Compare input reconstruction for all endpoints and selected intermediate
   rows; report every overflow case under the declared policy.

Use scalar count branches for the count census where possible; particle data
are needed for pT/salience loss and matching. Do not read final-test particles
or optimize K/cropping thresholds against classifier report/test performance.
If K=2 looks scientifically undesirable, discuss and register a new design;
do not change K, remove jets, or suppress poor diagnostics automatically.

## 10. Resources, artifact isolation, and queue prerequisites

Target SPORC/debug for this separate experiment, subject to its own measured
acceptance and exact pushed source. Existing debug/tier3 jobs stay untouched.
Debug's current 24-hour limit must be checked against measured training cost.

Three times as many tokens means roughly nine times as many dense attention
pairs as native HLT at the same jet length, not a guaranteed ninefold wall-time
increase. Determine capacity from authenticated HLT counts and the x3 policy,
not the old single-view 320 constant. A conservative bound from 320 native
slots would be 960 before any required padding, but the actual HLT-only bound
must be calculated. Never silently clip the expanded input to fit old memory.

Measure full-population cache RAM, longest-batch installed-Weaver forward/
backward and optimizer peaks, pair-bias temporaries, throughput, and inference
cost. Old single-view or two-encoder acceptance cannot certify this input.
Any microbatching, batch-size, trimming, or precision change needs explicit
registration and verification, not an automatic performance workaround.

Create separate contracts for the source import, capacity-two matcher,
retention/slot policy, view transform, foundation, graph, campaign, acceptance,
teacher banks, and reports. Bind hashes and parent identities, not bare paths.
Keep compact assignments, retention identities, audits, selected weights,
reports, and class-probability banks; do not persist dense cost matrices or
all materialized particle views. Recovery and cancellation are exact-ID only.

If the old selected foundation/formula is used as a parent, authenticate the
screen's completion, selection, and foundation before new preparation. The
current screen completion is 21748725; derive and validate the actual ID from
its live ledger rather than hardcoding it into reusable code. If completion
is already durable, authenticate it rather than depend on an expired job ID.
This dependency only publishes a parent: new capacity-two preparation and its
own gates must still run. Do not depend on or modify the separate jc2fc ladder.

## 11. Required tests and implementation order

Implementation should proceed in separately verified stages:

1. Review the completed count/minimum-pT audit, freeze remaining choices, and
   implement the exact registered salience-policy loss audit.
2. Add new matcher/retention contracts, exact capacity-two solver, compact
   assignment publication, and bounded exhaustive-reference tests.
3. Add fixed-slot view construction and standalone HLT x3/x1 adapters.
4. Add the fresh CE/KD graph, controls, final compression, reporting and dry run.
5. Obtain genuine installed-Weaver/SPORC acceptance on the expanded inputs.
6. With explicit authorization, queue the isolated source-pinned campaign.

Required correctness tests include exact 3*n active support at every rung;
no loss of original HLT; deterministic salience retention before geometry;
rich-entry uniqueness and owner capacity; exchangeable-slot ties; exhaustive
small-case optimum; no-overflow/overflow/filler/unequal-count examples;
thread/shard-independent bytes; mixed-type validity; exact HLT-only endpoint
parity with offline access unavailable; corruption and wrong-parent rejection;
sealed final-test access; train-bank row alignment; and no existing-artifact
mutation. Installed-Weaver tests must exercise duplicate p4 pairs and prove
finite outputs/gradients and adequate memory at the longest registered length.

## 12. Decisions recorded versus still open

Recorded from this discussion:

- K=2 rich destinations per native HLT particle; three total active tokens.
- K=2 count-overflow tradeoff accepted after the broad census and K=3 check;
  D100 versus pure offline remains an empirical validation comparison.
- One undifferentiated concatenated set, not learned two-branch fusion.
- D100 starts from native offline + native HLT + required HLT fillers.
- Least-salience rich-only cropping for overflow; fillers otherwise.
- Global salience-aware capacity-two matching and fixed support throughout.
- No separate U removal ladder; only rich-to-HLT content progression.
- Exact ladder D100 -> D075 -> D050 -> D025 -> D000, with quarter strengths.
- HLT-only x3 endpoint followed by a distinct logit-KD x1 compression fit.
- Separate experiment on the dzfix population; existing campaigns untouched.

Implementation decisions frozen for this first campaign:

- Authenticate the existing debug screen's winning salience formula; construct
  fresh K2 assignments, with frozen native salience and the exact ordered
  utility/tie objectives specified in the new contract.
- Ten fits: the six-model D ladder/compression, HLT x1 CE, HLT x3 CE,
  pure offline CE and direct D100-to-HLT-x3 KD. All cold, paired seeds,
  C25P75/T2 students, batch 256, the existing 100-pass hold/floor-tail recipe.
- Separate class-stratified 50/25/25 checkpoint/diagnostic/report validation;
  sealed final test. Parent matching selection used the validation reservoir.
- Derive expanded capacity from inventory, not 320. The archived SPORC
  inventory implies 832 tokens; workers authenticate/rederive it.
- All new launchers, preparation, acceptance, science and reporting jobs
  request debug. New acceptance must measure memory and runtime before science.

Remaining external requirement: exact pushed checkout and genuine installed-
Weaver/SPORC expanded-input acceptance. The queue workflow runs this gate
automatically; it never imports old acceptance or alters the recipe to fit.
No new scientific matching/classifier result, real GPU acceptance, or Slurm
submission is claimed by local implementation and tests alone.

# Seven supervision candidates: three implementation groups and a40-row campaign

Status: implementation plan v0.5, 2026-11-05. The user requested three implementation
groups, approximately40 runs weighted toward the stronger bets, and a queue
order that mixes candidate families with each preferred setup submitted first.
This document selects the campaign. Group A now has an additive implementation
and [runbook](../LUKA_NEXT_SUPERVISION_GROUP_A_RUNBOOK.md). Group B now extends
that foundation with its own [contract](../contracts/LUKA_NEXT_SUPERVISION_GROUP_B.md)
and [runbook](../LUKA_NEXT_SUPERVISION_GROUP_B_RUNBOOK.md); C remains unimplemented.
No immutable execution specification or live submission is created by this plan.

The [ideas catalog](LUKA_NEXT_SUPERVISION_CAMPAIGN_IDEAS.md) defines the seven
objectives and their wider variant space. This document selects the proposed
first campaign from that catalog. Catalog rows outside the selection are
deferred at planning time, not registered experiments to skip after seeing
performance. Existing campaigns/specifications keep their original semantics.

The user's queue policy is explicit: **science fits have no scheduler dependencies
on other fits; interleaved submission order alone encourages mixed-family
execution.** All groups' fits can be in the queue together. No priority tiers,
nice adjustments, holds or per-family scheduling lanes are part of this policy.
Preparation/technical acceptance must be complete before science submission;
that requirement is not a chain between candidate experiments.

## 1. Common design and budget

All selected arms retain native HLT inference and the original
`CE + 5*U100_embedding_SmoothL1` baseline. Use the catalog's proposed Hydra
ALLJETS population,299234 TRAIN/74895 development-validation jets, batch128,
30 epochs, matched initialization/sampler seeds and native17-input/11-class
recipe. Heads and target preparation differ; deployed student capacity does not.
Original50k overlaps validation and remains report-only; no independent test is
introduced. No privileged model is warm-started into the student.

Budget **40 comparison rows**:38 candidate/control configurations plus the two
shared native same-seed references C0/C1. Reuse compatible completed reference
fits by default. J5 may also be reusable. Thus40 rows need not mean40 new GPU
fits; the final manifest must distinguish reused and newly trained rows.
Do not retrain compatible controls or invent replacement experiments to spend
unused slots. At most40 new scientific fits are represented by this proposal.

Setup, target preparation, technical miniatures, durable release and reporting
are operational work, excluded from the scientific-fit count but included in
compute/storage accounting. Resource acceptance remains mandatory before science.

| Candidate | Family | Selected rows | Allocation | Reason |
|---|---|---|---:|---|
| 5: particle soup | P | P1,P2,P3,P4,P5,P6,P7,P8,P9,P11 | 10 | User's strongest current bet; tests several distractor laws |
| 6: intact pairs/triplets | G | G1,G2,G3,G4,G5,G8 | 6 | Tests joint information beyond individual-particle matching |
| 3: learned compatibility judge | J | J1,J2,J3,J4,J5,J6 | 6 | Extends the successful whole-jet identification objective |
| 4: structured questions | Q | Q1,Q2,Q3,Q5,Q6 | 5 | Covers richer physical targets and their interaction structure |
| 7: learned pattern frequencies | V | V1,V2,V6,V7 | 4 | Compact target with source and weighting controls |
| 1: embedding distribution | D | D1,D2,D3,D5 | 4 | Tests multiple hypotheses against learned single-Gaussian uncertainty |
| 2: set denoising | S | S1,S2,S5 | 3 | Larger cost/uncertainty; tests the main hypothesis before a wide sweep |
| Shared native references | C | C0,C1 | 2 | Matched seed0/seed1 U100-w5 references; reuse when compatible |
| Total | | | 40 | Includes source/mechanism controls and replication |

These allocations express research judgment and compute tradeoffs, not measured
probabilities of success. Every family's preferred recipe is represented in
both seeds. No new family-specific learned teacher fit is needed: V uses frozen
clustering preparation, and the other teachers/measurements are existing sources
to authenticate. The optional VQ tokenizer is not selected here.

## 2. Three implementation groups

| Implementation group | Candidates | Shared implementation | Selected configurations |
|---|---|---|---:|
| A | 3 learned judge,5 particle soup,6 pairs/triplets | Native adapter/trainer, source readers, compatibility scorers and candidate sampling | 22 |
| B | 4 questions,7 pattern frequencies | Structured target preparation, question oracle, frozen dictionaries and compact target banks | 9 |
| C | 1 embedding distributions,2 set denoising | Conditional likelihood/denoising modules, numerical/masking checks and measured decoder paths | 7 |

Implement these as A, then B, then C, sharing the common foundation. This ordering
is for engineering and review. It does not divide the science queue into three
campaigns or require group A experiments to finish before B/C experiments start.
The two shared native references sit outside the three method allocations.

### Group A: compatibility and membership, candidates3/5/6

Implement P first, then the shared group/jet extensions. This is22 selected
configurations:10 P,6 G and6 J.

The implemented foundation is `src/hlt_classification/luka_next_supervision_hydra`.
Its [science-v1/execution-v2 contract](../contracts/LUKA_NEXT_SUPERVISION_HYDRA.md) freezes the remaining
feature, bin-expansion, capacity and bounded G-feature-reservoir decisions.
It preserves the40-row registry and rejects a full40 release until C is
implemented/qualified. An explicitly reviewed Group A-only release filters the original
interleaving, reuses C0/C1 (and compatible J5), and adds no fit dependencies.
This optional partial release does not replace the combined first-seven policy.

The first genuine Hydra setup150801 failed resource admission on P1 before any
science release. Execution v2 reduces indexing/preparation overhead and measures
individual production phases; it preserves the scientific v1 recipe and requires
a fresh resource acceptance. The existing raw transfer/references remain reusable.
This is an execution revision, not a change to the selected40-row campaign.

Build the common foundation as part of this group:

- Authenticate D000 inputs, original U100 bank, OFFLINE inputs/bank, reference
  fits, environment and source lineage. Audit Brown availability and export
  only missing required TRAIN fields. Preserve older source/specs/artifacts.
- Reuse a single native-model adapter and trainer, with isolated head RNG,
  explicit training-only target capabilities and native-only checkpoint replay.
  Separate raw source readers from family-specific coordinate transforms: P/G
  parent-local frames and S/Q HLT-relative frames are not interchangeable.
- Implement independent particle scoring, intact-group scoring and learned
  whole-jet compatibility as separate objectives over shared matching utilities.
  Preserve the original dot-product comparison and matched candidate streams.
- Implement declared sampling, exclusions, parent caps, feature/bin matching
  and bounded semi-hard mining, with source labels kept out of model inputs.
- Establish the common result/epoch format and per-row resource measurements,
  followed by one campaign registry, thin CLI/Slurm interfaces and exact-ID
  submission/recovery. Later groups extend this registry instead of cloning
  another complete execution stack.

Verification must cover positive/negative reductions, intact source groups,
candidate/member permutation, leakage controls, mining gradients, frozen target
roles, unchanged native initialization and HLT-only export. Poor finite matching
does not fail tests of scientific admission or cancel registered fits.

### Group B: questions and learned local-pattern statistics, candidates4/7

Implemented additively in the existing namespace as9 independent B rows, with
separate immutable registration/output roots and the common native trainer,
worker, resource measurement, submission journals and reporting. The user's
existing A jobs/source remain pinned and untouched. B reuses original C0/C1
and already transferred raw/U100 inputs; it does not wait for A science.

Implement Q and V on the authenticated raw-source foundation from A. This is9
selected configurations:5 Q and4 V.

Build semantic question encoding and the four-family target oracle, with
answer-independent sampling, source validity, bounded pair calculations and
per-family/per-jet loss normalization. Build the TRAIN-only frozen K128
dictionary, count/momentum frequency banks and separate HLT-source control.
Preserve the exact codebook/normalization identities on every consumer path.

Main evidence paths are Q's full-versus-marginal comparison and V's source and
momentum-weighting controls. The wider question-count, codebook-size and spatial
variants remain in the catalog. Tests must cover target formulas, sampling
independence, missing-value behavior, frozen codebook fitting/assignment,
frequency normalization and exact replay of compact banks.

The [B v1 contract](../contracts/LUKA_NEXT_SUPERVISION_GROUP_B.md) closes the
catalog/dictionary choices before execution registration. Q proposes up to2048
distinct semantics per family and measures support over the entire TRAIN set,
retaining up to1024 nonconstant entries with no rare-prevalence cutoff. Actual
shortfall is disclosed in the frozen catalog/science review without duplicate
padding. If a whole family is constant, its fixed queries have zero applicability
and are reported as supplying no nonconstant supervision; finite poor targets
never fail or cancel a row. This replaces the earlier unresolved instruction
to change the draft only after seeing short support. Sampling uses semantic
parameters, never per-anchor answers or class. V uses frozen K128 MiniBatchKMeans,
equal TRAIN block energy and separate OFFLINE/HLT target banks. These data-driven
target realizations are frozen in setup before any B scientific fit.

B alone submits in original filtered order: Q1,V1,Q2,V2,Q5,V6,Q3,V7,Q6. Its
six distinct source/loss paths require genuine warmed Hydra resource acceptance.
It shares A's6CPU/32000MiB A6000 profile,48h ceiling and full30 budget. There is
no dependency on an A scientific fit and no fit-to-fit dependency within B.

### Group C: conditional distributions and set denoising, candidates1/2

Implement D and S through the same model/training/registry interfaces. This is7
selected configurations:4 D and3 S.

D consumes the existing standardized U100 target and adds the whole-vector
K4 mixture or its K1 learned-variance control. Verify joint likelihood
reduction, scale bounds, FP32 logsumexp, gradients and valid negative NLL.
S consumes raw paired OFFLINE sets and adds the query-conditioned denoiser and
count head, with a corresponding HLT-set control. Verify noise/velocity targets,
permutation, padding/measurement masks and absence of target access from the
classification route. Measure the complete preparation/full-cardinality path.

This group comes last in implementation because its new numerical and decoder
paths have more technical uncertainty. That does not make its science jobs
wait for A/B training results.

### Shared completion standard

The groups are coding milestones, not three serialized scientific campaigns.
Separately authorized A-only and B-only releases may run alongside one another.
Complete and technically validate all three before releasing the combined
first campaign so its first seven primary submissions span all seven candidates.
Each group should have focused tests and a concrete reviewable change set.
Record exact donor files/commits in LEGACY_SOURCE_MAP during implementation;
add new versioned recipes/targets/specs rather than repurposing old contracts.

Before release, require installed-Weaver parity, native export replay, full dry
review and genuine warmed Hydra production miniatures/resource measurement for
every selected execution path. Reuse measured path evidence only when the
configuration and support actually match. Do not duplicate an expensive measured
path solely for an identical seed variant, or skip a distinct loss/sampler path.
No scientific result is inferred from local miniature accuracy.

## 3. Queue all science fits independently in interleaved order

The first seven scientific submissions are the preferred seed0 settings,
one per candidate, in confidence order:

```text
P1 -> G1 -> J1 -> Q1 -> V1 -> D1 -> S1
```

The arrows describe successive submission calls, not execution dependencies.
After the common input/target preparation and real technical acceptance are
complete, the queue helper submits every non-reused scientific row in the table
below. It does not stop after the first seven to wait for starts, epochs, results
or completion. All candidate families and all remaining variants can then be
pending/running together.

The scientific submission contract is:

- One independent one-GPU job per new row; skip authenticated reused rows.
- No `--dependency` flag on scientific jobs: no `afterok`, `afterany`, `after`,
  `singleton`, finish-to-start chain or hidden first-wave barrier between fits.
- No family-specific lanes, artificial array-concurrency cap, user holds or
  relative `--nice`/priority changes. Site/account resource limits still apply.
- Do not request all seven GPUs in one co-allocation. Any independent fit can
  start when its measured resources and scheduler policy permit.
- Native source/reference checks happen before reuse or publication, and actual
  setup acceptance precedes science submission. No scientific result supplies
  training targets or admission for another scientific fit in this graph.
- Record the exact row/job mapping and submission order durably. Once the
  whole graph is queued, ongoing dispatch does not depend on a login session.
  Inspect partial submission journals before any recovery; never blindly retry.

The ordering has three readable portions: preferred seed0 fits at1..7,
references/seed1 fits at8..16, and interleaved controls/variants at17..40.
These portions have no scheduling effect beyond their position in submission
order. They are not Slurm priority tiers or delayed-release stages.

Slurm priority, resources, time limits and backfill determine actual starts.
A later-submitted job may start first. The plan encourages broad early coverage
but cannot promise a particular start sequence or simultaneous starts for all
seven families. Accurate measured requests help backfill without changing the
scientific work.
[Slurm scheduling guide](https://slurm.schedmd.com/sched_config.html).

### Exact proposed40-row order

| Position | ID | Role |
|---:|---|---|
| 1 | P1 | Preferred mixed particle soup,seed0 |
| 2 | G1 | Preferred intact pairs,seed0 |
| 3 | J1 | Preferred learned jet judge,seed0 |
| 4 | Q1 | Preferred full question task,seed0 |
| 5 | V1 | Preferred K128 count+pT frequencies,seed0 |
| 6 | D1 | Preferred K4 distribution,seed0 |
| 7 | S1 | Preferred OFFLINE set denoising,seed0 |
| 8 | C0 | Shared native seed0 reference;reuse by default |
| 9 | C1 | Shared native seed1 reference;reuse by default |
| 10 | P2 | Preferred particle soup,seed1 |
| 11 | G2 | Preferred pairs,seed1 |
| 12 | J2 | Preferred jet judge,seed1 |
| 13 | Q2 | Preferred question task,seed1 |
| 14 | V2 | Preferred pattern frequencies,seed1 |
| 15 | D2 | Preferred embedding distribution,seed1 |
| 16 | S2 | Preferred set denoising,seed1 |
| 17 | P11 | HLT-particle source control |
| 18 | G8 | HLT-group source control |
| 19 | J6 | HLT-reference-key control |
| 20 | Q5 | HLT-answer control |
| 21 | V6 | HLT dictionary/target control |
| 22 | D5 | Single-Gaussian uncertainty control |
| 23 | S5 | HLT-set denoising control |
| 24 | P3 | Gentler particle-soup weight0.1 |
| 25 | G5 | Independent-member score on the same pairs |
| 26 | J5 | Matched dot-product reference;reuse if compatible |
| 27 | Q3 | Gentler question weight0.1 |
| 28 | V7 | Count-only pattern frequencies |
| 29 | D3 | Gentler mixture weight0.1 |
| 30 | P5 | Broad particle distractors |
| 31 | G3 | Gentler group weight0.1 |
| 32 | J3 | Gentler jet-judge weight0.1 |
| 33 | Q6 | Marginal-only question families |
| 34 | P6 | Same-class particle distractors |
| 35 | G4 | Intact triplets |
| 36 | J4 | Stronger jet-judge weight1 |
| 37 | P7 | Parent-kinematic particle distractors |
| 38 | P8 | Particle-feature-matched distractors |
| 39 | P9 | Semi-hard mixed particle distractors |
| 40 | P4 | Stronger particle-soup weight1 |

All weights, samplers and model defaults are the corresponding catalog
definitions. Families are intentionally mixed, with extra late P slots reflecting
its larger allocation. The first variant of every family is its preferred
configuration, not a cheap control. The order is independent of validation
performance and does not change automatically after inspecting early results.

### Results as jobs finish

Implement a read-only partial-results view that works from the start of the
campaign. It authenticates and reports every completed row, regardless of whether
the other six candidates or remaining variants have finished. Show each family's
completed selected checkpoints, AUC, historical recovery, selected epoch and
same-seed control delta. A pending/missing control yields an explicit unresolved
delta; never substitute another seed or invent a completed comparison.

The progress view reads published epoch metrics from running fits and reports
epoch-by-epoch AUC/recovery, best so far and the last logged progress marker.
Label these provisional and keep them distinct from authenticated completed
fit results. Both views mark failed, pending and incomplete rows explicitly;
neither starts inference/training, changes jobs or rewrites source artifacts.

Final report assembly is on demand after attempts terminate. No dependent report
job is needed to obtain interim results, and no candidate is held for report
generation. All registered rows retain their declared budget despite poor
finite performance; early results do not cancel or automatically reprioritize
the queued variants.

## 4. What the first campaign defers

Keep the full catalog for traceability. The first40-row selection defers the
most aggressive hard mining, positive-stratification/pool-size sweeps, extra
kinmix judge comparisons, wider dictionary/question/denoising sweeps and combined
multi-objective variants. They are not silently deleted from prior ideas.
The selection reserves budget for second seeds and meaningful controls.

Combined objectives, neural tokenizers, larger deployed models and the proposed
alternative U/D-ladder use are not added to this first campaign. If later results
motivate additional experiments, register those separately. Every selected row
completes its declared budget despite poor finite results; technical corruption,
forbidden access, nonfinite required values and invalid lineage fail closed.

## 5. Remaining implementation decisions

The grouping, row selection and interleaving are now reviewable. Before live
execution, finish the catalog's exact data/source/masking/sampling/codebook
semantics, authenticate reference reuse, pin clean pushed source/environment,
implement versioned artifacts and durable release/recovery, and measure real
resources for the selected methods. Publish one manifest resolving all40 rows
to either authenticated reuse or an exact new task; do not infer reuse from
path existence or an old Slurm COMPLETED state.

Hydra RTX A6000 is the intended initial profile. Availability and actual memory/
runtime needs must be measured; this plan is not current queue telemetry.
Complete genuine data/technical prerequisites before submitting the independent
fits. Read interim results at any time and assemble the final report from saved
terminal-attempt evidence, including failed/incomplete rows. All current
campaigns/jobs remain untouched by this planning change.

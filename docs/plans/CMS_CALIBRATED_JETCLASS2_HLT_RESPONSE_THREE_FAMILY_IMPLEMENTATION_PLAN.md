# CMS-calibrated JetClass2 HLT response: three-family comparison

Date: 2026-09-17. Short name: **CMS2JC2-Response v1**.

Status: **implementation-authoritative design; local implementation and staged
queue tooling complete; full-science submission awaits real SPORC acceptance**. See the
[implemented boundary and remaining gates](../contracts/CMS2JC2_RESPONSE_PREPARATION.md).
The user requested this document, a substantial three-family comparison, and
SPORC execution. Writing this plan does not authorize live submission, new
remote access, deletion, changes to current jobs, or final-test access.

## 1. Scientific objective and exact scope

### 2026-09-17 user amendment: explicit provisional continuation

The user authorized continuing implementation and the registered full-size
comparison under most-likely, explicitly recorded producer assumptions while
awaiting Luka's response. This supersedes the requirement to pause all work
for producer confirmation; it does not authorize live submission or waive
resource, access, source, or final-test gates. Use the separately identifiable
`CMS2JC2_RESPONSE_PROVISIONAL_COMPATIBILITY/v1` artifact rather than labelling
assumptions as verified review. Its immutable protocol is
`CMS2JC2_NATIVE_PROVISIONAL/v1`: GeV p4, CMS cm -> mm, Delphes mm, CMS dxy sign
as canonical and Delphes D0 sign flipped, finite displacements retained with
missing errors explicit, regular CMS PF only, no lost-track predictors.

Preserve stored p4: the inspected CMS writer is unweighted while the JC2 card
exports PUPPI-processed particles. This known domain difference and differing
vertex references are not magically removed by unit conversion. No inverse
PUPPI weights or absent event-level vertex corrections are invented. CMS
held-out closure may be reported under these assumptions; JetClass2 transfer
remains explicitly provisional and cannot earn a physically qualified label.
Producer confirmation/corrections require a new compatibility artifact and new
dependent fits where input semantics change, never mutation of old outputs.

Learn the conditional change from native CMS FullSim offline constituents to
native CMS HLT/scouting constituents, using only input quantities that can also
be constructed defensibly from JetClass2 offline constituents. Freeze the
response and apply it to JetClass2 offline jets to construct a new HLT proxy.

The primary scientific question is whether a small, interpretable response
model can reproduce important **conditional and joint** CMS reconstruction
differences on unseen CMS source files, then operate sensibly on JetClass2.
It is not whether the response can make the existing KD method perform well.

```text
CMS paired raw rows, ordinary training-file reservoir only
    -> shared physical representation and response-specific associations
    -> three response families x three complexity settings
    -> large-data held-out model comparison
    -> immutable selected-response lock
    -> separate CMS confirmation-file evaluation
    -> frozen response applied to JetClass2 OFFLINE ONLY
    -> transfer diagnostics and qualification report
```

The output is named a **CMS-calibrated HLT proxy**, never actual CMS HLT,
reprocessed CMS detector simulation, or truth-level detector reconstruction.
Both input collections are already reconstructed. The calibration estimates
an offline-to-HLT conditional response; it is not a generator-particle detector
simulation and need not reconstruct the particular hidden fluctuation in each
observed CMS pair.

### 1.1 Decisions frozen by this plan

- Compare three families: low-dimensional calibration, neighbourhood-aware
  smooth calibration, and small neighbourhood-aware boosted trees.
- Compare three complexity settings per family: nine full-budget candidates.
- Every candidate reaches the full registered calibration budget. Poor early
  scores do not prune a family or cancel later registered work.
- Include nested 250k and 1M calibration learning-curve budgets in addition to
  the full budget: **27 response fits**, not 27 classifier trainings.
- Use at least 2M CMS calibration jets, at least 250k CMS selection jets, and
  at least 250k CMS locked confirmation jets, subject to authenticated capacity.
  Use all eligible jets in their allocated files, not just the minima.
- Use reproducible statistical response: fixed label-independent random keys,
  fixed response replica, no resampling between epochs.
- Default compute is CPU-only SPORC. No neural generator, diffusion model,
  GAN, large neural response network, or GPU classifier sweep belongs here.
- Preserve raw data, existing memberships, jobs, worktrees, and benchmarks.
- Response fitting and selection never use jet-class labels as model inputs,
  classifier scores, KD recovery, or JetClass2's supplied HLT observations.
- Poor closure completes normally with an unqualified scientific result;
  corrupt inputs, invalid lineage, forbidden reads, and nonfinite execution
  fail closed.

### 1.2 Relationship to existing authority

This is a new family, not a modification of the
[JetClass2 migration](JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md),
[salience benchmark](JETCLASS2_DELPHES_FULLCARD_SALIENCE_PERSISTENT_500K_PLAN.md),
or [native CMS study](CMS_SALIENCE_LEARNED_DENSE_500K_PLAN.md).
Their existing artifacts retain their original meanings.

For this family only, this plan explicitly permits CMS-fitted response
parameters to be applied to JetClass2. That is a deliberate new scientific
boundary, not a silent override of the old migration's prohibition on importing
FullSim calibrations. Likewise, SPORC CPU-worker acceptance replaces the generic
historical Tigris/GPU miniature for this CPU study only. A future neural
classifier study still needs its own installed-Weaver and real GPU acceptance.

## 2. Data locations, authentication, and access limits

### 2.1 Paths

| Purpose | Recorded location |
| --- | --- |
| Local repository | `C:\Users\22rya\ComputerScience\CERN\HLT_Classification` |
| Local CMS compact raw data | `C:\Users\22rya\ComputerScience\CERN\data\ScoutingAK8_native_compact\2024\train` |
| RC CMS compact raw data | `/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train` |
| Local JetClass2 raw data | `C:\Users\22rya\ComputerScience\CERN\data\jetclass2_10M_20260910\jetclass2` |
| RC JetClass2 raw data | `/home/ryreu/atlas/datasets/jetclass2_10M_20260910/jetclass2` |
| RC JetClass2 inventory | `/home/ryreu/atlas/datasets/jetclass2_10M_20260910/manifests/inventory.json` |
| RC JetClass2 split bundle | `/home/ryreu/atlas/datasets/jetclass2_10M_20260910/jetclass2_delphes_scaling_splits_20260911_v1` |
| RC repository | `/home/ryreu/atlas/HLT_Classification` |
| Login host | `sporcsubmit.rc.rit.edu` |
| Recorded x86-64 scientific environment | `/home/ryreu/miniconda3/envs/atlas_kd_sporc` |

The local CMS directory was inspected and contains 53 ROOT files. The RC CMS
path is documented by the Scouting usage guide; **visibility from a SPORC
compute node has not been established by writing this plan**. Both remote
datasets and the chosen CMS split manifest must pass a read-only worker check.
Do not silently fall back to another directory, to the original ten-class
JetClass dataset, or to a newly generated split.

The documented historical CMS split-manifest reference is
`/home/ryreu/atlas/HLT_Classification/checkpoints/pmard_pilot_c3e40850_prefix_recovery_r5/data/splits/split_manifest.json`.
This is a discovery hint, not evidence that the path still exists. Creation
requires an explicit authenticated `--cms-split-manifest`; relocating identical
bytes is permitted, inventing an equivalent-looking split is not.

Sources: [SPORC dataset handoff](../JETCLASS2_DELPHES_SPORC_AGENT_HANDOFF.md),
[native CMS plan](SCOUTING_ALPHA_REPAIR_DISTILLATION_OPTIMAL_CAMPAIGN.md), and
the local `C:\Users\22rya\ComputerScience\FCV\SCOUTING_AK8_USAGE_GUIDE.md`.

### 2.2 What is reused, and what is not

Authenticate file hashes, schema, latest ROOT tree cycle, entry counts, source
manifest, original role membership, and exact row identities before use.
The losslessly compacted CMS files contain paired rows; do not rematch jets
across entries or count multiple ROOT cycles as independent events.

Reuse raw data and original role boundaries. Do not import fitted classifiers,
teacher banks, old normalization constants, forced full-cardinality assignments,
or old resource acceptance as response-calibration evidence.

The CMS calibration population starts from the original `train` file reservoir
and registered baseline/mapped-row selection. The historically observed train
count was 2,777,855; this is a capacity expectation to verify, not a hardcoded
identity. Calibration is conditional on those paired-jet selection cuts. It
does not estimate the rate of offline jets with no reconstructed HLT jet.

### 2.3 JetClass2 supplied HLT is not an input or target

Create a genuinely offline-only reader route. Neither `hlt_part_*`, HLT jet
kinematics, `hlt_jet_dr_offline`, nor freshly read `hlt_matched` may be required
by generation, response fitting, or transfer diagnostics. Test this with HLT
branches absent and with their contents changed.

For the initial transfer audit, reuse the exact existing TRAIN_500K training
membership and the 1M validation membership. These masks were originally
selected using `hlt_matched`; disclose this **historical selection conditioning**.
Reading frozen masks is allowed. Re-reading HLT to select convenient jets is
not. Extending to an offline-selected population independent of `hlt_matched`
requires a separate versioned selection/registry, not a quiet alteration here.

Do not read JetClass2 final-test particle branches, CMS original final-test
particle branches, or any sealed predictions. Metadata authentication follows
existing contracts and is not a scientific test-unlock permission.

## 3. Large-data split design

### 3.1 CMS outer response roles

Partition **only the original CMS training files** into:

| Response role | Target fraction | Minimum eligible jets | Permitted use |
| --- | ---: | ---: | --- |
| `response_fit` | 0.80 | 2,000,000 | Association calibration, response fitting, preprocessing |
| `response_select` | 0.10 | 250,000 | Nine-candidate comparison and predefined diagnostics |
| `response_confirm` | 0.10 | 250,000 | Locked, post-selection confirmation only |

The sum of hard minima is 2.5M. Whole-file grouping means the exact 80/10/10
fractions generally cannot be met. Use a deterministic integer allocation
minimizing maximum relative row-count deviation, then summed deviation, then
SHA256 file-order tie breaks (seed 20260917). Require both production-source
categories in every role. Class metadata may verify coverage, but must not
become response predictors or per-class response parameters.

The existing CMS `event_no` is not a globally unique physical event identifier;
the usage guide reports repeated values 1--500. Therefore **source-file
disjointness is mandatory**, not a per-particle or per-row random split.
If authoritative cross-file generation/event grouping is obtained, merge those
files into indivisible groups before allocation. Record the residual unknown
cross-file dependence if such metadata is unavailable.

If the frozen reservoir cannot satisfy the minima and grouping, publish a
capacity report and stop before science creation. Do not borrow original
validation/final-test files, split the same event across roles, or silently
reduce the advertised budget. A user-approved capacity amendment must precede
any different experiment.

The confirmation files are untouched **by this response development**, not
claimed never to have appeared in historical CMS classifier training.
Historical trained models are not reused in the response comparison.

### 3.2 Internal fitting roles and learning curves

Within `response_fit`, allocate whole source files into 80% `fit_location` and
20% `fit_residual` by the same count-based deterministic rule. The former fits
probabilities, locations, and scale predictors; the latter calibrates residual
distributions, probability calibration, and correlations without evaluating
them on their own fitted responses. Neither is a selection or confirmation
role. The full-budget model uses both subsets for these distinct purposes.

Construct nested **250,000**, **1,000,000**, and **FULL** memberships inside
this partition. Small budgets apportion 80/20 across the two internal roles and
proportionally across files, using smallest canonical-identity hashes without
replacement. FULL uses every eligible row in `response_fit`. Every candidate
uses identical memberships at each size.

All 27 fits run. Model selection compares the nine FULL fits only; smaller
budgets diagnose saturation and data dependence. Do not choose a smaller fit
because it happened to win on the selection sample, and do not treat fits at
different sizes as independent seeds.

### 3.3 Large jets, rare particle types, and statistical weighting

Do not truncate constituents to historical ParT limits (200 HLT or 90/60
offline). Read full available collections. Every eligible jet contributes to
jet-level/topology statistics; all compatible particles contribute to streamed
calibration summaries.

For nonlinear per-object fits that cannot use all particle records in RAM,
permit a single shared deterministic reservoir of at most 20M records per
target module. Stratify by observable particle category, momentum, and crowding,
never jet label. Store inclusion probabilities and inverse-probability weights;
all families use the same effective record selection. A 20M record cap is a
computational limit, not permission to restrict the number of contributing jets.
Report unique jets, object counts, weighted effective counts, and rare-stratum
coverage separately. Do not call millions of particles millions of independent
jets. Any cap/weight approximation must be visible in the fit report.

## 4. Shared physical interface and pre-fit compatibility gate

Use a new `CMS2JC2_SHARED_PARTICLES/v1` representation before any model fitting.
It is not either existing classifier's normalized input tensor.

| Quantity | CMS source concept | JetClass2 source concept | Required resolution |
| --- | --- | --- | --- |
| Four-vector | Charged/neutral offline and scouting p4 | `part_px/py/pz/energy` | Common units, weighting and physical convention |
| Charge | Native charge and neutral applicability | `part_charge` | Signed unit charges and neutral state |
| Particle identity | Hadron/photon/electron/muon flags | Five `part_is*` fields | Same exclusive categories, explicit unknown policy |
| Transverse displacement | `dxy` | `part_d0val` | Units, sign, reference point, applicability |
| Longitudinal displacement | `dz` | `part_dzval` | Units, reference point, applicability |
| Displacement uncertainty | Significance or a proven native error | `part_d0err/dzerr` | Defensible conversion and invalid-state handling |
| Neighbourhood | Recomputed from compatible offline particles | Same computation | Identical geometry and definition |

### 4.1 Mandatory compatibility decisions

- Canonical momentum unit is GeV and length unit is mm. Record documentary
  evidence and numerical checks for each conversion. Do not assume a cm/mm
  conversion merely because it makes distributions look similar.
- For this response, the shared offline population is regular PF candidates.
  CMS `cpfcandlt_isLostTrack` is used only to exclude lost tracks at ingestion,
  because an equivalent separate JetClass2 lost-track collection is not present.
  It is not a response predictor. Audit the effect of this restriction.
- PUPPI-weighted and unweighted four-vectors must not be interchanged. Determine
  the stored p4 meaning on both sides before freezing the bridge.
- A CMS uncertainty may be reconstructed as `abs(value/significance)` only
  where the producer definition establishes that relationship and both inputs
  are valid with nonzero significance. Zero/zero is missing, not zero error.
- Finite displacement with unavailable uncertainty remains a finite displacement
  plus an explicit internal validity state. No division by epsilon, invented
  significance, or unconditional erasure is permitted.
- Neutral tracking values follow documented applicability. Internal applicability
  and measurement-validity masks are permitted physical bookkeeping; response
  construction IDs and random keys are never classifier features.
- Source-only track quality, lost-hit, vertex association, truth, labels, and
  sample identity cannot condition the transferable response.
- Derived axes and neighbourhoods use the input offline constituent set, not
  true HLT jet axes or latent CMS-only event quantities.

Publish a field-by-field `compatibility_lock.json`, with unit evidence,
conversion formulas, validity rules, retained/excluded fields and populations,
pre/post-transform distributions, and unresolved items. Unresolved physical
definitions block a **fully qualified** shared-field response. A kinematics-only
exploratory study requires a separately named/versioned scope; it must not
silently become the requested complete response.

### 4.2 Fixed candidate predictors

Basic predictors: category, charge/applicability, `log(pT/GeV)`, `abs(eta)`,
`log(E/GeV)`, valid d0/dz and valid log uncertainties, plus explicit missing
indicators. Floors used for logs are numerical devices recorded in the input
lock, not imputed measurements. Continuous predictors are robust-scaled using
`fit_location` only.

Neighbourhood predictors, added only in families B/C:

- nearest-neighbour delta-R and nearest charged-neighbour delta-R;
- counts and summed pT within delta-R 0.02, 0.05, and 0.10, excluding self;
- charged fraction of local pT within 0.05, with an explicit empty flag;
- closest-neighbour pT ratio;
- particle pT fraction of the offline jet and distance from its own summed axis.

No jet mass, class score, truth flavour, generation channel, filename, or label
is a predictor. Source IDs are used only for splits, joins, and random keys.
Candidates are label-blind, not necessarily statistically independent of
flavour: measured particle properties can legitimately correlate with flavour.

## 5. Randomness and reproducibility contract

The generator is `synthetic_hlt = R_theta(offline, key, replica)`.
The user selected this reproducible statistical model over randomness-free
point prediction. The same offline input, response artifact, key and replica
must produce the same output on rerun.

Freeze a SHA256 counter-based random contract:

```text
key fields = domain "CMS2JC2_RESPONSE_RNG/v1", salt 20260917,
             canonical jet identity, response replica,
             component name, canonical object/group key, draw counter
h = uint64(first 8 digest bytes, big endian)
u = ((h >> 12) + 0.5) / 2^52
```

Use binary64 arithmetic. Keeping 52 bits makes the midpoint uniform strictly
between zero and one even after rounding; neither an inverse CDF at an endpoint
nor platform-dependent conversion of a full 64-bit integer is permitted.

Length-prefix every serialized field. Use canonical source/content/tree/entry
identity, not an absolute machine path. Do not include labels, model family,
candidate complexity, scheduler IDs, worker count, epoch, or mutable timestamps.
The response hash binds the output separately but does not alter the random
stream: comparable candidates get common random numbers.

Use separate components for topology, survival, identity, kinematics, tracking,
shared jet residual, and additional-object generation. Derive group keys from
sorted immutable input constituent IDs; generated child keys include the parent
group and child ordinal. Never consume an order-dependent global RNG.

Primary production replica is 0. Selection uses replicas 0, 1, and 2 for every
candidate; confirmation adds 3 and 4. Do not select a lucky replica. All future
methods in the same benchmark use replica 0 unless a separately registered
robustness study says otherwise. The user can regenerate additional replicas,
but these are response variations, not additional independent CMS observations.

Pin inverse-CDF/quantile interpolation, floating-point precision, categorical
ordering, and tie rules. Require byte equality across chunking/process counts
on the same supported numerical environment. Cross-platform bitwise equality
must be demonstrated, not assumed from deterministic keys; otherwise record a
numeric-tolerance portability result and restrict canonical production to the
validated SPORC environment.

## 6. Associations are calibration hypotheses, not detector truth

Do not reuse the ladder's forced full-cardinality match as reconstruction truth.
Jet rows are already paired, but constituent correspondences are inferred.
Allow losses, merging, splitting, ambiguity, and HLT objects without accepted
offline associations. Do not equate every unmatched object with inefficiency
or a fake detector particle.

### 6.1 Registered primary association procedure

Operate on compatible raw physical fields, not normalized classifier inputs.
Enumerate local hypotheses per paired jet:

1. one offline particle to one HLT particle;
2. two or three nearby offline particles to one HLT particle;
3. one offline particle to two nearby HLT particles;
4. unmatched offline and unmatched HLT particles.

Primary cross-view angular gate is delta-R 0.10. Within a merge/split group,
require all pairwise same-side distances <= 0.05. Candidate groups use at most
the six nearest neighbours of an object within that radius; this bounds
combinatorics. No pT-rank or constituent-count truncation of the jet is allowed.

For a group use the sum of raw p4. Require the target/source group pT ratio
to lie in [0.2, 5] for an accepted hypothesis. Minimize the disjoint set-cover
cost over all objects, with each object covered exactly once by a hypothesis
or a dustbin:

```text
c(group) = (deltaR(group axes)/0.10)^2
           + [log(pT_HLT / pT_offline)/log(2)]^2
           + 0.5 * (number_of_offline + number_of_HLT - 2)
c(unmatched object) = 1
```

Do not add an identity-disagreement penalty: PID changes are response targets.
Use deterministic exact component-wise optimization with canonical-ID
lexicographic tie breaking. Record optimum, alternative-cost margin and runtime.
Do not enumerate every whole-jet assignment. The grouped optimization can still
be combinatorial; decomposition is not a guarantee of cheap exact inference.
After profiling the fit-only 100k sample, freeze structural component limits
and a deterministic search-node budget in the association policy before science.
Use canonical search ordering. A component exceeding those fixed limits is
unresolved, not approximately labelled as an exact optimum. A wall-clock timeout
is an operational retry, never a reason for the same jet's scientific label to
change with machine load. Retain unresolved jets in collection-level evaluation.
Do not replace them with forced nearest-neighbour truth or discard them from
the population. An unproved second-best margin is recorded as unavailable,
not fabricated as zero or infinity.

This is a frozen operational association, not a proven physical reconstruction
history. Primary response fitting uses resolved hypotheses; report their
coverage and the characteristics of unresolved regions. Require >=99% resolved
jet coverage for response qualification, but failure to meet it is a scientific
limitation, not grounds to crash a completed diagnostic task.

### 6.2 Association sensitivity

Before response selection, evaluate gates 0.05/0.10/0.20, with within-group
radii 0.025/0.05/0.10 respectively, on the same 100k hash-selected `response_fit`
jets. Preserve all reports. This diagnostic cannot silently select a better
gate for one family. The primary remains 0.10.

After one finalist per family is chosen, refit those three FULL configurations
with each of the two alternative gates: **six mandatory sensitivity fits**.
Score on `response_select` only. They are robustness checks, not additional
winner candidates. A sensitive result is disclosed; any change to the primary
association requires a new campaign/version before confirmation is opened.

## 7. Shared modular response architecture

All families have the same output schema, topology representation, physical
constraints, random contract, and diagnostic access. Only predictor interactions
and fitted functional complexity differ. This avoids giving one family an
unrelated richer simulator.

### 7.1 Topology and neighbourhood ordering

Build local candidate groups from **offline-only** geometry. In generation,
visit candidate merges by increasing maximum pairwise delta-R, then decreasing
summed pT, then canonical IDs. Accept according to fitted probabilities and
the group's frozen uniform draw; accepted groups consume their constituents.
No overlapping merge groups are allowed. Consider singleton survival/splitting
after merges. The order is frozen and tested, not a knob tuned per candidate.

Fit topology probabilities from inferred CMS hypotheses using the same ordered
candidate exposure, so an object already consumed by a merge is not counted
again as an independent survival opportunity. Report order dependence as an
approximation. A separate diagnostic reverses tie-equivalent visitation order;
it does not redefine the primary generator.

Explicitly model 0, 1, or 2 singleton outputs. Larger splits and larger merges
are unresolved topology in v1 and must be counted. HLT objects assigned to the
unmatched side get a small conditional count/composition/relative-kinematics
module anchored to the offline jet and local density. Call these **unexplained
HLT components**, not truth-certified fakes. If their contribution is large or
poorly reproduced, v1 does not qualify; do not hide them by renormalization.

### 7.2 Kinematics

For each output object/group fit response in physical coordinates:
`log(pT_out/pT_in)`, delta-eta, wrapped delta-phi, and a nonnegative mass
response with an explicit zero-mass state. Additional components use pT fraction
and angular offsets relative to the offline jet. Split pT sharing is bounded
to [0,1] and uses a jointly generated group response.

Reconstruct p4 from generated pT, eta, phi and mass with positive energy.
Never independently perturb px, py, pz and E into an unphysical vector.
Merges sum input p4 before response; no rule forces observed jet energy to be
conserved through inefficiency or measurement changes.

### 7.3 Identity and tracking

Draw output category/charge from a calibrated categorical response, then apply
compatible tracking applicability. Fit displacement and log-positive-error
responses only in the correct measurement states. Fit missing/available
measurement transitions explicitly; neutrals do not acquire fictional tracks.
Unknown or invalid CMS input states are not converted into precise zeros.

Fit joint residual dependence for kinematics and valid tracking groups, rather
than smearing every feature independently. Estimate a shared per-jet component
from within-jet residual cross-covariances on `fit_residual`, accounting for the
number of objects. The intended decomposition is
`residual_jet,object = shared_jet + independent_object` in standardized latent
residual coordinates. Constrain both covariance components to be positive
semidefinite and their sum to match the calibrated marginal covariance. Do not
add an extra jet smear on top of an already full-variance particle smear.
Sparse cells back off as below; unavailable shared covariance is explicitly
zero, with insufficient-calibration status, not guessed from noise in jet means.

### 7.4 Common residual-distribution backend

Use held-out `fit_residual` predictions to construct monotone conditional
quantile tables at probabilities
`[.001,.005,.01,.025,.05,.1,.25,.5,.75,.9,.95,.975,.99,.995,.999]`.
Interpolate linearly between knots. Outside the outer knots, v1 clamps to the
outer quantile and explicitly reports the resulting tail limitation; extreme
tail coverage is an acceptance diagnostic, not silently claimed exact.

Retain categorical/missing-state mixtures separately. Within each valid state,
use a shrinkage rank-correlation Gaussian copula to couple marginal residuals,
with the shared/independent latent decomposition above. Apply each calibrated
marginal inverse CDF only after combining the latent components. Shrink
correlations toward identity using
`lambda = 1000 / (1000 + number_of_independent_calibration_jets)` in that cell.
Project to a positive-definite correlation matrix with recorded correction.
Tail dependence is not guaranteed by this backend; joint-tail checks are
mandatory and may show that every family needs a future richer backend.

Residual cells use category and coarse pT/eta strata; B/C additionally use
crowding terciles. Minimum supported cell is 1000 independent jets. Back off
deterministically by dropping crowding, then eta, then pT, then using category
alone. Never borrow another category's incompatible tracking semantics.
Export fitted tables and correlations, not libraries of raw CMS particles.

### 7.5 Edge conditions

- Empty generated jets are possible inefficiency outcomes. Preserve and count
  them; do not keep the hardest particle merely to ensure classifier input.
- Output-count overflow is reported, never silently truncated. Downstream
  padding/capacity is chosen from the new proxy, not inherited as 200 or 240.
- Sorting is stable decreasing output pT with canonical generation key ties.
- Internal lineage can record retained/merged/split/generated provenance for
  audits, but none of it is a deployable model feature.
- Unphysical/nonfinite arithmetic is an execution error. Unusual but finite
  physically allowed jets are scientific outputs, not quality-filtered away.

## 8. Exact three-family, nine-candidate registry

Topology probabilities use calibrated categorical/logistic losses. Continuous
location models use weighted squared error on registered transformed response
coordinates; conditional scale models fit log absolute residual with the
training-locked numerical floor. Final residual quantiles restore non-Gaussian
marginal shapes. Every family uses the same record weights and residual backend.

| ID | Family | Fixed complexity |
| --- | --- | --- |
| A_L | Basic tables | 6 pT x 3 abs-eta bins per category; shrinkage pseudocount 2000 |
| A_M | Basic tables | 12 pT x 6 abs-eta bins; pseudocount 1000 |
| A_H | Basic tables | 24 pT x 8 abs-eta bins; pseudocount 500 |
| B_L | Neighbourhood smooth additive | Cubic splines, 4 interior knots, ridge penalty 10 |
| B_M | Neighbourhood smooth additive | Cubic splines, 6 interior knots, ridge penalty 1 |
| B_H | Neighbourhood smooth additive | Cubic splines, 8 interior knots, ridge penalty 0.1 |
| C_L | Neighbourhood boosted trees | Depth 2, 200 trees, minimum 5000 weighted-effective records per leaf |
| C_M | Neighbourhood boosted trees | Depth 3, 400 trees, minimum 2000 weighted-effective records per leaf |
| C_H | Neighbourhood boosted trees | Depth 4, 600 trees, minimum 1000 weighted-effective records per leaf |

A bins use fit-only quantiles, with sparse-bin shrinkage toward category-level
estimates. Missing flags and output-state conditioning remain explicit. Valid
tracking responses also use category-level smooth linear dependence on their
corresponding input displacement/error; A is not allowed to erase those inputs.

B includes univariate smooth terms and only these two-way tensor interactions:
log-pT x abs-eta, log-pT x local-pT-within-0.05, nearest-distance x neighbour-pT
ratio, and tracking value x its valid uncertainty. Discrete categories interact
through category-specific intercepts and partial pooling. Penalize every
non-intercept coefficient; use the same fit-only scaling for all B settings.

C uses learning rate .05, L2 penalty 1, all registered predictors, and no
automatic validation split/early stopping. All listed trees are fitted. If the
chosen library's leaf parameter counts raw records rather than effective
weighted records, enforce the effective-count rule separately and document the
mapping. Freeze library/version, thread counts and fit seed 20260917.

Probability calibration uses a single fit-residual temperature per categorical
module, minimizing weighted log loss on that internal role. Do not use CMS
selection/confirmation labels or JetClass2 observations for recalibration.

Initial topology models use the permitted family predictors of their object
groups. A uses group pT/eta/category composition without distance predictors;
the shared topology proposal already imposes geometry. B/C additionally learn
the dependence within that proposal's permitted neighbourhood.

There is no hyperparameter search beyond this registry in v1. Adding new
interactions, tail laws, or candidates after viewing results requires a logged
new version; confirmation data cannot be recycled as ordinary validation.

## 9. Selection metrics, uncertainty, and qualification

### 9.1 Evaluation population and random replicas

Evaluate every one of the 27 primary fits on every `response_select` jet with
replicas 0/1/2. The same native CMS HLT is the comparator each time. Average
metrics across replicas but do not triple the apparent number of real jets.
Cluster uncertainty by source file, with paired resampling across candidates.
Use 200 deterministic bootstrap replicates, seed 20260918. Publish the number
of independent source groups and warn about weak coverage; a large jet count
does not make a three-file sample equivalent to hundreds of files.

All raw-feature scalar distributions use fit-only frozen transforms/scales.
Continuous values use `asinh(x/s)` for signed tracking quantities and logs for
positive scale quantities, with s fitted on `fit_location`. Use exact empirical
CDFs or shared fixed-accuracy mergeable summaries; freeze their implementation
and error bounds. Do not compare unrelated adaptive histogram binning.

### 9.2 Primary score

Define six blocks, each weighted 1/6:

1. Jet and category multiplicity, survival/merge/split/unexplained-component rates.
2. Particle pT/energy/direction response and leading/subleading fractions.
3. PID/charge and measurement-availability transition frequencies.
4. Valid displacement/error distributions and tracking correlations.
5. Jet pT/mass/axis response, constituent radial profile, and energy correlators.
6. Joint dependence: particle response vectors and jet summary vectors.

The metric registry must expand this into the following fixed core observables,
not choose interesting-looking plots after fitting:

- Counts: total and each exclusive output-category multiplicity, and the
  association-derived loss, merge, split and unexplained-component rates.
- Particle response: log-pT ratio, log-energy ratio, delta-eta, wrapped
  delta-phi, and leading/subleading output pT fractions. Zero/undefined states
  are separate categorical probabilities, not epsilon-imputed observations.
- Identity: input-to-output category/charge transition matrices and per-field
  tracking applicability/availability transition matrices.
- Tracking: applicable d0, dz, log-positive uncertainties, valid d0/error and
  dz/error, and their registered pairwise Spearman correlations.
- Jet response: multiplicity, log summed-pT ratio, mass distribution and mass
  response where defined, axis delta-R, charged pT fraction, and pT fractions
  in radial annuli [0,.02), [.02,.05), [.05,.10), [.10,.20), [.20,.40),
  [.40,.80), and [.80,infinity), about the jet's own summed-p4 axis.
  Include `width = sum_i z_i * deltaR(i,axis)` and
  `e2_beta = sum_{i<j} z_i*z_j*deltaR(i,j)^beta` for beta=1 and 2, where
  `z_i = pT_i/sum_j(pT_j)`. These are exact all-constituent sums; no cubic
  three-point correlator is required in v1.
- Joint vectors: particle (log-pT response, delta-eta, delta-phi, applicable d0,
  dz, log errors) within compatible measurement states; and jet (multiplicity,
  log summed pT, mass, charged fraction, width, e2_1, e2_2). Include all
  pairwise correlations of the jet vector. Empty/axis-undefined cases contribute
  explicit state probabilities; conditional continuous metrics report coverage.

Additional plots do not change these primary weights. Field units, log/zero
states and conditional-grid edges must be resolved in the compatibility and
metric locks before fitting, not filled in after examining candidate results.

For a continuous comparison, use Wasserstein-1 divided by the corresponding
real-CMS fit-role IQR. If IQR is zero, use the fit-role 95th--5th percentile
range; if that is also zero, treat equality as zero discrepancy and otherwise
report a point-mass mismatch rate. Categorical discrepancies use total
variation. Correlation discrepancy is absolute difference in Spearman rho.
Joint vectors use mean sliced Wasserstein across 64 fixed unit projections
generated with seed 20260919 after fit-only coordinate scaling.

Within a block, average equally over its registered observable definitions,
then over eligible conditional strata, not over whichever measurements happen
to look good. Conditional strata use particle category, fit-role pT/abs-eta
quartiles, and crowding terciles, separately rather than a huge Cartesian grid.
Require 1000 real jets per reported selection cell; pool sparse cells by the
predefined backoff, mark absent cells, and disclose coverage. Do not silently
drop an entire difficult observable. Before science, a metric-registry artifact
must enumerate all observables, projections, transforms, strata and weights;
the six-block list alone is not an executable metric contract.

Pair-based response summaries use resolved associations and carry a coverage
denominator. Unconditional collection/jet comparisons include **all** jets,
including unresolved associations and generated empty jets. This prevents a
good fit to convenient matched particles from hiding a bad whole-jet response.

### 9.3 Selection rule and robustness

Find the lowest mean primary score among the nine FULL fits. Form the
one-standard-error set: candidates whose score is no greater than the best
score plus that best candidate's source-block bootstrap standard error.
Choose the first in the registered simplicity order
`A_L,A_M,A_H,B_L,B_M,B_H,C_L,C_M,C_H` that passes the qualification rules below.
If none qualifies, report an exploratory best scorer but publish **no qualified
response**. Report the best candidate within each family independently.

The six association-sensitivity refits must finish and be disclosed before
the selection lock. If the selected candidate's primary score changes by more
than 20% under either gate, or a core qualification decision changes, mark
association robustness unresolved. This does not crash the comparison; it
blocks claiming a robust qualified calibration without an explicit follow-up.

### 9.4 Predeclared engineering qualification targets

These are initial, explicit approximation tolerances, not CMS-certified
detector-performance requirements:

- At least 99% resolved-jet association coverage on calibration and selection.
- Mean jet multiplicity, pT and mass biases <=5% of the real reference mean.
- Each of the six normalized discrepancy blocks <=0.10 overall.
- No eligible conditional cell has discrepancy >0.25.
- PID/measurement-availability total-variation discrepancy <=0.05 overall.
- Pairwise registered correlation discrepancies <=0.10.

Class-specific discrepancies are mandatory non-selecting disclosures, not an
undefined subjective veto or a reason to fit separate responses by class.
Physical invalidity remains a contract error regardless of class. Any proposed
new class-closure qualification threshold requires a new pre-confirmation
protocol version.

Publish point estimates and uncertainty intervals. Flag intervals crossing a
threshold as inconclusive rather than certainty. Threshold failure is a
scientific outcome: finish the graph, retain all rows, and report unqualified.
Thresholds may be amended only through an explicit new pre-confirmation plan
version, never secretly relaxed after the confirmation result is known.

### 9.5 Non-selecting diagnostics

Report class-conditioned closure, tail exceedances, quantiles .001/.01/.5/.99/.999,
rare-category counts, bin occupancy, and clipping/backoff rates. Labels may
stratify these diagnostics; they never condition the response.

Train a fixed shallow two-sample discriminator using response-select files
split into file-disjoint training/evaluation halves. Predict real versus proxy
HLT from the common physical observables, with equal domain weights and paired
jets kept together. Report its held-out AUC as a sensitivity diagnostic, not as
the single optimization objective or proof that matching marginal histograms
establishes realism. If too few files support the split, report unavailable.

Classification AUC, R50 and KD gains are **not selection targets**. After the
response is frozen, a separate approved common-feature CMS classifier check
can compare actual-HLT and proxy-HLT performance and KD behaviour. That is a
GPU study with new acceptance, not an implicit extension of this CPU campaign.

## 10. Locked CMS confirmation and JetClass2 transfer

### 10.1 Confirmation

Publish source, candidate, fitted-parameter, metric-registry and selection
hashes before any `response_confirm` particle access. Require an explicit
confirmation execution claim. Evaluate the selected response and the two other
family finalists, with replicas 0--4, on all confirmation jets.

Do not switch winners on confirmation or refit using confirmation data.
The additional finalists expose the family comparison, not a second selection
round. If no candidate qualified on selection, confirmation can still be
separately authorized as a diagnostic, clearly marked nonqualifying.
Historical CMS final-test files remain sealed throughout.

### 10.2 JetClass2 transfer scale

After freezing the response, generate/evaluate on the existing 500k training
jets and 1M validation jets, using offline-only branch reads. Selection never
uses these results. The 1M/1.5M/2M existing training profiles can be transformed
later with the same frozen response and memberships; those larger downstream
campaigns are not automatically submitted here.

For every conditioning variable report how often JetClass2 falls outside CMS
fit-role 0.1--99.9 percentile support, outside observed category/state support,
and into sparse neighbourhood combinations. Estimate nearest-support distances
on standardized allowed predictors, not labels.

Default extrapolation policy: clamp predictor coordinates to calibrated support
for evaluating response parameters, not the actual physical input p4; report
every clamp and hierarchical fallback. Never silently discard such jets or
claim validated response there. If >5% of jets require any support clamp or
>1% require an unseen physical-state fallback, transfer qualification is
`out_of_domain`; generation still completes and reports the result. Those
thresholds are engineering flags, not a guarantee below them.

JetClass2's native HLT is not used as a target, discriminator domain, matching
partner, or visual-validation reference. There is no paired CMS HLT truth for
a JetClass2 jet. Transfer diagnostics establish support and plausibility, not
detector truth. The response remains provisional outside validated CMS support.

### 10.3 Visual and trace-based audit

Register 200 hash-selected CMS selection examples, 200 CMS confirmation
examples, and 200 JetClass2 training examples. Stratify display coverage by
offline pT, multiplicity, particle composition and crowding. Include an
additional labelled worst-50 selection-error panel to expose failure modes;
never present it as a random sample.

For CMS, draw offline, real HLT, and proxy HLT in the same eta/phi coordinates,
with pT-scaled markers and category colours. Show retained/lost/grouped/generated
operations, response factors, tracking states, random-key identifiers and
aggregate differences. For JetClass2, show offline and proxy only, plus support
flags. Keep full trace payloads bounded to these examples. Every figure must
name its sample-selection rule, response hash and replica.

No tuning to visually pleasing selected examples. A discovered implementation
bug requires new source-bound artifacts and appropriate re-evaluation; a
scientific deficiency is documented even if inconvenient.

## 11. SPORC execution, resources, and storage

### 11.1 Site contract

Use `sporcsubmit.rc.rit.edu` for submission, **not heavy computation**. Per the
2026-09-17 user routing change, every stage uses SPORC `debug`, account
`reu-aisocial`, QoS `qos_tier3` (the QoS name remains unchanged): preparation,
acceptance, science, confirmation, transfer, reporting, and recovery.
This is an operational partition change only; scientific settings and resource
envelopes are unchanged. Recreate source-pinned specifications and dry plans
with the updated source; never edit an existing immutable spec or live job.
All requests remain CPU-only, one node/task per job, no GPU GRES. This derives account/site
intent from the current SPORC work, but **CPU-only eligibility, partition limits,
account access and filesystem visibility must be queried and authenticated**
before freezing the executable site object. Do not reuse a GPU-mandatory site
validator or assume that removing its GPU request creates a valid CPU contract.

Use the recorded `atlas_kd_sporc` environment read-only if its packages support
the worker. If new dependencies are required, create a separately approved
environment; never upgrade an environment used by active jobs. Every worker
sets `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, unbuffered output, and
prepends `${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`. Source helpers by absolute
`${PROJECT_DIR}`, pin a clean pushed commit, and record numerical-library bytes
and versions.

### 11.2 Proposed envelopes, to be replaced by measurements

| Task | CPUs | RAM | Initial walltime |
| --- | ---: | ---: | ---: |
| Inventory/schema/role audit | 4 | 16 GiB | 2 h |
| Association/summary shard | 8 | 32 GiB | 6 h |
| Table/smooth response fit | 16 | 128 GiB | 24 h |
| Tree response fit | 16 | 128 GiB | 24 h |
| Evaluation/visual shard | 8 | 32 GiB | 8 h |
| Aggregate/lock/report | 2 | 8 GiB | 2 h |

These are planning envelopes, not measured needs or a live command. A real
CPU production-worker miniature on 20k fit jets must exercise every module,
all three families, deterministic parallel replay, and a representative large
jet. A further 100k-jet throughput/RAM probe sets full-size predictions.
Select 1.5x measured RAM and 2x projected walltime within site limits. If a
stage exceeds its envelope, shard it or explicitly revise the resource plan;
never silently reduce scientific data, candidates, or diagnostics.

Initial global limits: at most 4 association/evaluation shards, at most 3 fit
jobs, and at most 64 allocated CPU cores for this campaign at once. The
submitter must enforce the combined cap across task kinds, not just separate
array throttles. Emit total estimated CPU-hours before live authorization.
No unbounded autonomous sweep or hidden budget expansion.

Use spawned processes for ROOT/association work. During threaded numerical
fitting, do not also launch one multithreaded fit per process. Pin threadpool
limits so total runnable workers do not exceed the allocation. Profile actual
active cores/RSS and report phase-level progress and rows processed.

### 11.3 Storage contract

Raw data are read-only. Do not duplicate ROOT, save full synthetic datasets,
persist dense particle views, save every replica, or write rolling optimizer
states. Working features, candidate responses and residual samples live in RAM
or explicitly approved node-local scratch that is accounted and cleaned by the
same task. Node-local scratch is not assumed to exist and is not a hidden
fallback to shared home storage.

Persistent artifacts are compact memberships/associations where justified,
streamed sufficient statistics, fitted model parameters, residual tables,
small reports, and bounded example plots. No raw CMS donor-particle library is
exported as the response. Estimate every durable payload before creating it.

Initial shared-storage hard planning cap is **12 GiB** for the whole campaign,
including temporary/failed attempts; at most 2 GiB is reports/figures/examples.
Check for at least twice estimated remaining writes plus 5 GiB free space.
If all-row association maps would breach the cap, recompute them in bounded
workers or retain only compact module summaries, with the choice recorded in
the execution plan. Do not discard scientific rows to fit a disk quota.
If fitted trees themselves exceed the cap, report a resource blocker and ask
for a new budget rather than reduce trees/data without disclosure.

Outputs go in a fresh root such as
`checkpoints/cms2jc2_response_<commit8>_r1`; source worktrees use a distinct
`HLT_Classification_cms2jc2_response_<commit8>` path. No output is under either
raw dataset. Recover by authenticated completed task/shard, not by filename
existence. Restart incomplete fits from zero; preserve completed outputs.

## 12. Staged task graph and authorization

1. `inventory_authenticate`, `site_probe`, `compatibility_audit`.
2. `response_split` and `compatibility_lock` after human review of genuine
   unresolved producer semantics; these are not guessed by the submitter.
3. `cpu_miniature`, `resource_profile`, `execution_lock`.
4. Fit-role association/summary arrays and shared preprocessing lock.
5. 27 primary fits, each followed by selection evaluation shards and reduction.
6. `family_finalists`, six association-sensitivity fits and their selection
   reports, then `selection_lock`.
7. Separately authorized confirmation arrays/reports for three family finalists.
8. JetClass2 offline-only transfer audit for the selected response, visual audit,
   qualification report, aggregate and campaign completion.

Preparation, primary science, and confirmation/transfer have separate canonical
dry ledgers and explicit live authorization phrases. No task submits an
unreviewed new candidate. Dry plans show exact resources, source, input hashes,
dependencies, output roots, and expected task/array counts.

Use `afterok` dependencies only for valid required artifacts, not for favourable
physics scores. A completed unqualified candidate is still an `afterok` parent.
Final completion reports `qualified`, `unqualified`, or `inconclusive`
separately from operational completion. Confirmation authorization can be
withheld while ordinary comparison and its completion report remain valid.

Use journaled pre-submit intents and exact `sbatch --parsable` receipts.
Ambiguous acknowledgements require reconciliation, not blind resubmission.
No broad `scancel`, holding, reprioritizing, partition switching, or edits to
active worktrees. Requeue/recovery acts only on explicitly approved exact IDs.
This plan does not grant remote credentials or ongoing access by itself.

## 13. Required implementation surface and contracts

Proposed new namespace: `src/hlt_classification/cms2jc2_response/`.
Keep calibration semantics separate from the old ladder's matching/view code.

| Module | Responsibility |
| --- | --- |
| `contracts.py`, `provenance.py` | Versioned artifacts, access roles, source/data identities |
| `bridge.py`, `readers.py` | Shared raw fields, CMS paired and JetClass2 offline-only reads |
| `splits.py` | Nested budgets, grouped fitting/selection/confirmation roles |
| `association.py` | Partial/group association, ambiguity and sensitivity policies |
| `features.py`, `rng.py` | Label-blind predictors and keyed statistical response |
| `topology.py`, `response.py` | Modular set transformation and physical constraints |
| `families.py`, `residuals.py` | Nine exact candidates and shared calibrated residual backend |
| `metrics.py`, `selection.py` | Closure, paired uncertainty, simplicity rule and qualification |
| `audit.py`, `plots.py` | Field, transfer-support and representative-example reports |
| `campaign.py`, `worker.py`, `submission.py` | Staged immutable graph, resource gates and exact recovery |

Thin proposed CLI: `scripts/cms2jc2_response.py`, with `audit`, `create`,
`dry-run`, `submit`, `run-task`, `monitor`, `results`, `inspect-examples`, and
`confirm` commands. Proposed worker: `sbatch/run_cms2jc2_response_cpu.sh`.
These paths are design targets, not claims that executables already exist.

Use new `CMS2JC2_RESPONSE_*/v1` contracts for source inventory, compatibility,
role memberships, association policy, RNG, predictors, candidate registry,
calibration data, fitted response, metric registry, evaluation, selection,
confirmation claim/report, transfer, examples, site/resources, campaign,
command plan, submission receipt, output inventory and completion.
Every reusable artifact binds content hash, parents, semantic source hashes,
source commit, schema and payload checksums, with atomic publication.

Likely reusable donors: `data/cache_contracts.py` (hash/publication),
`scouting/schema.py`, `scouting/splits.py` and native readers (CMS definitions),
`jetclass2_delphes/schema.py`, `reader.py`, `split_registry.py` (new-data
identities), and existing bounded-process/submission-journal patterns.
Existing `data/hlt_v3.py` may inform tests for physical invariants and keyed
response, but its hand-chosen degradation parameters are **not** CMS calibration.
Record actual donor paths and their exact commits in `docs/LEGACY_SOURCE_MAP.md`
when code is migrated. No donor code is migrated by this documentation change.

## 14. Testing and implementation acceptance

Follow [the testing ladder](../TESTING.md) with these focused additions:

- Shared-field unit/sign/validity fixtures, unknown categories, neutral tracking,
  lost-track exclusion and raw-versus-normalized interface rejection.
- JetClass2 output invariance to missing/modified HLT branches and labels.
- Group-disjoint splits, nested memberships, minima failures, no sealed-role reads.
- Fixed-key replay across ordering/chunks/workers and no label/candidate/epoch
  dependence; common-random-number and independent-component tests.
- Hand-calculated identity/no-effect fixtures and survival-frequency checks,
  merge/split non-overlap, dustbins, alternative optima, large-component handling.
- p4 physicality, valid categorical states, empty outputs, count overflow,
  correlations, tails and missing-state preservation.
- Synthetic known-response recovery for all three families, including an
  intentionally misspecified response that completes but fails qualification.
- Fit-residual leakage rejection, weighted reservoir correctness, sparse-cell
  backoff, tree-complexity bounds and monotone quantile tables.
- Metric zero-IQR cases, common populations, bootstrap clustering, replica
  dependence, candidate ties, one-SE selection and confirmation-lock enforcement.
- Source/artifact corruption, interruption, duplicate writers, out-of-domain
  transfer, CPU oversubscription, storage exhaustion and partial submission.
- End-to-end synthetic ROOT -> bridge -> response fitting -> held-out evaluation
  -> selection -> separately claimed confirmation -> offline-only transfer.
- Real SPORC CPU miniature/resource probe using production workers, not mocks.

Readiness requires tests, compatibility/split locks, a measured resource plan,
exact pushed source, and a fully reviewed dry run. Local tests cannot establish
remote dataset availability, scheduler permissions or scientific closure.

## 15. Deliverables, stop conditions, and next step

Deliver:

1. Verified shared-field and population report with explicit physical assumptions.
2. Large, disjoint CMS role manifests and exact learning-curve memberships.
3. Complete 27-fit comparison and six association-sensitivity fits.
4. Family finalists, locked selection, and separately authorized confirmation.
5. Portable response artifact, deterministic generation interface and provenance.
6. JetClass2 transfer-support report and representative/worst-case visual panels.
7. Honest operational completion and independent qualification status.

Except for the explicitly authorized provisional implementation in section 1,
pause for explicit resolution if common-field meanings cannot be established,
large-data minima cannot be met, source files are inaccessible, compute/storage
would exceed the agreed bounds, or release of CMS-derived parameters needs
approval. Do not substitute invented physics, more permissive data roles, or
undocumented infrastructure. Poor closure itself is not an execution failure;
it can establish that v1 is insufficient.

Export/publication of CMS-derived calibration tables or example jets requires
the appropriate collaboration/advisor permission. Public JetClass2 inputs do
not automatically make all CMS-derived outputs publicly distributable.

**Implementation checkpoint (2026-09-17, staged tooling):** local CMS authentication
and capacity allocation succeeded: 2,222,819 calibration, 277,546 selection and
277,490 confirmation jets. Source-bound preparation, bridge/split contracts,
association, complete set generation, bounded weighted records, six-block metric
and paired-selection engines, transfer-support diagnostics, storage guards and
the 71-task scientific graph now have focused development tests. Small real fit-role
checks exercise all three low-complexity families under the named provisional
contract; they are not held-out closure or production acceptance. Implementation
details and current boundaries are recorded in
`docs/contracts/CMS2JC2_RESPONSE_PREPARATION.md`. Staged creation/submission,
durable intents/claims/receipts, monitoring, terminal-only restart-zero recovery,
independent confirmation, transfer, tails/occupancy/two-sample diagnostics and
bounded physical example plots are implemented. Real CPU acceptance remains
unperformed. No new full HLT dataset or science sweep has been launched.
Keep the provisional/verified distinction when the producer reply arrives.

### 15.1 Frozen executable details

These details refine the implementation without reducing populations or fits:

- Preparation remains a separate scalar/metadata task. Acceptance has 15 tasks:
  three genuine 20k low-complexity fits, common serial/process and module checks,
  three 100k high-complexity fits, common fit-only metric calibration, three
  100k three-replica closure profiles, three 100k association-gate profiles,
  and a measured execution lock. Gate profiles follow closure profiles so no
  more than four association/evaluation tasks overlap. Probe rows use per-file
  bottom canonical-identity hashes, not a first-row prefix.
- The 71 science tasks include all 27 primary and six sensitivity fits, all 33
  closure evaluations, metric/finalist/selection locks, selection examples and
  completion. Two 16-CPU fit lanes and four evaluation lanes remain under 64
  CPUs. Current serial closure/metric/visual workers request **one CPU**, not
  eight idle CPUs; the graph's eight-CPU evaluation allowance is an upper bound.
- Nine separately authorized confirmation/transfer tasks freeze the selected
  response plus the best FULL response from each other family. Confirmation
  cannot choose a new winner. Claims precede held-out/transfer reads; no task
  automatically submits the next stage.
- Runtime projects high-complexity 100k profiles to full declared populations
  with 2x margin. RAM uses 1.5x sampled process-tree RSS, conservatively scaling
  fit-record growth to the bounded nine-module reservoirs. Closure/transfer
  estimates use the slowest high-family closure profile (confirmation 5/3
  replicas). Exceeding the envelopes produces an explicit resource blocker,
  never a silently smaller study. Predictions are not runtime guarantees.
- Scalar quantiles use 16,384 bottom-hash records per observable/side; tail
  exceedances and histogram occupancy use every observation. Joint-tail counts
  report any, at least two, and all coordinates beyond fit-only 0.1/99.9%
  thresholds. No extra diagnostic changes primary score weights.
- The non-selecting two-sample check uses a depth-3, at-most-eight-leaf tree,
  minimum 100 retained records/leaf, seed 20260917; equal real/proxy weights
  with inverse file sampling fractions and at most 4096 paired jets/file.
  Hashed file halves need at least two independent files each. With the current
  three selection files this diagnostic is explicitly unavailable, not replaced
  by a leaking within-file split or counted as a successful test.
- 200 examples per selection/confirmation/JC2-train role use sixteen fixed
  binary offline strata: scalar pT 500 GeV, count 50, charged-pT fraction 0.5,
  median R=0.05 crowding count 3. Empty strata fill from a global bottom-hash
  reservoir. Fifty additional worst-selection examples rank mean relative
  count/scalar-pT/mass error, denominators max(1, observed). No other real
  particle arrays persist. SVG panels share coordinates and include JSON traces.
- Reverse-tie-order sensitivity uses only the 200 fixed selection examples,
  replica zero, the frozen response and identical random keys. Reverse only
  canonical-key ordering of merge proposals with exactly tied diameter and
  summed pT; do not alter the proposal set, retrain, or change primary outputs.
- Numerical environment evidence hashes installed Python/native files and
  loaded BLAS/OpenMP libraries. Same-source terminal recovery authenticates
  payload bytes before reuse; source changes require fresh acceptance. No
  running/pending job is canceled, held, or modified by this tooling.

## 16. Method references

- [Fast simulation of detector effects in Rivet](https://arxiv.org/abs/1910.01637):
  precedent for efficiency and kinematic-smearing detector-response emulation;
  not validation of this particular CMS-to-JetClass2 transfer.
- [Delphes module reference](https://delphes.github.io/workbook/modules/):
  modular efficiency, identity and resolution concepts, not a source of fitted
  constants for this study.
- [Histogram gradient boosting documentation](https://scikit-learn.org/stable/auto_examples/ensemble/plot_hgbt_regression.html):
  conventional bounded tree and quantile-response implementation options.

The particular calibration, splitting, association, candidate and qualification
rules above are this study's proposed protocol, not results established by
those references.

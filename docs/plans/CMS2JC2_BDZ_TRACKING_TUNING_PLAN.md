# B_DZ tracking calibration: one bounded development screen

## Authority and purpose

This plan implements Ryan's 2026-09-25 request to tune B_DZ after inspecting
the completed bounded comparison. It authorizes a **new**, disjoint development
study, not an amendment to any published model, selection, or running campaign.
The earlier bounded comparison remains an exploratory result. Its 20,000-jet
confirmation is not consumed by this study. There is no automatic confirmation,
JetClass2 application, classifier training, or production qualification.

B_DZ improved histogram discrepancy but still underrepresented displacement
tails and misrepresented uncertainty widths. A standard deviation alone cannot
tell whether rare CMS observations are physical or pathological. Do not delete
those observations, infer that a flag rate is a particle rate, or scale everything
to match the standard deviation. Jet mass, width, charged momentum fraction,
association coverage and source convention questions remain separate limitations.

## Frozen inputs and scope

Import a completed `bounded_compare` stage whose authenticated selection is
B_DZ. Authenticate its model registry, all four shards, selection receipt,
preparation, numerical environment and scientific source. Reconcile the known
JSON dictionary-order change in guard-reason lists by sorting the lists; require
the same candidate keys, reason multiplicities, winner, model and parents.
This is an ordering-only reader, not permission to change a result.

Keep the original B fit, selected dz residual scale, association policy, source
splits, units, feature interface and keyed three-replica randomness fixed. The
generator still consumes offline-only shared features, never labels or real HLT
at generation time. Real CMS HLT is used only as a calibration/evaluation target.
Particle keys, order, p4, PID, charge and all four validity masks must remain
bitwise unchanged for every candidate on every jet. Invalid tracking placeholders
remain zero; valid uncertainties remain positive.

## Exactly three candidates

1. `B_DZ`: exact historical control.
2. `TRACK_HALF`: half-strength calibrated tracking response.
3. `TRACK_FULL`: full-strength calibrated tracking response.

Fit one set of monotone marginal quantile transports for d0, dz, d0err and dzerr
using the existing 4,000 training-residual calibration jets. These jets already
influenced the B residual calibration: this is additional calibration, not an
independent validation sample. Pool the three generated replicas; count unique
jets, not replicas, for support. Real targets enter once per jet. Fit each
generated PID category separately only with at least 1,000 contributing jets
on BOTH sides for that coordinate; otherwise use the pooled map. An unsupported
or degenerate pooled map is identity. Publish the complete support/fallback table.

Use signed `asinh(value / 1 mm)` for d0/dz and `log(error / 1 mm)` for errors.
The fixed quantile probabilities are 0, .0001, .001, .005, .01, .025, .05, .1,
.25, .5, .75, .9, .95, .975, .99, .995, .999, .9999, 1. Collapse repeated source
knots by averaging the corresponding target knots. Interpolate linearly in
transformed space, extending constantly beyond the fitted endpoints. Blend the
mapped and original transformed values with strength .5 or 1, then invert.
There is no random jitter or invented within-atom variation. Exact valid zero
displacements remain zero and are disclosed as a limitation of this map.
This cannot recover missing event-level physics or guarantee cross-domain transfer.

No raw generated-particle files are published. Calibration arrays are temporary
RAM only, with a 2 GiB raw-array budget that fails rather than silently subsamples.
Persist quantile knots, support counts, lineage, and aggregate diagnostics only.

## Evaluation and frozen choice

Replay all existing 10,000 development jets in four fixed 2,500-jet shards,
including unresolved-association jets. Never select jets by class or quality.
Require historical B_DZ histogram, identity and generator-flag replay. The same
jets have informed earlier decisions; all results are explicitly exploratory.

Retain the original histogram ranges and conditional cohorts (count, scalar pT,
crowding and offline displacement). Publish their full moments, TV, underflow,
overflow and plots. Add exact tracking exceedance counts for absolute d0/dz at
.1, 1, 10, 100 mm; errors at .01, .1, 1, 10 mm; absolute significances at
1, 3, 10, 30, 100. These are diagnostic thresholds, not observation cuts.
Record six-coordinate transformed tracking correlations on particles where all
four fields are valid, with the complete-case count disclosed. Correlations use
asinh(d0), asinh(dz), log(d0err), log(dzerr), asinh(d0/d0err), asinh(dz/dzerr).

The selection score equally averages: (a) the existing conditional tracking TV
block; (b) a tail-balanced discrepancy; (c) tracking correlation discrepancy.
For each tail threshold use |p_proxy-p_real| / (p_proxy+p_real+1/n_real), average
thresholds per observable, then observables and replicas. Both missing = 0;
only one missing = 1, always disclosed. Correlation discrepancy averages absolute
Pearson differences divided by two across the 15 pairs and three replicas;
both undefined = 0, one undefined = 1, with missingness disclosed.

Guard against >.02 absolute regression versus B_DZ in any individual conditional
tracking-variable TV, individual tail-observable score, or the joint tracking
correlation block. Unchanged nontracking distributions are an exact invariant,
not a soft performance guard. Select the lowest-scoring eligible candidate,
breaking ties B_DZ, TRACK_HALF, TRACK_FULL. Poor candidates still finish and
publish; retaining B_DZ is valid. Publish source-file leave-one-out score gains
and the file count, not a claim of independent significance. Replicas are not
independent jets. No confidence or qualification claim follows from three files.

## Execution and stop rule

CPU-only SPORC debug, account reu-aisocial, qos_tier3, no GPU. Two explicitly
authorized stages, each with dry-run review and pinned pushed clean source:

- `bdz_gate`: 32-real-jet serial/two-process replay and resource acceptance
  (2 CPUs, 32 GiB, 2 h), followed by calibration (36 CPUs, 128 GiB, 8 h).
- `bdz_compare`: four evaluations (36 CPUs, 128 GiB, 8 h each), followed by
  selection/report (1 CPU, 32 GiB, 4 h).

Use spawn processes, one numerical thread per process, eight-jet chunks, at most
twice the worker count outstanding, deterministic merge order and progress every
15 seconds. The acceptance run exercises all candidates using a temporary tiny
map; it is not the published calibration or a quality gate. Project resources
conservatively before permitting calibration. Reuse the existing immutable
submission/receipt/claim machinery. Do not cancel, update or resubmit old jobs.

Stop after this screen and inspect the frozen report. If no candidate helps,
report that result rather than silently launching a larger search. Any independent
confirmation requires a separately reviewed plan and explicit authorization.

# Bounded B / dz-recalibrated B / B–C response comparison v1

## Decision and authority

User approval, 2026-09-25: finish a finite comparison instead of indefinitely
tuning two response families. This is an **additional development protocol**,
not the original 27-fit production campaign or a relaxation of its qualification
limits. The original three-family plan and CPU-development contracts continue
to govern their existing artifacts. This amendment alone authorizes a new,
separately claimed, bounded confirmation reader; the old development reader
still cannot read `response_select` or `response_confirm`.

No old artifacts, source worktrees, scheduler jobs, or scientific results change.
No JetClass2 particles, classifier validation/test data, labels, or truth enter
this study. All CMS files remain within the authenticated historical training
inventory. All inference uses offline shared-interface fields and the existing
label-independent keyed random stream. Physical conventions remain provisional.

## Exactly three candidates

1. `B`: the completed original B_L response, unchanged, rerun as a parity control.
2. `B_DZ`: the same B_L with one global nonnegative multiplier on the **dz
   residual quantiles only**, for every output of every continuous module.
   The mean predictor, scale predictor, correlations/random keys, d0, errors,
   momentum, topology, identity, charge, validity, limits and fallback policies
   remain unchanged. This is a narrowly defined residual recalibration, not a
   claim to fix a sign convention or all tracking deficiencies.
3. `BC`: all B categorical modules (merge/survival/split/additional counts and
   output identity/charge/validity/mass state), with C's **complete joint
   continuous modules**: mean, scale, conditional residual backend and limits.
   C is conditioned on the actual B-generated state and offline group, never an
   observed HLT state. p4/tracking/errors are sampled jointly; separately
   generated particle lists are not joined. Missing C states retain the existing
   flagged central/fallback policy and are reported, not hidden or substituted
   from observed HLT. B's topology/keys/state must replay exactly for each jet.

No refits of B or C. Import authenticates completed original comparison fits,
all evaluation shards, reports, receipts, source semantics, sample masks,
numerical environment, association policy and compatibility. A wrapper contract
identifies derived response semantics; an embedded runtime response is not
presented as an unchanged original fit.

## Finite calibration, common comparison, sealed confirmation

- Reuse the original 16,000-location / 4,000-residual / 10,000-development
  file-disjoint sample registry and fit-only plotting ranges.
- The only calibration grid is dz multiplier `{0, .25, .5, .75, 1}` on all
  4,000 residual-role jets, replicas 0/1/2. These are **training calibration**
  rows already used by the frozen residual backends, not validation evidence.
  Minimize mean histogram TV of valid dz and dz significance, first averaging
  the two observables and replicas, then equally over `all` and offline-defined
  cohorts with >=1,000 real jets. Ties choose the multiplier closest to one.
  Empty-on-both distributions contribute zero with an explicit status; missing
  on one side contributes one. There is no per-class/per-module scale search.
- Re-evaluate B, B_DZ and BC on all 10,000 development jets, replicas 0/1/2,
  in four exact 2,500-jet shards. No unresolved jet is dropped. B must reproduce
  the historical evaluation. These development rows have already been inspected;
  selection on them is explicitly exploratory.
- Selection publishes an immutable candidate definition and response digest,
  reports for **all three** candidates, eligibility reasons and fixed metrics.
  Confirmation cannot select another candidate or tune a parameter.
- A separately created/authorized confirmation stage reads exactly 20,000
  bottom-hash CMS `response_confirm` rows, stratified by the original fit-source
  mixture, covering every confirmation source file. It evaluates only the
  selected response and B control, on replicas 0/1/2 in four 5,000-jet shards.
  Source files are disjoint from all fitting and development-selection files.
  A claim binding roles, compatibility, selection, membership and stage precedes
  any confirmation particle read. Registry creation reads metadata/masks only.
  This is **not** the original >=250k full confirmation. The 20k access must be
  disclosed if that larger protocol is later revisited; these are no longer
  untouched confirmation files. No automatic access to unused rows or JC2.

## Locked development decision metrics

Use the existing 128-bin fit-only transformed histogram definitions, retaining
under/overflow, raw means, PID counts, conditional plots and jet correlations.
The primary score is the equal mean of five blocks, not the original six-block
association score:

1. Counts/state: jet multiplicity, six PID counts, four tracking-validity flags.
2. Particle kinematics: pT, energy, eta, radius, and per-PID pT.
3. Tracking: d0, dz, both errors and both significances.
4. Jet kinematics/shape: scalar/vector pT, energy, mass, eta, charged/leading/
   subleading fractions, width, e2_1/e2_2, seven radial fractions, pT ratio,
   axis displacement.
5. Joint jet correlations: mean absolute difference / 2 of off-diagonal
   Pearson correlations in the existing frozen JET_JOINT coordinates. This
   does not establish joint tracking closure and is labelled accordingly.

Histogram blocks average observables, replicas, then `all` plus each offline
cohort with >=1,000 real jets. Undefined on both sides is explicitly zero;
undefined on one side is one. No favorable-observable dropping. Correlations
are overall only. All-jet plots and rare-cell rows remain diagnostic regardless
of primary eligibility. Flags, missing states, backoff and clamps are reported.

Original B is always eligible. A challenger is eligible only if no block score
exceeds B by >0.02 and each of jet multiplicity/scalar-pT/mass absolute relative
mean bias is <= B's corresponding bias +0.02. Choose minimum primary score
among eligible candidates; exact ties prefer B, then B_DZ, then BC.

Report paired 200-draw source-file bootstrap differences, keeping replicas
together, with seed 20260925. Fewer than four independent files means uncertainty
is insufficient for a positive confirmation claim. These intervals describe
sampling uncertainty, not model-selection bias or detector systematic errors.

A **bounded confirmation improvement** requires >=0.005 absolute and >=10%
relative score reduction versus B, positive lower 95% bootstrap improvement,
the same no-regression guards, and >=4 source files. Otherwise the completed
result is `no_clear_improvement`; the selected B control is `control_retained`.
All statuses end this version; no further grid is queued. A positive status
is not a qualified detector response. The original >=99% association coverage,
six-block/tail/conditional criteria and verified physical-interface requirements
are unchanged and not established by this study. Current ~73.5% resolved
development coverage remains an explicit qualification blocker.

## Execution, gates and stopping

Three explicitly submitted stages, no auto-submission:

- `bounded_gate_r1`: real 32-residual-jet serial/process acceptance and historical
  generator parity (2 CPUs/32 GiB/2h), then 4k dz calibration (36/128/8h).
- `bounded_compare_r1`: four common evaluation shards (36/128/8h each), then
  report/selection/plots (1/32/4h). Requires completed gate receipts and measured
  acceptance envelope; no quality threshold can block a registered fit row.
- `bounded_confirm_r1`: four confirmation shards (36/128/8h each), then fixed
  confirmation report (1/32/4h). Requires completed immutable selection and
  separate dry-plan review/authorization. Exactly one confirmation in this root.

All jobs use debug, account reu-aisocial, qos_tier3, one node/task, **no GPUs**.
Spawned single-thread workers handle eight-jet chunks with a bounded 2*workers
window, deterministic merge order and 15-second progress reports. CPU count
is not capped by source-file count. Source/environment/allocation checks,
durable submission intents, exclusive task claims, exact-ID reconciliation,
atomic receipts and storage limits reuse development infrastructure.

Raw particle/calibration arrays are RAM-only. Persist only portable models,
aggregate histograms (including per-source aggregates for uncertainty), scalar
statistics, metrics, plots and authenticated registries. Study cap 12 GiB,
figures 2 GiB. Existing ~20-minute diagnostic timings are planning evidence,
not guaranteed performance. New real acceptance checks execution and measures
the new generation path; it does not certify physics. No cancel/requeue,
recovery/override grid, external writes or campaign launch is implicit.

Deliverable: one selected frozen development response and one honest final
confirmation result. If neither intervention is useful, retain B as the control
and stop. A later JC2-transfer proposal requires its own explicit scope/audit;
this comparison will not automatically generate synthetic HLT datasets.

## Implemented entry points and local evidence

`scripts/cms2jc2_response_dev.py create-bounded --parent-spec ORIGINAL_COMPARE
--project-dir CLEAN_PUSHED_WORKTREE --source-commit EXACT_COMMIT --root FRESH_ROOT`
creates only the gate spec and dry plan. The parent is the completed original
`cms2jc2_dev_de9890b2_r1/stages/compare_search_b1_r1/stage_spec.json`, which owns
both B_L and C_L fits; neither diagnostic replacement is a fitting parent.

Use existing `dry-run --spec ...` then `submit --spec ... --execute
--authorization-phrase ... --reviewed-plan-hash ...` to authorize exactly one
stage. Phrases are `AUTHORIZE CMS2JC2 BOUNDED_GATE EXACT PLAN`,
`AUTHORIZE CMS2JC2 BOUNDED_COMPARE EXACT PLAN`, and
`AUTHORIZE CMS2JC2 BOUNDED_CONFIRM EXACT PLAN`. `advance-bounded --parent-spec`
creates the next spec only after authenticated completion; it never submits.
`bounded-results --spec` prints authenticated results without particle access.

Local validation (2026-09-25): 42 focused baseline tests before editing;
207 response regressions passed with one original full-campaign test deselected,
plus an additional result-column regression. Ten new bounded checks cover
synthetic ROOT fits, serial/process replay, all three stages, confirmation seals,
read-only original outputs, invalid protocol/role/model rejection, explicit
missingness and finite stopping. No real SPORC gate or mapping-quality result
has been produced by these local tests. Source must be committed/pushed and
the genuine CPU gate must run before the larger comparison stage is created.

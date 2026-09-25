# Frozen C: observed-topology closure diagnostic

## Question and interpretation

The completed C accounting audit shows approximately correct charged counts but
about twice the observed average charged-particle momentum. Missing-value
fallback and residual noise alone do not explain it. Before refitting, separate
the learned topology/identity decisions from continuous response prediction.

This is an **oracle diagnostic**, not a candidate deployable mapping. Observed
HLT is used to fix association-derived topology, output PID, charge, tracking
validity and positive-mass state. The association remains an inferred hypothesis,
not detector truth. Improvement does not prove the physical association correct.

## Frozen inputs and populations

Reuse the completed `C_L` from the original `compare_search_b1_r1` CPU development
stage, its exact fitted parameters, association policy, review, numerical
environment, samples and histogram ranges. Authenticate its receipts and scientific
source. Do not fit anything, change the association search limits, select a new
random seed, or read JetClass2. Use all 10,000 registered development evaluation
jets, the same four 2,500-jet shards, and replicas 0, 1, 2. These are development
holdouts inside `response_fit`; neither `response_select`, `response_confirm`
nor classifier final-test data may be accessed.

## Registered computation

For every jet, generate ordinary frozen C and recompute the existing bounded
offline/HLT association. Keep ordinary results for the full population and for
resolved and unresolved subsets separately. Report conditional resolution counts
and unresolved-component reasons; a partial solution is never completed by guess.

For every fully resolved jet:

1. Require exact coverage of both particle sets by the association hypotheses.
   Reconstruct each target through the existing emission-state/coordinate codec,
   using the corresponding offline group (or whole-jet origin for additional
   particles). Require agreement with canonical `calibration_records`, categorical
   fields exactly, and four-vectors/tracking to rtol=1e-8, atol=1e-8 in canonical
   units. Record maximum absolute differences. Mass-state threshold is the existing
   1e-6 GeV convention. For the tiny spacelike p4 permitted by source-rounding
   tolerance in the shared bridge, compare energy to the nonnegative-mass
   projection and separately report its maximum raw-energy correction. Do not
   clip target coordinates to fitted model envelopes.
2. Bypass learned survival, merge, split, additional-count and output-state draws.
   Supply the observed association topology and categorical emission state, but
   predict continuous coordinates from the frozen offline-conditioned C model.
   HLT continuous values must not enter these predictions. Keep frozen response
   envelopes, scale limits and input preprocessing unchanged.
3. Evaluate `FIXED_FULL` (original correlated residual draws) and `FIXED_CENTRAL`
   (zero residuals). Frozen label-independent jet/replica keys remain unchanged;
   group keys identify singleton/merged/split emissions, and additional emissions
   use `additional:j` in canonical HLT-key order. This ordering is privileged and
   is not claimed to pair noise with an ordinary free-generation particle.
4. Missing continuous modules make that entire fixed-topology jet unestimable:
   never fill it using observed HLT momenta or an identity fallback. Report its
   count and missing modules. Compare both fixed variants with ordinary C only
   on the identical `comparable` subset. Missing state/category residual support
   retains the frozen central prediction with explicit counters, as in C itself.

Publish six distributions: `FREE_ALL`, `FREE_RESOLVED`, `FREE_UNRESOLVED`,
`FREE_COMPARABLE`, `FIXED_FULL`, `FIXED_CENTRAL`. Each contains its own offline,
real HLT and three proxy histograms. Zero-size subsets are explicitly unavailable,
not failures. All resolved targets also receive mechanism/PID count and scalar-pT
accounting, with a separate comparable-target accounting. Persist sufficient
statistics only; no particle arrays or association library on disk.

## Outputs and decision

Report particle counts, per-type pT and scalar-pT budgets, charged fraction, jet
kinematics, shape and tracking distributions, all existing offline-defined
cohorts, mechanism/PID budgets, response/scale clipping and residual fallback.
For the fixed variants additionally count the *actually selected* residual-bin
levels (category, category+pT, category+pT+eta, category+pT+eta+crowding), missing
backend and missing category per emission. This is distinct from stored-cell
inventory. Include real-vs-generated plots and the exact population denominators.

If charged-pT excess persists with fixed topology/state, investigate continuous
calibration and its conditioning next. If it substantially improves, investigate
the topology/state models and group selection. Both can contribute. Do not name
a winner or claim statistical significance from these exploratory results;
replicas and particles are not independent jets.

## Execution and isolation

Fresh disjoint root and pushed, clean pinned worktree; no old jobs or artifacts
are changed. Six CPU-only `tier3` jobs with account `reu-aisocial`, qos `qos_tier3`:

| Task | CPUs | RAM | Limit | Depends on |
| --- | ---: | ---: | ---: | --- |
| correctness acceptance | 2 | 32 GiB | 2 h | none |
| four evaluation shards | 36 each | 128 GiB each | 24 h | acceptance |
| combined report | 1 | 32 GiB | 4 h | all four shards |

Evaluation reads each shard once, with source checks before/after. Bounded spawned
processes work on eight-jet chunks, not one long task per input file. At most two
chunks per worker are in flight; each native numerical pool is one thread. Raw pairs
and emissions exist only in RAM. Workers return compact statistics, not emissions.
Parent aggregation may limit scaling: 36 CPUs is
capacity, not a claim of continuous 100% utilization or a measured speedup. Log
completed/total jets, resolved/comparable counts and heartbeat every 15 seconds.
Record resource measurements through the existing worker receipt mechanism.

Acceptance checks up to 32 frozen shard-0 jets, codec closure, exact ordinary-C
replay and serial/two-process equality (floating sufficient statistics tolerate
rtol=atol=1e-10). It tests implementation correctness, not scientific quality.
Quality never cancels a registered row. Invalid data, lineage, codec closure or
source fail closed. Resource envelopes remain unmeasured for this new diagnostic;
site `sbatch --test-only`, explicit plan hash and authorization are required.

## CLI

`cms2jc2_response_dev.py create-c-topology --parent-spec <original compare spec>
--project-dir <pinned checkout> --source-commit <full commit> --root <fresh root>`
creates `stages/observed_topology_r1/stage_spec.json` and its plan. Existing
`dry-run`, `submit`, `monitor` and journal reconciliation apply. Live phrase:
`AUTHORIZE CMS2JC2 CPU DEVELOPMENT CTOPO EXACT PLAN`.
`c-topology-results --spec <new stage spec>` prints authenticated results; `--json`
prints the full report. Creation, inspection and dry-run never submit jobs.

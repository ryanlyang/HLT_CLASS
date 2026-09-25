# Frozen B tracking audit

Date: 2026-09-25. Authorized isolated development diagnostic, additive to the
CMS2JC2 CPU development plan. Existing B/C models, studies and scheduler jobs
are read-only. This plan authorizes implementation, not implicit submission.

## Question and controls

The user's completed 10k development evaluation reports B dz histogram TV
0.2951 and dz significance TV 0.3240, versus C 0.0587 and 0.0539. B has better
count/kinematic closure on many printed observables. Signed means alone do not
establish a sign bug. Determine whether B's tracking mismatch originates in
its conditional central prediction, sampled residual response, or clipping.
Do not refit, retune on these results, change conventions or splice B and C.

Import the original completed comparison's B_L response, fit, four histogram
shards, report, samples and ranges through authenticated receipts. Require
unchanged scientific source, CMS inputs, compatibility and numerical environment.
Use all of the SAME 10,000 development evaluation jets (four shards of 2,500),
with replicas 0/1/2 and original label-independent keys. No association search,
training, JetClass2 particle access, outer response_select/response_confirm or
final-test access. Evaluation includes unresolved, empty and poor outputs.

## Registered interventions

| Variant | Change to sampled residual, in every eight-coordinate emission block |
| --- | --- |
| FULL | None; exact original B generator replay |
| TRACK_CENTRAL | Zero d0, dz, log(d0err), log(dzerr) residuals |
| DZ_CENTRAL | Zero dz residual only |
| DZERR_CENTRAL | Zero log(dzerr) residual only |

All variants retain original predicted central values, categorical topology,
PID, charge, validity, kinematic coordinates, scale and response envelopes and
fallbacks. They must preserve FULL four-vectors exactly. CENTRAL here means
central prediction in the model coordinate system, not an unbiased physical
mean after exponentiation/clipping. No observed HLT values enter generation.
Missing backends retain the original flagged central fallback; missing modules
retain original identity fallback, not observed targets. Do not disable safety
clipping to manufacture a better output. FULL matches the production generator
on every jet-replica; interventions check exact preservation of unaffected fields.

## Required output

Keep full existing offline/real/proxy histograms, conditional cohorts, raw
means, tails/overflow, valid tracking, significance and jet-summary correlations.
Use original location-fitted bins, not newly tuned ranges. The report prints
tracking means AND histogram TV, clipping denominators and missing-state reasons.
Means and replicas are descriptive, not independent statistical measurements.

For each emission module/mechanism, output PID, validity mask and tracking
coordinate, record applicable and inapplicable counts separately. On applicable
coordinates save aggregate central, raw residual increment, applied increment,
pre-clip and post-clip response moments; central-outside-envelope counts;
response low/high clipping and absolute correction; scale clipping when a scale
was evaluated; residual availability and disabled-residual counts. Error
coordinates are in log(mm), displacement coordinates in mm under the frozen
bridge. Do not exponentiate unbounded pre-clipping log errors. Also retain the
existing all-coordinate counters (which include inactive placeholders), clearly
separate from valid tracking counters and boolean-any-per-jet flags.

Record actually selected residual-cell keys/levels and low-statistics/backoff
reasons; stored cell inventory alone is not occupancy. Publish model scale and
response limits by coordinate, with model hash, for interpretation. No raw
particles, per-jet traces, feature rows or proxy datasets are persisted.

## Execution and isolation

Fresh disjoint root, stage `frozen_b_tracking_r1`, source-pinned clean pushed
worktree; all jobs tier3, reu-aisocial, qos_tier3, one node/task, no GPU:

1. `bt_acceptance`: 2 CPUs, 32 GiB, 2h; 32 evaluation jets, all replicas and
   variants; compare serial and two-process aggregates and exact FULL replay.
2. `bt_eval_0..3`: each 36 CPUs, 128 GiB, 24h, afterok acceptance. Each shard
   evaluates all four variants together. Bounded spawned pool, eight-jet chunks,
   at most 72 outstanding chunks, single-thread native libraries. Print completed
   jets and 15-second waiting heartbeats. There is no guaranteed speedup or
   full-time 36-core utilization; these are unmeasured development envelopes.
3. `bt_report`: 1 CPU, 32 GiB, 4h, afterok all shards. Aggregate and plot all
   variants; no automatic winner, refit or follow-on submission.

Full dry review, site test-only resource admission, exact plan hash and explicit
`AUTHORIZE CMS2JC2 CPU DEVELOPMENT BTRACK EXACT PLAN` are required for live
submission. No hold/cancel/requeue of another job. Failed attempts retain their
root. Atomic authenticated outputs and existing 12-GiB study / 2-GiB figure
and report budgets apply. Historical FULL bins/counts/identities are exact;
floating aggregate moments allow only rtol=atol=1e-10 across reduction order.
Runtime source/environment and parent receipts are checked at start and end.

## Verification and next decision

Test real B fits, 8/16-coordinate masks, immutable unaffected fields, missing
states/modules, validity-aware counters, empty outputs, deterministic keys,
serial/process aggregation, exact historical replay, source/receipt corruption,
six-job dry/live mock dependencies/idempotence, and tiny ROOT through report.
No local test claims SPORC acceptance already passed. No Weaver/GPU check applies.

After B and the separately running C topology diagnostic finish, inspect
conditional tracking and momentum results before proposing a new model. The
repeatedly inspected development jets cannot supply independent confirmation.

## Interface

`scripts/cms2jc2_response_dev.py create-b-tracking --parent-spec <original
compare stage> --project-dir <clean checkout> --source-commit <full pushed SHA>
--root <fresh sibling root>` creates only metadata. Use `dry-run`, then the
separately authorized `submit`, against `stages/frozen_b_tracking_r1/stage_spec.json`.
`b-tracking-results --spec <stage> [--json]` reads authenticated completed
products only. All existing interfaces retain their original semantics.

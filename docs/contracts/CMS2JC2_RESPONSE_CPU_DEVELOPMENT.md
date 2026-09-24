# CMS2JC2 CPU development v1

Authority: [development plan](../plans/CMS2JC2_RESPONSE_CPU_DEVELOPMENT_PLAN.md).
This is additive to CMS2JC2 response v1, not a relaxation of production gates.

Artifacts have the existing content-hashed CMS2JC2_RESPONSE_ prefix and new
DEV_* kinds at v1: import, samples, study, stage, plan, claim, outputs, association,
B diagnostic, fit, metric ranges, histogram shard and evaluation report.
Every artifact carries final_test_accessed=false. Validation rederives canonical
membership and stage plans; hashing alone does not authorize new input roles.

Only authenticated original response_fit files are readable. Location,
residual and development-evaluation files are disjoint. Original response_select
and response_confirm are never exposed through a development API. All new
outputs have fresh roots disjoint from raw data and imported preparations.
The old JC2 source is not imported; dzfix transfer is explicitly deferred.

The four association policies retain exact v1 optimization and change only
declared deterministic resource limits. Development histogram closure is not
the production six-block score. Low coverage/poor closure completes with honest
diagnostics; invalid data, stale source, corruption and missing required model
receipts fail closed. No candidate can become a qualified response here.

## CPU64 execution extension, 2026-09-24

New additive `DEV_STAGE64/v1`, `DEV_CPU64_EXECUTION/v1` and
`DEV_CPU64_REUSE/v1` artifacts isolate the new execution from legacy
`DEV_STAGE/v1`. They retain the 16k/4k/10k populations, SEARCH (or the exact
declared confirmed policy), all model families and evaluation definitions.
The execution profile binds 64 CPUs per fit, eight-jet RAM work chunks,
128-item maximum outstanding queue, 15-second progress heartbeat, and 128-GiB
requests. A/C numerical fit threads remain 16 and B remains one.

The parent confirms the identity/count of each returned chunk. Stratified
bottom-hash reservoirs merge to their unchanged per-file quota and final N/k
weights; neither completion order nor chunk size changes calibration records.
The reader opens each file once with beginning/end checksum authentication;
raw pairs and calibration records never become durable products. Failed
workers propagate an error, not incomplete records or apparent completion.

Fresh comparison roots may authenticate the previous completed preparation and
confirmation instead of rerunning them. Reuse binds exact report/receipt bytes,
frozen canonical samples/ranges, matching-policy hash, source-surface scientific
hashes, numerical environment and compatibility. Only the enumerated execution
source files may differ. There is no cross-root import of old models or live
jobs and no mutation of old artifacts. Standard CPU-only scheduler admission,
explicit dry/live submission, claims and output receipts still apply. This
extension supplies no scientific acceptance or measured 64-CPU speed guarantee.

Source-pinned CPU stages use journaled submission, explicit stage authorization,
allocation checks, environment fingerprints, exclusive claims and last-published
output receipts. No live submission by default, no automatic downstream stages,
no editing/reusing incomplete outputs, no modifying existing scheduler jobs.

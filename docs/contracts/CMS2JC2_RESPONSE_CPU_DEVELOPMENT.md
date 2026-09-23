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

Source-pinned CPU stages use journaled submission, explicit stage authorization,
allocation checks, environment fingerprints, exclusive claims and last-published
output receipts. No live submission by default, no automatic downstream stages,
no editing/reusing incomplete outputs, no modifying existing scheduler jobs.

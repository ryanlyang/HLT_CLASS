# JetClass2 dz-fix fusion chain contracts

Authority: `plans/JETCLASS2_DZFIX_FUSION_CHAIN_500K_PLAN.md`.

Artifacts use `JETCLASS2_DELPHES_DZFIX_FUSION_CHAIN_*` and canonical content
hashes. LAUNCH_SPEC, SOURCE_IMPORT and CAMPAIGN_SPEC are now **v2**; all other
artifact kinds retain v1. The v1 continuation-bound input route is obsolete
and is not silently interpreted as direct-screen provenance.

LAUNCH_SPEC/v2 binds the explicit producer SALIENCE_SCREEN_SPEC/v2, its exact
eight-task live ledger and `complete` job, new source checkout, independent
launch/campaign roots, debug resources, graph and population. Parent commands
and dependency closure must equal the source screen plan after exact-ID
resolution. Dry, incomplete, foreign, or duplicated-job ledgers fail closed.
SOURCE_IMPORT/v2 additionally binds screen/selection/completion/profile and
all eight task receipts, selected foundation spec/lock, and current semantic
preparation file hashes. Foundation location and hash must match the screen's
candidate registry as well as selection. No continuation/production preview is
required. All compact payloads are authenticated before reuse.

The producer's debug-screen/tier3-production separation remains intact. The
consumer does not inherit that production resource profile: it registers its
own debug execution and fresh acceptance. The current completion boundary is
21748725, read from the live ledger, not hardcoded in the reusable Python code.

CAMPAIGN_SPEC embeds that import, immutable graph/recipe/resource registration,
validation firewall and 25-task graph (4 gates including storage + 21 science).
TASK_REPORT binds campaign, source, task, dependency attestations and every output
byte hash. TRAINING_REPORT wraps the unmodified shared training-kernel report
and adds report-subset metrics, teacher lineage, selected checkpoint and native
input semantics. Banks retain the existing exact-row probability-bank contract
(train T2, validation T1), never unlabelled/raw-logit mixtures.

VALIDATION_PARTITION binds ordered identities/labels and deterministic stratified
50/25/25 assignments. ACCEPTANCE binds the exact campaign/source/runtime and
actual single/paired execution, bank/weight round trips, memory and time evidence.
No acceptance-only weights count as scientific fits. All ordinary artifacts
record `final_test_accessed=false`; ordinary tasks have no test capability.

Publication is immutable/atomic; existing roots are not overwritten. Exact DAG
submission uses canonical dry runs, exclusive submission claims, journalled IDs
and afterok only. An existing live ledger is verified and returned, not resubmitted.
Each task has one immutable output directory; a partial failed task is not
silently overwritten or resumed. Recovery needs operator review and a new
registered execution, not an automatic retry. No cancellation of other
campaigns is authorized by these contracts.

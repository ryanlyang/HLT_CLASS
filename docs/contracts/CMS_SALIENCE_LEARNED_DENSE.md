# CMS_SALIENCE_LEARNED_DENSE contracts

## Standalone direct fusion comparison (2026-09-19)

Scientific authority: `docs/plans/CMS_SALIENCE_DIRECT_FUSION_500K_PLAN.md`.
CAMPAIGN_SPEC/v6 requires `ladder=direct_fusion`, SPORC/debug, GRAPH/v3,
original dense preparation, shared references and accepted GPU evidence.
The single arrow is persistent U000 -> D000. Its new `ACQUIRE_D000` sees
primary D000 plus privileged U000 and learns from U000 logits. `WITHDRAW_D000`
starts from acquisition with a fresh optimizer, learns from its frozen bank,
and selects the alpha-zero route under the unchanged withdrawal loss/schedule.
`CARRIER_D000` removes context parameters with exact prediction parity.
Ordinary `DIRECT_D000` remains imported single-view KD, not this fusion route.
This two-fit method is not compute matched to the one-fit KD control.

There are eleven science tasks: five CPU shared imports, two fresh GPU fits,
one new GPU reducer, one GPU extraction, aggregate and complete. GRAPH/v3 has
six logical fits (four imported), one extraction and two logical reducers
(one imported). Old dense/coarse hashes and v1-v5 campaign semantics stay
unchanged. Existing native scientific kernels, seeds, dataset/split/matching,
batch, resources and final-test seal are unchanged. Historical family naming
is retained only for compatibility and the seed namespace.

PREPARATION_IMPORT/v3 uses the existing exact reviewed preparation-code
equivalence, but binds a direct-fusion consumer instead of a coarse consumer.
SHARED_SOURCE/v2 authenticates the same five original dense reference tasks.
ACCEPTANCE_IMPORT/v2 and ACCEPTANCE_REUSE/v2 bind the new consumer to the
original accepted dense-v3 U000/U000 GPU envelope. No new GPU measurement is
claimed and no threshold is relaxed. These versions cannot be relabelled as
coarse import evidence. Consumer receipts and source hashes are mandatory.

All new jobs use `cmsdf_`, with explicit `import_` labels for reused tasks.
`create-direct-fusion` is dry by default; `queue_cms_direct_fusion.sh prepare`
additionally verifies/imports CPU evidence, without submitting jobs. Its
`submit` mode uses `AUTHORIZE CMS DIRECT FUSION 500K EXACT SPEC`. The normal
source-pinned gate/dry-run/live journal checks apply. Completed references
need valid output receipts, not old Slurm dependencies. The separate coarse
campaign is neither a source nor a dependency. `retire-dense` rejects direct
campaigns before any scheduler query or mutation, including with a retirement
authorization phrase. No cancellation or source output rewriting is permitted.

## Opt-in accepted dense preflight reuse (2026-09-19)

Following explicit user authorization to skip the duplicate GPU preflight,
new CAMPAIGN_SPEC/v5 extends the coarse replacement with a required
`acceptance_import`. It must also have a compatible `shared_source` and
original preparation import. V1-v4 semantics are unchanged: old roots cannot
acquire this option by adding a field, editing a file or relabelling evidence.

ACCEPTANCE_IMPORT/v1 binds the genuine dense-v3 execution acceptance and
preflight receipt, source campaign/commit/job, resource-probe implementation
and scientific runtime code. It reuses the exact accepted full-population
U000/U000 paired-cache / longest-jet batch-256 envelope. All 30 probes and
85% CPU / 90% GPU limits must still validate against both source and consumer.
Data, preparation, batch, model, training, partition and requested resources
remain identical. Native kernels, shared model/data trees, probe functions,
worker shell and allocation adapter are source-fingerprinted. The preflight
driver allows only the exact reviewed `==3` to `>=3` schema-predicate AST pair.
Any other runtime change requires fresh measurement. The declared conda
environment must match; unchanged installed software is an explicit assumption,
not a newly measured claim.

The exact reviewed coordinate/preflight AST pairs include both Python 3.10
and Python 3.13 `ast.dump` encodings of those same source functions. The latter
omits empty-list fields. Existing fingerprint serialization is unchanged, so
old imported preparation descriptors remain verifiable. This correction does
not accept arbitrary AST edits, cross-encoding coordinate pairs, or changed
preparation/model/runtime files. No schema or scientific semantics change.

The consumer publishes `execution_acceptance_import.json` as
ACCEPTANCE_REUSE/v1, with `fresh_gpu_measurement=False` and original measurements
labelled as source evidence. It never synthesizes `execution_acceptance.json`
or an EXECUTION_ACCEPTANCE claiming a new GPU run. Its preflight task becomes
a CPU evidence import, followed by the normal authenticated task receipt.
The gate rechecks original evidence and compatibility on every use. Creation,
import or a passing gate still does not itself submit/cancel science.

Use `create-coarse --reuse-dense-preflight` to opt in. The explicit helper mode
`reuse-and-switch DENSE_SPEC NEW_ROOT` creates a fresh v5 root, verifies/imports
foundation and acceptance on CPU, then performs the normal exact-ID retirement
and full science submission. It submits no new GPU preflight. Existing `prepare`
remains the v4 fresh-preflight path. No source root, job or file outside the
authorized dense-specific retirement set is changed.

## Coarse replacement extension (2026-09-19)

`docs/plans/CMS_SALIENCE_LEARNED_COARSE_500K_PLAN.md` authorizes the coarse
variant. CAMPAIGN_SPEC/v4 requires `ladder=coarse`, GRAPH/v2 and an explicit
`shared_source` (null for a fresh campaign, SHARED_SOURCE/v1 for reuse).
Its five arrows are U000/U050/U100/D066/D033/D000, with exact half/third
coordinates. Legacy v1-v3 specs keep GRAPH/v1 and its exact content hash.
The historical family namespace is retained to preserve common-node seeds;
it does not mean that a v4 graph is dense. Execution acceptance v2 and the
strict 85% CPU / 90% CUDA policy are unchanged.

PREPARATION_IMPORT/v2 binds both preparation-code identities and the reviewed
coordinate-AST migration `add_coarse_coordinates_v1`. Only that exact pair
of AST hashes (or identical code) is accepted; all preparation modules/trees,
split, budgets, view semantics and foundation lineage still have to agree.
New names do not change old coordinate values. Original foundation payloads
are not rewritten. Existing PREPARATION_IMPORT/v1 stays strict and unchanged.

SHARED_SOURCE/v1 authenticates one original accepted dense campaign and its
full exact live science ledger. Only M0HLT, OFFLINE, U000, reduce_U000 and
DIRECT_D000 may be imported. Full model/training file/tree Git identities and
reference-worker/seed/schedule AST identities must agree. Source gate, shared
node identities, resources and original preparation producer must agree.
Each completed source output needs its exact task receipt, report/checkpoint
hashes and teacher-bank lineage. Copies go into the new root; imported reports
retain explicit original artifact/receipt identity and banks bind the imported
U000 report. Parent data are read-only. No dense fusion carrier is imported.

The coarse science DAG has 31 tasks, 14 logical fits, five extractions and ten
reducers. With shared reuse, five tasks are CPU imports and only ten fits are
new. Source dependencies remain symbolic in the canonical dry ledger. Live
submission resolves pending/running jobs from exact authenticated ledger IDs;
authenticated completed outputs remove the external scheduler dependency.
Internal dependencies are always preserved. Every actual command/job is
journaled before the next submission, and replay validates exact commands
before resuming. Missing/corrupt outputs, absent failed sources and changed
ledgers fail closed. Only a dependency rejection with newly authenticated
completion may retry without the old external job ID.

`create-coarse --source-spec DENSE_SPEC --campaign-root NEW_ROOT
--source-commit COMMIT` infers original preparation and resources. Creation
is dry and never cancels. Run/import `foundation`, submit a fresh gate, then
validate that gate. Coarse live submission uses
`AUTHORIZE CMS SALIENCE LEARNED COARSE 500K EXACT SPEC`.
`retire-dense --spec COARSE_SPEC` previews only the 41 dense-specific tasks.
Execution additionally requires the new gate and the separate phrase
`CANCEL CMS DENSE LADDER KEEP SHARED`. It checks live owner/name/account
against the source ledger, cancels exact IDs, preserves all five shared tasks,
and never deletes source artifacts or touches another campaign.

## Original dense contract (unchanged)

Scientific authority: `docs/plans/CMS_SALIENCE_LEARNED_DENSE_500K_PLAN.md`.
This family is not interchangeable with either historical CMS Strategy B or
JetClass2 learned-handoff artifacts. Scientific artifacts use schema version 1 and
canonical content hashes. Parent artifact hashes and payload SHA256s must be
verified before use. All ordinary workers reject final-test access.

The graph, 500k/250k/250k budgets, PT_LINEAR matcher, native schema, persistent
support, optimization recipe and SPORC site are registered, not free-form
overrides. Requested CPU count/preprocessing workers/memory may be set
within the site envelope at creation, then measured by this campaign's gate.
`CAMPAIGN_SPEC/v2` and `/v3` freeze `site.partition` to `tier3` (default) or
`debug`. Account, QOS, environment, cluster and A100 type stay fixed. Legacy
`CAMPAIGN_SPEC/v1` validates only with the original tier3 site. An acceptance
artifact must match its own campaign's exact selected site; changing a queued
job with `scontrol` is not a supported migration.
The full staged command plan is fixed at creation. Live submission needs the
exact authorization phrase and canonical dry-run ledger for that stage.

Preparation publishes authenticated raw-file fingerprints, role selection,
per-source assignments, train-only scales and a foundation lock. The final-test
budget is a sealed commitment, not falsely reported as already evaluated.
Science publishes 20 training reports, 8 exact extractions, 16 probability
banks and an aggregate. Per-task completion receipts bind every output file;
an output's existence is never sufficient to skip or reuse a task.

Inference selects the batch layout from the registered model route, not from
whether the RAM cache happens to retain a context view. Ordinary single-view
and alpha-zero fusion predictions request only primary inputs; privileged
fusion predictions still request both views. Preflight compares the zero-route
fusion and extracted ordinary model on the same paired validation cache with
byte-exact probability equality. This enforces the existing extraction contract,
not a new scientific protocol or a relaxed parity threshold.

Preflight runs its four miniature routes with separate model/optimizer/loss
lifetimes. Memory telemetry prints CPU RSS high-water, current/peak CUDA
allocation, CUDA reservation, the CPU request and GPU capacity after cache
construction, each fit, explicit alpha regime, extraction and route cleanup.
CUDA peak statistics are reset only once at preflight entry; cleanup cannot
erase an earlier route's high-water mark. Legacy campaign v1/v2 requires both
CPU RSS and peak CUDA allocation strictly below 85% of their respective limits.
A refusal identifies CPU RAM, CUDA, or both and prints the measured bytes;
it never publishes execution acceptance or authorizes science.

Newly created `CAMPAIGN_SPEC/v3` binds an explicit `acceptance_policy`:
CPU limit 0.85, CUDA limit 0.90, five consecutive withdrawal updates per
alpha (1, 0.5, 0), batch 256, longest-U000-jet selection. Both paired routes
execute these probes using one optimizer per route, without clearing the
allocator or resetting high-water statistics between steps. The 256 longest
training jets are selected explicitly; additional class-coverage rows in the
miniature cannot displace them. Model, losses, scientific batch and schedule
are unchanged.

`EXECUTION_ACCEPTANCE/v2` is required for campaign v3. It records the exact
policy and all 30 ordered `withdrawal_probe` rows, each binding route, alpha,
step, actual batch size, peak RSS and peak CUDA allocation. Science checks
complete coverage, memory measurements and the strict 85%/90% limits again.
Missing/shortened probes, reduced batches, stale versions and altered policies
fail closed. Campaign v1/v2 continues to require acceptance v1 and cannot carry
the relaxed policy. Local tiny fixtures reduce population budgets only for
tests; genuine 500k acceptance requires full batches of 256. Only a fresh
source-pinned campaign/gate may use v3; old failed gates do not become valid.

The CMS fusion adapter shares one compact, padding-merged cross-attention
bias across its four residual injections. Full Weaver pair construction,
BatchNorm inputs, stochastic call order, trainable gradients, both withdrawal
routes and exact alpha-zero extraction are preserved. No state-dictionary
keys, scientific semantics, batch size or acceptance thresholds change, so
existing scientific v1 / campaign v3 / acceptance v2 schemas remain unchanged.
This allocation-only implementation still needs a fresh source-bound A100
preflight; legacy acceptance or an estimated saving cannot authorize science.
Non-CMS fusion adapters keep the prior mask-allocation path.

## Operator surface

`scripts/cms_salience_learned.py create` requires `--split-manifest`,
`--data-root`, `--campaign-root`, and `--source-commit`. It uses the checkout
containing the script and insists on clean pushed source. The root must be new
and disjoint from its read-only sources. Default CPU/worker/memory request is
16/16/192000 MiB; overrides are frozen in the spec, not applied with `scontrol`.
Use `--partition debug` to select debug explicitly, subject to RC policy.
This flag controls preparation, preflight, and all 46 science tasks.

Optional `--reuse-preparation-spec /absolute/old/campaign_spec.json` imports
only a completed original preparation (no chained imports). `PREPARATION_IMPORT/v1`
binds the canonical source spec, source foundation, all preparation receipts,
and matching preparation-code Git objects. The registered native preparation
module, storage module, scouting and data trees must be identical between
producer and consumer commits; the coordinate implementation is also checked.
This deliberately conservative compatibility rule rejects unreviewed code
changes. Scientific graph, budgets, split and view identities must agree.
The source root must be disjoint from the new root and remain available.
The new prepare stage has one `foundation` import task: it verifies all hashes
without rewriting or copying source payloads. A source GPU acceptance cannot
satisfy the new gate. There is no automatic job cancellation or submission.

`submit --spec SPEC --stage prepare|gate|science` is dry by default. Creation
already writes each canonical dry ledger. Add `--execute` and
`--authorization-phrase "AUTHORIZE CMS SALIENCE LEARNED DENSE 500K EXACT SPEC"`
only for the explicitly intended stage. Finish preparation before gate, and
verify `gate --spec SPEC` before science. No stage cancels or requeues anything.
`status` verifies task receipts; `results` prints completed references, direct
KD, and extracted rungs without waiting for the aggregate. Every command takes
the canonical `--spec` path except `create`.

Live submission uses the shared exact-ID journal and can resume a partially
submitted stage without duplicating already journaled jobs. Failed training
has no rolling resume; partial output and stale running-lock cleanup are
deliberately not automatic. Diagnose the exact task before authorizing a
restart; this CLI does not broadly cancel, delete, overwrite, or relaunch it.

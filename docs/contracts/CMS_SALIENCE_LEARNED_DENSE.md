# CMS_SALIENCE_LEARNED_DENSE contracts

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

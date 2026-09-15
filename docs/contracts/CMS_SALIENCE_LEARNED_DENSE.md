# CMS_SALIENCE_LEARNED_DENSE_*/v1

Scientific authority: `docs/plans/CMS_SALIENCE_LEARNED_DENSE_500K_PLAN.md`.
This family is not interchangeable with either historical CMS Strategy B or
JetClass2 learned-handoff artifacts. Contracts use schema version 1 and
canonical content hashes. Parent artifact hashes and payload SHA256s must be
verified before use. All ordinary workers reject final-test access.

The graph, 500k/250k/250k budgets, PT_LINEAR matcher, native schema, persistent
support, optimization recipe and SPORC site are registered, not free-form
overrides. Only requested CPU count/preprocessing workers/memory may be set
within the site envelope at creation, then measured by this campaign's gate.
The full staged command plan is fixed at creation. Live submission needs the
exact authorization phrase and canonical dry-run ledger for that stage.

Preparation publishes authenticated raw-file fingerprints, role selection,
per-source assignments, train-only scales and a foundation lock. The final-test
budget is a sealed commitment, not falsely reported as already evaluated.
Science publishes 20 training reports, 8 exact extractions, 16 probability
banks and an aggregate. Per-task completion receipts bind every output file;
an output's existence is never sufficient to skip or reuse a task.

## Operator surface

`scripts/cms_salience_learned.py create` requires `--split-manifest`,
`--data-root`, `--campaign-root`, and `--source-commit`. It uses the checkout
containing the script and insists on clean pushed source. The root must be new
and disjoint from its read-only sources. Default CPU/worker/memory request is
16/16/192000 MiB; overrides are frozen in the spec, not applied with `scontrol`.

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

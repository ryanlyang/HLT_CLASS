# K2 original-memory restoration / v1

Execution-only authorization: on 2026-10-05 the user explicitly requested the
original 320000-MiB (312.5-GiB) segmented execution back, without another GPU
preflight. Supplied read-only evidence shows original D025 part2 saved through
epoch 76, incomplete; original remainder IDs 21798745-21798755 are CANCELLED
with Requeue=0. Unused 128-GiB gate/launcher 21807909/21807910 are PENDING and
no 128-GiB science ledger exists. These are observations, not hardcoded reusable
job identities. Registration derives and authenticates all IDs from ledgers.

## Immutable restoration, not an in-place rewrite

The original native acceptance and complete part1/part2 artifacts remain at
the clean `71c1bde2d759b11cddb1f5cfbc1d199847bd6eef` execution. Scientific imports
remain at `c891da0d45dd3251dea9ea72df975bb96bae3570`. The unused
`c640942f8a4a91086016fa4bd151ae5d3bc9c31a` v2 attempt is only a retirement
target. It contributes no model, optimizer, metric or acceptance measurement.
All three prior roots/checkouts stay unchanged. Slurm restart flags, original
ledgers, checkpoints and specs are not patched or re-labelled.

Register `K2_SEGMENTED_CAMPAIGN_SPEC/v3` in a fresh sibling root and clean
pushed executor. New job IDs replace only the unfinished chain. Retain every
scientific setting, membership, teacher, original full-state binding, RNG,
AdamW state, selected/current weights, history, absolute LR/epoch and patience
clock. D025 resumes after the authenticated part2 endpoint, not from scratch.
There is no new epoch budget. If part2 completed the fit, part3 remains a spare.

The `K2_SEGMENTED_RESTORE_RESUME_IMPORT/v1` descriptor binds the same full
checkpoint chain as the v2 import, but explicitly declares original 320000-MiB
resources. Verify byte-identical training kernel, donor source/prefix, actual
completed native gate, both successful segments, every saved payload/manifest,
teacher/population binding and endpoint. Any donor advancement, remainder
claim/execution/receipt or mismatched source fails closed. Independently copy
all authenticated state bytes into the new root; no hard links, edits,
deletions or overwrites. Preserve the 32-GiB state budget and 4-GiB reserve.
This costs additional disk space and CPU/IO authentication time, not GPU work.

## Original native gate reuse

`K2_SEGMENTED_REUSED_NATIVE_GATE/v1` binds the actual original acceptance hash,
preflight receipt, donor spec, independently verified checkpoint-copy receipt
and original resources. It explicitly says `native_gate_reused=true` and
`new_gpu_preflight_run=false`. The returned acceptance retains its original
campaign identity and measurements. No synthetic fresh acceptance or fresh
preflight task receipt is created.

This narrow amendment supersedes the fresh-gate requirement only for v3,
returning to the already accepted native envelope with the unchanged kernel.
The 128-GiB v2 campaign still requires its own fresh gate and cannot borrow this
exception. Training/reduction workers still authenticate their exact Slurm
allocation, GPU model/capacity and installed software against original native
evidence. Validate the real cache/teacher binding before restoring D025 state.
Tests do not manufacture new SPORC evidence. No final-test inference occurs.

## Eleven science jobs; no gate job

The full dry plan and science plan each contain eleven jobs:

1. D025 part3, then its completed-selected-fit T2 reducer.
2. D000 parts1/2/3, then its completed-selected-fit T2 reducer.
3. HLT-x1 compression parts1/2/3, aggregate, complete.

The completed native gate/part2 are authenticated artifact dependencies, not
new Slurm jobs or dependencies on expired IDs. All remaining new jobs form an
exact sequential afterok chain. No preflight, after-gate launcher, D025
parts1/2, matching or completed scientific fit is submitted. Live stage `full`
remains forbidden; authorize stage `science` after the full dry run. Stage
`gate`, or an attempt to execute preflight/after_gate, fails explicitly.

GPU segments retain one A100, four CPUs, 320000 MiB and 23h. Reducers retain
the same GPU/CPU/RAM and 6h. Aggregate/complete retain one CPU, 8192 MiB and 4h.
Default debug, pending partition-only debug/tier3 changes allowed. Keep
account reu-aisocial, QoS qos_tier3, no-requeue and the existing clean-epoch
pause/resume behavior. No guarantee of faster queueing is made.

## Exact retirement and publication

`K2_SEGMENTED_ABANDONED_EXECUTION/v1` binds the unused v2 spec, its validated
gate ledger and two job IDs. Reject any 128-GiB worker claim/execution/task,
live or ambiguous science submission (including intents/journal), or unknown,
running, completed or failed gate accounting. An unused dry plan is allowed.
The original eleven jobs must already be cancelled at registration.

The new retirement registry contains those eleven already-cancelled IDs plus
the two unused gate IDs. Dry retirement changes nothing. Authorized live
retirement uses the PENDING filter and normally cancels exactly the two unused
128-GiB jobs; a race to RUNNING blocks science, never kills an active worker.
Require all thirteen IDs cancelled before submitting any restored science.
Do not touch completed jobs or jc2fc/jc2fcr/other campaigns. Revalidate the
native gate, copy and completed accounting before exact journaled submission.

## Interfaces

CLI: `jetclass2_k2_segmented.py create-restored --abandoned-spec ...
--campaign-root ... --source-commit ...` authenticates/copies on CPU and writes
the two dry plans. `prepare-restored --spec ...` idempotently completes an
interrupted CPU copy/attestation; it never runs a native preflight and is
forbidden for v1/v2. `scripts/queue_jetclass2_k2_restored.sh` requires PROJECT_DIR,
K2_RESTORE_COMMIT, K2_RESTORE_ROOT and K2_ABANDONED_SPEC. Default is dry-only;
`--execute` additionally requires K2_DEBUG_POLICY_CONFIRMED=yes, retires only
registered pending IDs and submits science directly. Existing retirement and
exact-spec authorization phrases are unchanged. Keep all prior roots for
audit/recovery and never rerun an ambiguous submission blindly.

# All-eligible fusion continuation contracts v1

Authority: [plan](../plans/LUKA_FULLSIM_ALLJETS_FUSION_PLAN.md).

LUKA_ALLJETS_FUSION_{GATE,SPEC,PREFLIGHT,PLAN,TASK,RESULTS,DIAGNOSTICS,SUBMISSION_CLAIM}/v1
carry canonical content/parent hashes, historical_test_evaluated=true,
historical_test_repurposed_for_training=true, independent_test_available=false,
final_test_accessed=false, evaluation_scope=exploratory_development_validation.
The last flag is not a claim of an untouched historical holdout.

GATE binds new source and original parent spec file bytes, original completion
and all task receipt hashes, both-population comparison rows, prepared/cache
hashes, U000 bank, scientific registration and fixed GPU envelope. Parent
authentication runs its old validator in its own worktree. All prior src bytes
remain identical, and only the five production fusion modules plus the named
diagnostic-only module may be added. Cache inputs
keep their original ALLJETS contracts, source and physical fingerprints.

PREFLIGHT binds this gate, exact source, installed environment/GPU/allocation,
parent prepared data, every measured technical family, parity, independent role
stress maxima, resource projections and output bytes. It cannot admit science
unless every technical check passes and projected requests fit debug24h. Reports
whose metric is poor still pass if technically valid. Missing/altered payloads,
wrong precision/backend, nonfinite timings, invalid support or RAM fail closed.

SPEC binds accepted preflight plus its exact submitted job and COMPLETED/0:0
accounting, the unchanged gate/source/parent, and all eight fits and both routes.
Every new fit starts fresh; parent weights are comparison evidence only. Only
the authenticated parent U000 TRAIN bank crosses the external supervision edge.

TASK binds spec/preflight, exact dependency receipt hashes (including external
reduce_U000), source, selected report/checkpoint or TRAIN-only bank, every output
file and exact worker job. Receipts publish last. Partial directories stop rather
than resume. Banks require exact identities, original prepared hash, selected
scientific report hash, node, role=train and T=2. No validation probabilities enter
KD. DIAGNOSTICS bind same checkpoint and ordered validation probabilities to full
and original50k metrics. RESULTS contain all nine parent and eight new rows,
input/capacity labels, missing rows explicitly unfinished, and paired baseline
recoveries in each respective population. All rows are required for completion.

PLAN binds complete argv/dependencies; external completed tasks have no scheduler
dependency. Reuse the exact-DAG dry/live ledger contracts, exclusive live claim
and durable journals. Only exact saved-plan-hash execution is authorized.

## CUDA_DIAGNOSTIC_* /v1 (separate, non-admitting artifacts)

`LUKA_ALLJETS_FUSION_CUDA_DIAGNOSTIC_{SPEC,PLAN,CASE,FAILURE,REPORT}/v1`
retains the exploratory/historical-test flags above. SPEC binds new clean pushed
source, original failed gate and runtime file hashes, immutable four-case policy,
one-hour/6CPU/256000MiB/A100 envelope and fresh output. Validate the original gate
and submission journal in the original source worktree, not by rewriting it.
Exact named source compatibility forbids changes to old model, cache, training,
preflight and science execution. Only the source allowlist and new diagnostic
module are permitted differences from the failed source.

Each CASE records the actual environment/GPU, longest-TRAIN lengths, original
prepared hash, full256/3-step stress, synchronous/asynchronous restore choice,
strict candidate parity (when applicable), offload counters, peak allocation and
durable operation trace. Runtime must match the failed job's environment/GPU;
launch blocking is set per case BEFORE process startup. No CUDA queries after a
failure; preserve traceback/operation metadata even with a poisoned context.
REPORT binds each fresh child exit code, completion and output/log hashes. An
all-pass report is still NOT resource or scientific admission. Candidate parity
failure prevents that child's stress, not the remaining diagnostic children.

No cache recreation, validation/test evaluation, scientific weights/banks, batch
or schedule changes, performance-driven model selection, full campaign queueing,
or production admission are exposed. No silent replay into an existing case
directory. Submission uses a separate exact plan, dry ledger, exclusive claim,
durable journal and authorization phrase. Partial submissions are not retried.

## Synchronous production recovery /v2

`LUKA_ALLJETS_FUSION_{GATE,SPEC,PREFLIGHT,PLAN,TASK,RESULTS,DIAGNOSTICS,SUBMISSION_CLAIM}/v2`
retain the scientific registrations and exploratory flags. GATE and SPEC add
`pair_storage` with `version=pair_saved_tensors_cpu_sync_v3` and
`h2d_non_blocking=false`; the remainder matches the old physical-storage policy.
PREFLIGHT echoes that policy; every paired FP32/BF16 parity report must record it.
Version1 evidence, plans or task receipts cannot satisfy a version2 dependency.
The old generic kernel/checkpoint/TRAIN-bank contracts are unchanged; their new
source/report/gate parent hashes bind execution provenance.

GATE/v2 parents bind active source, original failed gate and exact diagnostic
report content hash. `recovery` freezes original gate/runtime and diagnostic
spec/report file references, allowed named module additions/changes, per-file
before/after digests, and creation-time terminal accounting. Current source must
be clean, exact committed/pushed source; original sources are verified in their
own worktrees. Check both exact submission journals. Authenticate every diagnostic
output/log; both candidate cases require real CUDA offload counters, both strict
precisions and full256 stress. Reusing a diagnostic never means science admission.

Replacement preflight repeats all existing full-population technical checks using
the production synchronous subclass and same software/A100 as the diagnostic.
No execution-time global monkeypatch, microbatching, tiling, truncation, data
regeneration, checkpoint/teacher import beyond the registered U000 TRAIN bank,
or changes to batch/BN/schedule/precision/physics. Parity-only constructor scoping
must restore on failure. Checkpoint state keys and cold initialization remain
identical to v1. Ordinary single-HLT factories are unchanged.

Only exact one-job dry/live preflight submission is exposed by the recovery
helper. Science creation/submission is still separate and requires an admitted
v2 preflight and exact COMPLETED/0:0 accounting. Insufficient memory/runtime
headroom fails closed after preserving non-admitting measurement.json; the
unchanged conservative debug24h check may still block this campaign.

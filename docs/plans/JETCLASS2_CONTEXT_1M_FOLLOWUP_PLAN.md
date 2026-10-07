# Automatic Oscar CONTEXT_V1 1M science after the existing gate

User-authorized operational amendment, 2026-10-07. This supersedes only the
manual second approval in `JETCLASS2_CONTEXT_OSCAR_1M_DIRECT_COARSE_PLAN.md`.
The science remains frozen CONTEXT_V1, 1M train / 250k validation, DIRECT and
COARSE, nine fits, five reducers, aggregate/completion, with final test sealed.
Do not modify the original plan, active worktree, dataset or gate artifacts.

The submitted gate is source `9cebdbe133da20f00cc14ec4155bd627accb7e85`, hash
`f4ad62f5a85f83c4df3e5c43b1fac81380249c175ee3a90e9ab2d4826b68624e`, jobs
7083139 (authentication), 7083140 (foundation), 7083141 (GPU preflight).
The user-provided queue snapshot showed pending jobs, not completed admission.

An execution-only controller in a separate clean pushed checkout authenticates
the original gate, source, canonical command plan, live ledger and per-job
journal using the original scientific code. It waits for all three exact jobs
COMPLETED/0:0, with checked Oscar cluster, owner, account and full job names.
Foundation success alone does not admit science: native GPU preflight provides
the installed-Weaver acceptance and measured full-population resource profile.

After success, reauthenticate the binding; create/validate science through the
original pinned CLI; produce the full canonical dry plan; validate and record
its actual hash against the authorized envelope; submit with that hash using
the original exclusive journaled submitter. No second interactive prompt.
Envelope: Oscar gpu/default/norm-gpu, one L40S, six CPUs/180000 MiB for fits and
reducers, measured fit walltime <=48h and reducer <=24h. All original internal
afterok edges remain. No performance threshold, extra branch or rerun.

This is conditional submission, not premature registration of the sixteen
science jobs with guessed resources. A detached controller polls every 60s for
at most 14 days, remains outside the scientific source snapshot, and survives
SSH disconnects (not host failure or administrative process cleanup).
Existing live/ambiguous science submission is never retried automatically.

Local tests must cover binding/journal corruption, failure/missing accounting,
source changes, exact population/resource scope, gate completion without valid
artifacts, duplicate controllers, partial submission, canonical dry-before-live
ordering, environment isolation and the original-source subprocess boundary.
Real Oscar arming/submission remains a user action after clean publication.

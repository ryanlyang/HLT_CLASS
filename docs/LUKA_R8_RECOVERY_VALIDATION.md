# R8 pre-claim recovery validation

Incident evidence supplied by the user on2026-11-12: setup168256 and
release168257 completed, C3/168726 and R14/168727 failed at their first
`scontrol show job` query on gpu2201. No original claim or task output exists
for either row; controller records have been purged, while both controllers
answer ping. Captured stderr was not printed, so the exact original RPC error
is unknown. A transient scheduler/connectivity failure is an inference.

Implementation baseline6a98c2bda010779b71348ff2cfd07b89f8160fc3.
Scientific donor58d0c0edf5f41cff2398fb168fc16fc92e169294 remains entirely
unchanged. The new CLI loads the recovery module by its pinned file path,
while all scientific imports and original source checks execute from the
original worktree. No scientific runtime is copied, patched or monkeypatched
in production. Saved setup receipts, targets and measured requests are reused.

New files: `src/hlt_classification/luka_r8_recovery.py`, thin CLI, queue wrapper,
Slurm worker, focused tests, v1 recovery contract and runbook. The follow-up
plan gains an execution-recovery addendum; shared handoff/donor map retain
pre-existing edits. New SPEC/PLAN/CLAIM/TASK/RESULTS identities bind both sources
and original/replacement jobs, without rewriting either old job or its ledger.

Baseline focused release/registration tests:17 passed in9.42s. Initial new
recovery tests:23 passed in24.67s. Final combined recovery/release run had
50 passes and one reporting-fixture failure in25.74s: that fixture supplied
ten classes while this dataset has eleven. Corrected it to use the canonical
CLASS_NAMES and reran that case:1 passed in7.96s. Thus all51 distinct cases
pass across the combined and focused checks. No runtime change was needed
for that fixture error. Tests cover finite controller retries and stderr,
unchanged denial on validation failures, exact original terminal state and
absent claims, corrupted inputs, quota, durable three-job submission and partial
journal rejection, result merging and correct seed3 comparisons.

Two real synthetic30-epoch production fits (C3/R14) use the original native
training/export code with an installed Weaver0.5.3 compact configuration on
CPU. Both validate selected-checkpoint replay and prove original-directory byte
preservation. The two-worktree CLI test independently proves that the new driver
imports the old scientific package. These tests do not measure Brown runtime.
Environment: `C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe`,
PYTHONDONTWRITEBYTECODE=1, PYTHONPATH=src plus cached installed Weaver.
Python ASTs, CLI help, four Bash syntax checks (two documented blocks and
two scripts), relative links and scoped whitespace checks pass. Scientific
donor directories have no changes in this repair.

No remote job, commit or push performed in this repair step. Actual setup168256
already qualifies the unchanged science; replacement workers reauthenticate
its complete acceptance and exact environment/GPU before fitting. Next: scoped
commit/push and the [recovery runbook](LUKA_R8_RECOVERY_RUNBOOK.md). This submits
C3, R14 and a CPU report directly, with no new setup or waiting for manual
release. Real replacement completion and scientific results remain pending.

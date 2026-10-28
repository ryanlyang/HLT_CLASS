# Luka depth screen: reviewed parallel recovery v2

User-authorized 2026-10-28 operational correction to the serial execution section
of LUKA_SUPERVISION_DEPTH_PLAN.md. Its scientific registration is unchanged:
same 20 students, two OFFLINE teachers, populations, seeds, objectives, 30 epochs,
native HLT inference and full-validation selection. No additional CE variants.
This plan supersedes **only scheduling/receipt/retirement/restart ownership** for this
explicit recovery. Historical v1 specs, source and outputs remain immutable.

## Changeover

Dry-review exact original submission journal/accounting. Cancel only listed jobs
that are still PENDING, in reverse DAG order, holding and rechecking their exact
owner/account/name/comment/workdir/partition before scancel. A dispatch race is
preserved, never killed. Running jobs finish in their original allocation/source.
Do not create the recovery until ALL original jobs are terminal.

Authenticate each completed scientific task with its original validator and
COMPLETED/0:0 accounting. Targets and baseline must already be complete. Reuse
all other completed fits/teachers too. Remaining fits must be CANCELLED before
starting (elapsed zero, unknown start), with no output directory or symlink.
Except for the separately approved interrupted `set8_p3` restart below, any
partial/failed/ambiguous output stops recovery: no deletion, overwriting,
automatic retry, or silently restarting trained controls. One exclusive original
root claim prevents two corrected campaigns from writing the same task paths.

### Explicit interrupted-attempt amendment (2026-10-28)

The user cancelled original `set8_p3` job 21973955 after it started, then explicitly
authorized preserving the attempt and restarting only that fit later. New v2
recovery artifacts do not reinterpret v1 plans/specs. This exception is limited
to `set8_p3`, with its exact ID obtained from the authenticated original journal;
the code never accepts arbitrary failed jobs or uses AUC to decide restart/skip.

Before recovery creation, a separately hash-reviewed RESTART_PLAN verifies all
original jobs terminal, completed targets/baseline, original source/journal,
positive elapsed time and known start for CANCELLED set8_p3, no existing recovery
claim, and no completed checkpoint/report/fit/task receipt needing salvage.
It inventories every partial-task file/directory and retains the exact original
log hash. On explicit authorization only, write an exclusive durable intent and
rename the whole task directory to
`interrupted_attempts/set8_p3-<original-job-id>/output` inside the original root.
No file is deleted, overwritten, copied to home scratch, or loaded as a checkpoint.
Symlinks/junctions, hardlinked payloads, special files and cross-device moves are
forbidden. Recheck accounting and hashes immediately before moving; verify all
archived bytes afterward and publish RESTART. Interrupted archival fails closed
with intent intact; no automatic retry or claim stealing.

Recovery SPEC records this receipt, authenticates its archived files/log on reuse,
and registers one fresh fixed30 fit from epoch1 with identical seed/recipe. The
failed attempt is not a completed result and never participates in selection.
Only this restarted arm moves to the end of its existing GPU lane; the early
CE-diverse/duplicate/SupCon ordering and two-lane capacity bound remain unchanged.
Completed fits remain reused. With only targets/baseline complete, expect 22 new
jobs: 19 students (including set8_p3), two teachers and one report. No targets,
baseline, preflight, cleanup, extra scientific row, or new ladder is scheduled.

## Execution

Two independent GPU lanes, not a 25-job success chain. Early lane A starts
teacher_ce -> ce_diverse; lane B duplicate_offline -> teacher_supcon ->
supcon_diverse. Reused tasks disappear from the new queue. With baseline and
set8_p3 already complete, the CE-diverse experiment and duplicate control are the
next two student tests; exact chronological order depends on resource availability.

Only genuine teacher requirements use afterok. Same-lane capacity edges use
afterany, so unrelated technical failures do not block subsequent independent
experiments. Include all earlier same-lane jobs to retain the two-job bound even
when Slurm cancels a dependent job before its other parent terminates. Every fit
still authenticates its scientific parents. Metrics never control scheduling.
No old completed Slurm ID appears in new dependencies (avoid aged-out IDs).
The report waits afterany for all replacements and can publish explicit pending
rows for technical failures. No new standalone preflight or target-export job.

Same debug A100/CPU/memory/time requests and disposable per-fit checks. Canonical
node-local /tmp token scratch has process-locked conservative reservations;
old payload creation and retirement kernels are unchanged. Never fall back to
home storage. Source kernels remain byte-identical to original source; new module
boundary is 04d1f0f5945ab9af38d83e192eff48e124ad8dc5.

## Identity and storage

Shared targets/references and completed fits are reused **in place**, not copied
or regenerated. New fits write only absent registered task
directories under the original root, so unchanged bank paths/teacher coordinates
remain valid; the explicit archived set8_p3 attempt is retained separately. Their
scientific FIT/BUNDLE/TRAINING_REPORT identity is the original
spec; new RECOVERY_TASK receipts bind the separate recovery spec, exact new source,
accepted job ID, scientific parent receipts and exhaustive output hashes. New
receipts/logs/ledger/report live under the recovery root; no original TASK receipt
or ledger is relabelled. Use the recovery results CLI to combine both executions.

Retain borrowed shared banks and new teacher banks after reporting. No automatic
deletion of the original namespace; original cleanup job is cancelled with the
other pending jobs. This retains approximately 1 GiB compact targets instead of
attempting to retire borrowed provenance. Large per-job token scratch still
retires. Reserve eight GiB available headroom; datasets and Fresh_check untouched.

Acceptance: focused scheduling/identity/dispatch-race/failure tests plus unchanged
production training/publication and merged-result miniature. Local CUDA/Weaver
tests are not a new SPORC measurement. Source must be committed and pushed; both
cancellation and new full command plan require explicit hash-reviewed approval.

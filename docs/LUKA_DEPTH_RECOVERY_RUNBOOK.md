# Recover the depth screen into two debug lanes

No targets or completed controls are rerun. Running originals finish; only
pending originals are cancelled by the helper. A user-cancelled running set8_p3
can now be explicitly archived/restarted under v2, as below. This is the same 20-student study, not 20 more
fits. With targets/baseline/set8_p3 completed, expect 21 replacement jobs:
18 students, two teachers, one report. More completed work reduces that count.

## Commit locally

In PowerShell, review and stage only:

```powershell
git add -- src/hlt_classification/luka_depth_recovery scripts/luka_depth_recovery.py scripts/queue_luka_depth_recovery.sh sbatch/run_luka_depth_recovery.sh tests/test_luka_depth_recovery.py docs/plans/LUKA_DEPTH_RECOVERY_PLAN.md docs/contracts/LUKA_DEPTH_RECOVERY.md docs/LUKA_DEPTH_RECOVERY_RUNBOOK.md
git diff --cached --check
git diff --cached --stat
```

Review before committing/pushing. The new entries in shared HANDOFF.md and
LEGACY_SOURCE_MAP.md may be staged separately by hunk; those files contain
unrelated edits. Do not force-push or stage the whole worktree.

## Pinned SPORC worktree

```bash
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 1
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_depth_recovery_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main || exit 1
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}" || exit 1
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)" || exit 1
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}" || exit 1
fi
bash "${PROJECT}/scripts/queue_luka_depth_recovery.sh" stop
```

The helper prints a cancellation dry plan. Paste its hash and confirm. Only
pending exact `lukasd_*` IDs from the original ledger are affected. No wildcard
cancellation; no Fresh_check/tier3 jobs. If a job starts meanwhile it is preserved.
Do NOT release or requeue the cancelled original jobs.

## After the remaining original running jobs finish

```bash
bash "${PROJECT}/scripts/queue_luka_depth_recovery.sh" status
bash "${PROJECT}/scripts/queue_luka_depth_recovery.sh" queue
```

The queue helper defaults to the original
`checkpoints/luka_depth_9e90b7de_debug_r1/campaign_spec.json`. An alternate full
path may be passed as the second argument. Creation fails harmlessly before the
exclusive claim if originals are still active. It authenticates all completed
fits, reuses their files and targets, and refuses unreviewed partial-output restarts. Review
the printed full replacement plan, paste its hash and confirm. Submission detaches
into a logged process; inspect that log and do not submit a second time. No
preflight or target regeneration. `lukasr_*` jobs use two concurrent lanes;
pending capacity dependencies are expected, but no single success chain remains.

Keep the original campaign/source and the recovery root. The original command
cannot validate new execution receipts; merged results use:

```bash
# In the helper's atlas_kd_sporc environment with PYTHONPATH=$PROJECT/src:
python -u -B -s "${PROJECT}/scripts/luka_depth_recovery.py" results \
  --spec "/home/ryreu/atlas/HLT_Classification/checkpoints/luka_depth_recovery_${COMMIT:0:8}_debug_r1/campaign_spec.json"
```

The report retains compact targets (~1 GiB), models, predictions and histories;
only per-job node-local token payloads retire. No borrowed file deletion. Inspect
any interrupted claim/journal instead of deleting it or blindly retrying.

## If you cancelled set8_p3 after it started (v2)

The previous failure during inventory did not create a recovery claim or submit
new jobs. Keep that old recovery worktree/root and every original output. Commit
and push the v2 fix, then use a NEW source-pinned recovery worktree and root; do
not edit or retry the old 8a80162f source. The pinned-worktree block above applies,
but use `status` instead of `stop` if you already cancelled all pending originals.

From the new worktree, run:

```bash
bash "${PROJECT}/scripts/queue_luka_depth_recovery.sh" restart-set8-p3
bash "${PROJECT}/scripts/queue_luka_depth_recovery.sh" queue
```

The first command prints a read-only restart plan. Review the exact source path,
original job ID, archived path, file hashes and log. Paste its 64-character
PLAN_HASH **value** and type YES. This moves only the interrupted task directory
to `interrupted_attempts/set8_p3-<job-id>/output` within the original campaign;
it retains its log in place and writes an archive receipt. Nothing is deleted,
no model is resumed, and no jobs are submitted. Do not run this archive step twice.
Successful archival needs no repetition if subsequent queue creation is blocked.

The second command creates the v2 spec and requires a separate full submission
dry-run/hash/YES review. It queues all remaining work in two debug lanes, with
set8_p3 restarted from epoch1 at the **end of lane A**. Completed targets/baseline
and any other finished fits stay reused. If only targets/baseline finished,
expect 22 jobs: 19 remaining students, two teachers and a report. There is no
new shared-target job or preflight. The interrupted row is not silently omitted.

Any existing archive intent without its receipt, revived original job, changed
partial file/log, selected checkpoint/fit receipt, symlink or ambiguous output
stops the operation. Preserve the evidence and inspect; do not delete directories,
reset claims, release old jobs or blindly retry. Old node-local scratch is not
touched. New token fits still rebuild their own job-local token arrays as before;
that is distinct from the shared compact targets being reused.

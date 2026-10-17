# FullStat Oscar: sparse recovery, then automatic science

This is for the original `luka_fullstat_3m500k_20e_592a6dac_r1` gate ONLY.
Do not recreate its foundation, cache or source worktree. Do not reupload data.
Authority: the recovery addendum in [the plan](plans/LUKA_FULLSTAT_OSCAR20_PLAN.md)
and [the contracts](contracts/LUKA_FULLSTAT_OSCAR20.md).

The observed successful foundation and13 preparation tasks are reused. Only
prepare12,13,14 are retried (array12-14%3, excluding node1804), followed by new
assembly and GPU preflight. After ALL pass and their artifacts validate, the
controller submits all16 science jobs: controls, DIRECT, COARSE, banks and
reporting. Same3M/500k split, batch128 and20 epochs. The1.5M test remains sealed.

`CANCELLED by 0` records an administrative cancellation. The saved
`JobArrayTaskLimit` is a scheduling reason, not proof of why it was cancelled.
Excluding the common node is a precaution, not a proven root-cause diagnosis.

## 1. Commit/push locally

In Windows PowerShell at the repository root:

```powershell
git add -- src/hlt_classification/luka_fullstat/campaign.py `
    src/hlt_classification/luka_fullstat/submission.py `
    src/hlt_classification/luka_fullstat/worker.py `
    src/hlt_classification/luka_fullstat/recovery.py `
    scripts/luka_fullstat.py scripts/luka_fullstat_recovery.py `
    scripts/queue_luka_fullstat_recovery.sh tests/test_luka_fullstat_recovery.py `
    docs/plans/LUKA_FULLSTAT_OSCAR20_PLAN.md docs/contracts/LUKA_FULLSTAT_OSCAR20.md `
    docs/LUKA_FULLSTAT_OSCAR20_RUNBOOK.md docs/LUKA_FULLSTAT_RECOVERY_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git diff --cached --check
```

Stage only this task's sections of the two shared documents; preserve other
work. Review the staged diff, then:

```powershell
git commit -m 'Recover FullStat preparation and gate automatic Oscar science'
if ($LASTEXITCODE -ne 0) { throw 'Inspect git status and log before continuing; changes may already be committed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

No runtime changes outside the reviewed recovery files may be included since
parentbf04bb89. The compatibility check deliberately stops on unexpected source
changes instead of claiming the cached views remain interchangeable.

## 2. Oscar: get the pinned recovery source and review

On an Oscar login node, run this block. The repository is inside `source/`.
The old source592a6dac must still be present and clean for authentication.

```bash
read -r -p 'Pushed 40-character recovery commit: ' COMMIT
export COMMIT
export MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
export PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_recovery_${COMMIT:0:8}"
export ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_recovery_${COMMIT:0:8}_r1"
export ORIGINAL=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_3m500k_20e_592a6dac_r1/gate_spec.json
(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
checkquota
bash "${PROJECT}/scripts/queue_luka_fullstat_recovery.sh" review \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}"
)
```

Preserve the existing cache-space reservation (the original conservative total
bound was237.75GiB, not additional storage). Review actual quota, not `df`'s
filesystem-wide capacity. No automatic deletion or quota repair is performed.

This authenticates source, original submission journal, scalar foundation and
thirteen receipt metadata records. Full physical block authentication occurs
in assembly and preflight. Review the sparse recovery DAG and ALL16 future
science commands. The latter show48h ceilings, not measured requests; actual
requests must pass the replacement GPU gate and may only shrink within scope.
The dry helper runs `sbatch --test-only`, not real submission.

Copy the final **REVIEW_HASH** (combined recovery + follow-up review), not the
gate hash or its standalone plan hash. An existing unchanged recovery review
can be reopened; never delete it to bypass an error.

## 3. Arm once, then disconnect

Using the variables set above:

```bash
read -r -p 'Reviewed REVIEW_HASH: ' REVIEW_HASH
bash "${PROJECT}/scripts/queue_luka_fullstat_recovery.sh" arm \
    "${COMMIT}" "${ROOT}" "${REVIEW_HASH}"
```

Type YES at the helper's prompt. It launches a background controller, prints
its PID/log and the exact `tail -f` command. **Wait until that log says ARMED.**
A PID alone does not establish successful submission or arming. Ctrl-C exits
tail only; nohup handles normal SSH disconnects. Login-node restarts or process
termination can still stop the controller. Do not also submit science manually.

When successful, the log will end with `ALL 16 SCIENCE JOBS QUEUED` and exact IDs.
Inspect `followup/submitted.json` for the authenticated submission record.
This means queued, not necessarily running or completed.

Any failed prepare/assembly/preflight, source or artifact failure, resource
failure, or unapproved science plan stops the controller **without submitting
science**. Missing accounting waits at most14 days. A Slurm submission itself
can fail partway through a DAG; already submitted jobs are preserved and their
journal must be inspected, never blindly resubmitted. Gate failures are never
automatically retried. No unrelated jobs are changed.

The old root gains only `fullstat_recovery_claim/intent.json`; it prevents a
second recovery root from writing to the same cache. Do not delete claims or
journals. A clean restart of the same controller is safe when complete ledgers
exist; ambiguous partial submission is explicitly blocked.

## Progress and results

```bash
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
sacct -u "$USER" -S now-3days -X -n -P -o JobID,JobName%60,State,ExitCode,Elapsed
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

The results command becomes available after successful science admission and
can display completed models while later steps are pending. All outputs of the
new science campaign are in the recovery root; the shared prepared cache and
original foundation stay at their original locations.

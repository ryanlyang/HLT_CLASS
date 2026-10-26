# Finalize saved DE033, then train DE000

Job7938792 finished its twenty epochs and both TRAIN exports. Its final
authentication subprocess could not start in the damaged environment, so no
TASK receipt was published. This is not a restart of its optimizer or epochs.
Authority: [finalization contract](contracts/LUKA_FULLSTAT_SPLITHEAD_FINALIZE.md)
and unchanged [scientific plan](plans/LUKA_FULLSTAT_SPLITHEAD_PLAN.md).

Exactly two jobs: finalize_DE033 -> train_DE000. The finalizer rechecks native
validation probabilities and existing TRAIN targets; it does not train or
export. Only DE000 runs a fresh twenty-epoch fit. No separate preflight.
All original artifacts/worktrees and environments remain in place. Never
remove claims, edit old specs or repeat an uncertain live submission.

Both new jobs explicitly activate the user-verified replacement:
/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar.
This is still scratch, not persistent storage. The cause of the earlier file
loss has not been established. A matching environment fingerprint is required
at creation and execution; changed or missing files stop the workflow.

## 1. Commit only this recovery locally

Do not use git add .: there is unrelated relational-ladder work in this tree.
Its source boundary is separate. Inspect the staged diff before committing.
HANDOFF and LEGACY_SOURCE_MAP also contain unrelated sections; stage only the
new finalization sections with git add -p if including them.

```powershell
git add -- src/hlt_classification/luka_fullstat_splithead/campaign.py src/hlt_classification/luka_fullstat_splithead/worker.py src/hlt_classification/luka_fullstat_splithead/submission.py src/hlt_classification/luka_fullstat_splithead/kd_banks.py src/hlt_classification/luka_fullstat_splithead/finalize.py src/hlt_classification/luka_fullstat_splithead/finalize_source.py scripts/luka_fullstat_splithead_finalize.py scripts/queue_luka_fullstat_splithead_finalize.sh sbatch/run_luka_fullstat_splithead_finalize.sh sbatch/luka_splithead_finalize_common.sh tests/test_luka_fullstat_splithead_finalize.py docs/contracts/LUKA_FULLSTAT_SPLITHEAD_FINALIZE.md docs/LUKA_FULLSTAT_SPLITHEAD_FINALIZE_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --stat
git commit -m "Finalize saved DE033 using verified replacement runtime"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

## 2. Review and submit on Oscar

The default below is four GiB of PERSONAL available home quota, not total
filesystem space. Confirm checkquota supports it, or enter another integer
at least two. Existing banks are referenced in place, not copied. Old
campaigns retain their original sixteen-GiB minimum.

Run in an interactive Oscar shell. Prompts read /dev/tty, not Python's script
stdin. Stay connected until both job IDs are printed; if interrupted, inspect
the submission journals before attempting another execution.

```bash
(
set -euo pipefail
read -r -p 'Pushed finalization commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="${MAIN}_splithead_finalize_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_splithead_finalize_${COMMIT:0:8}_r1"
OLD=/oscar/home/rlyang/hlt_classification/checkpoints/luka_splithead_recovery_0c4a71a5_r1/campaign_spec.json
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" cat-file -e "${COMMIT}^{commit}"
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
checkquota
read -r -p 'Personally available home-result quota GiB (e.g. 4; minimum 2): ' QUOTA </dev/tty
HELPER="${PROJECT}/scripts/queue_luka_fullstat_splithead_finalize.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${OLD}" "${QUOTA}"
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Authorize finalizing saved DE033, then training DE000? Type YES: ' CONFIRM </dev/tty
test "${CONFIRM}" = YES
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${OLD}" "${QUOTA}" --execute "${PLAN_HASH}"
printf '\nFinalization root: %s\n' "${ROOT}"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
)
```

Review jobs=2, scientific_fits=1, preflight_jobs=0. Names must be
lukafssf_finalize_DE033 and lukafssf_train_DE000. DE000 has afterok on the
new finalizer, not on the failed old job. The finalizer inherits the old
DE033 walltime ceiling for safety; this is not an estimate of its runtime.
It does full validation readback, TRAIN joins and source/artifact checks,
but never repeats the seven-hour training. Scheduler waiting is separate.

## Progress and results

Logs are finalize_DE033-JOBID.out and train_DE000-JOBID.out in the printed
root. The former logs readback/join phases, not epoch training. DE000 logs its
normal twenty epochs. Once submitted, no login-node follow-up controller is
needed. DE000 cannot start if finalization fails.

To read saved validation results after setting PROJECT_DIR and ROOT to the
new pinned project/output paths:

```bash
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_splithead_finalize.py" results --spec "${ROOT}/campaign_spec.json"
```

Results reuse original selected DE066/DE033 reports and authenticated native
predictions, plus the new DE000. No test inference or checkpoint reselection.
Preserve the old failed root, first recovery, teacher banks, source worktrees
and replacement runtime until this closure and its results are verified.

# Recover saved DE066 without retraining it

This resumes the publication workflow after job 7910169, not its optimizer.
Same scientific plan; [recovery contract](contracts/LUKA_FULLSTAT_SPLITHEAD_RECOVERY.md).
No standalone preflight. Queue all three tasks once: recovery -> DE033 -> DE000.
Old results are preserved. The old source worktree and original artifacts must remain.

## Commit and push locally

```powershell
git add -- src/hlt_classification/luka_fullstat_splithead/training.py src/hlt_classification/luka_fullstat_splithead/worker.py src/hlt_classification/luka_fullstat_splithead/campaign.py src/hlt_classification/luka_fullstat_splithead/submission.py src/hlt_classification/luka_fullstat_splithead/kd_banks.py src/hlt_classification/luka_fullstat_splithead/recovery.py src/hlt_classification/luka_fullstat_splithead/recovery_source.py scripts/luka_fullstat_splithead_recovery.py scripts/queue_luka_fullstat_splithead_recovery.sh sbatch/run_luka_fullstat_splithead_recovery.sh tests/test_luka_fullstat_splithead_recovery.py docs/contracts/LUKA_FULLSTAT_SPLITHEAD_RECOVERY.md docs/LUKA_FULLSTAT_SPLITHEAD_RECOVERY_RUNBOOK.md
git diff --cached --stat
git commit -m "Recover published FullStat split-head fit after JSON ordering failure"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

HANDOFF/LEGACY_SOURCE_MAP have pre-existing edits; stage their relevant sections
selectively if desired. Do not use git add . or alter the old pinned source.

## Review, then submit on Oscar

Run checkquota first. Enter 16 only if at least 16 GiB is personally available.
This block does not submit until the exact plan hash and YES are supplied.

```bash
(
set -euo pipefail
read -r -p 'Pushed recovery commit (40 characters): ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="${MAIN}_splithead_recovery_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_splithead_recovery_${COMMIT:0:8}_r1"
OLD=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_splithead_a7efcef6_r1/campaign_spec.json
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
read -r -p 'Personally available result quota in GiB (at least 16): ' QUOTA </dev/tty
HELPER="${PROJECT}/scripts/queue_luka_fullstat_splithead_recovery.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${OLD}" "${QUOTA}"
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Authorize export-only DE066 recovery plus DE033 and DE000? Type YES: ' CONFIRM </dev/tty
test "${CONFIRM}" = YES
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${OLD}" "${QUOTA}" --execute "${PLAN_HASH}"
printf '\nRecovery root: %s\n' "${ROOT}"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
)
```

Review three jobs, two scientific fits and no preflight. First job must be named
lukafssr_recover_DE066 (not train_DE066). Its inherited walltime is an upper bound,
not an estimate that it will repeat training. The other two stages retain their
full normal training time. Jobs queue with afterok dependencies; no manual follow-up.
Do not repeat an uncertain live submission or remove claims; inspect journals first.

If accounting reports running/pending/nonzero-runtime descendants, stop and inspect
the exact original IDs. This helper does not cancel jobs. If report/payload/source
authentication fails, preserve everything and diagnose before proceeding.

## Progress / results

New logs are recover_DE066-JOBID.out, train_DE033-JOBID.out, train_DE000-JOBID.out
under the printed root. DE066 logs validation readback and TRAIN exports, no epochs.
After sourcing the same project's sbatch/jetclass2_delphes_common.sh with
JC2_SITE=oscar_l40s and CUBLAS_WORKSPACE_CONFIG=:4096:8, use:

```bash
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_splithead_recovery.py" results --spec "${ROOT}/campaign_spec.json"
```

Saved validation metrics, recovery and R30/R50 only; results do not infer on test.
Metadata, fit refs and successful exact job accounting remain required.

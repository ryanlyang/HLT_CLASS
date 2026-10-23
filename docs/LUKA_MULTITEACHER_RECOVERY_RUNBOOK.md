# Recover two LOGIT teachers, then run all five trials

This is the separately authorized [recovery](plans/LUKA_MULTITEACHER_RECOVERY_PLAN.md),
not a rerun of the failed original queue helper. Contracts:
[LUKA_MULTITEACHER_RECOVERY/v1](contracts/LUKA_MULTITEACHER_RECOVERY.md).
All eight jobs use SPORC tier3; no new preflight and no later manual submission.
Existing OFFLINE/U100 are reused. D066/D033 are NEW fits and retained afterward.

## Commit and push locally

Stage ONLY the new recovery namespace, three scripts, test and three documents
plus the reviewed recovery additions in HANDOFF/LEGACY_SOURCE_MAP. Existing
unrelated edits belong to the user. Never use `git add .` or force-push.
If push is rejected, preserve the commit and reconcile remote changes first.
The new source guard requires all recovery files tracked and old `src` unchanged
relative to01d45d6057e799b4e262f8c590b51706803ebfd1.

In local PowerShell at the repository root:

```powershell
git add -- src/hlt_classification/luka_multiteacher_recovery scripts/luka_multiteacher_recovery.py scripts/queue_luka_multiteacher_recovery.sh sbatch/run_luka_multiteacher_recovery.sh tests/test_luka_multiteacher_recovery.py docs/plans/LUKA_MULTITEACHER_RECOVERY_PLAN.md docs/contracts/LUKA_MULTITEACHER_RECOVERY.md docs/LUKA_MULTITEACHER_RECOVERY_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --stat
```

Review the staged paths. HANDOFF and LEGACY_SOURCE_MAP already contained other
uncommitted work; optionally stage only the new recovery sections with
`git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md`. Do not blindly include
their entire accumulated histories or any unrelated staged changes.

```powershell
git commit -m "Recover two Luka LOGIT teachers before five tier3 trials"
if ($LASTEXITCODE -ne 0) { throw "Commit failed; inspect status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed; do not force-push" }
git rev-parse HEAD
```

## Queue on sporcsubmit

Use the NEW pushed40-character commit, not01d45d6. This creates a fresh pinned
worktree; it does not alter the main checkout, old studies or Fresh_check.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the NEW pushed40-character recovery commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_multi5_recovery_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_multiteacher_recovery.sh"
BASH
```

Reserve at least5 personally available GiB for the ENTIRE campaign, paste the
printed PLAN_HASH value, then YES. Authentication/block hashing may take several
minutes before the dry plan appears. The queue helper checks surviving weights,
not the deleted D066/D033 paths. It replays retained original-source evidence;
do not delete those old metadata/results or original worktrees.

Watch the printed submission log until all EIGHT IDs are accepted. A helper PID
alone is not confirmation. The dependent jobs are submitted immediately and
wait in Slurm: D066 -> D033 -> embedding export -> five student arms.
No login-resident polling controller. Failed parents prevent children running;
never manually resubmit into a partial root or remove a live claim.

```bash
squeue --me -o "%.18i %.12P %.38j %.2t %.10M %R"
```

Jobs use `lukamr_` names. The two teacher logs show the original
`LUKA attention node=LOGIT_D066/LOGIT_D033 epoch=...` prefix: this is the original
trainer, but attention loss weight is ZERO. No large attention targets are saved.
Student logs show MULTI5 and per-epoch validation AUC/losses.

## Results

Activate the same atlas_kd_sporc environment and exports as the queue helper.
Run the new CLI in its pinned worktree (substitute the printed study path):

```bash
python -u -B -s "${PROJECT}/scripts/luka_multiteacher_recovery.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

Partial results work after creation and as each teacher/student completes.
NEW_LOGIT rows are replacements, never relabeled historical results. Original50k
is a report-only subset, not another checkpoint selection or independent test.
Native HLT inference excludes auxiliary heads; oracle teachers are not deployable.

Keep both replacement selected.pt files, surviving OFFLINE/U100, all student
selected models, receipts and results. No deletion occurs automatically. The
compact banks fit inside the5GiB reservation and support future reuse; any
later retirement needs a separately reviewed consumer inventory.

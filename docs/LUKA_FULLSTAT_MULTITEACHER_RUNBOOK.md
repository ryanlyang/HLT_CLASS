# Queue one four-teacher FullStat model on Oscar

See the [plan](plans/LUKA_FULLSTAT_MULTITEACHER_PLAN.md) and
[contract](contracts/LUKA_FULLSTAT_MULTITEACHER.md). One new fit only; reuse the
completed U100-w5 result. Existing campaigns, models and target banks are read-only.
The setup job automatically releases training only after verified technical
success. There is no login-node controller to keep alive.

## 1. Commit only this implementation on Windows

Preserve unrelated changes in HANDOFF/LEGACY_SOURCE_MAP; stage their new sections
selectively if desired. Do not use `git add .`.

```powershell
git add -- src/hlt_classification/luka_fullstat_multiteacher scripts/luka_fullstat_multiteacher.py scripts/queue_luka_fullstat_multiteacher.sh sbatch/run_luka_fullstat_multiteacher.sh tests/test_luka_fullstat_multiteacher.py docs/plans/LUKA_FULLSTAT_MULTITEACHER_PLAN.md docs/contracts/LUKA_FULLSTAT_MULTITEACHER.md docs/LUKA_FULLSTAT_MULTITEACHER_RUNBOOK.md
git diff --cached --stat
git commit -m "Add FullStat four-teacher embedding prediction campaign"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

## 2. Review on Oscar

Paste the pushed 40-character commit when prompted. These commands create a fresh
detached worktree and new immutable study; they do not submit jobs. The prior
U100-w5 run and its original worktree/banks must still exist. `checkquota` must
show at least 12 GiB personally available durable headroom before declaring 12.
Changing paths does not bypass authentication.

```bash
(
set -euo pipefail
read -r -p 'Pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="${MAIN}_fullstat_multiteacher_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiteacher_${COMMIT:0:8}_r1"
PREVIOUS=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_u100_embedding_w5_7ea8b0c6_r1/campaign_spec.json
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
bash "${PROJECT}/scripts/queue_luka_fullstat_multiteacher.sh" "${COMMIT}" "${ROOT}" "${PREVIOUS}" 12
printf '\nStudy: %s\nProject: %s\n' "${ROOT}" "${PROJECT}"
)
```

## 3. Submit the reviewed two-job graph

Only run after reviewing the exact two commands and their resource requests.
Use the same commit; paste the printed `PLAN_HASH`. This is explicit authorization
for setup plus ONE student fit. Do not repeat after an uncertain submission.

```bash
(
set -euo pipefail
read -r -p 'Same pushed commit: ' COMMIT </dev/tty
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_multiteacher_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiteacher_${COMMIT:0:8}_r1"
PREVIOUS=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_u100_embedding_w5_7ea8b0c6_r1/campaign_spec.json
read -r -p 'Authorize setup and one conditional 20-epoch fit? Type YES: ' CONFIRM </dev/tty
test "${CONFIRM}" = YES
bash "${PROJECT}/scripts/queue_luka_fullstat_multiteacher.sh" "${COMMIT}" "${ROOT}" "${PREVIOUS}" 12 --execute "${PLAN_HASH}"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
)
```

Expect `lukafsm_setup` and dependency-pending `lukafsm_train_multi4`. The latter
is submitted now, but cannot start successfully without setup admission. A
setup failure cancels its invalid dependency; no science should be resubmitted
manually. Preserve partial outputs/journals and inspect the error. No test job.

## Results and progress

In the study root, `train_multi4-JOBID.out` prints epoch AUC, best epoch, all four
auxiliary losses and progress every 128 batches. Estimated remaining time excludes
validation/checkpointing. Final results are immutable under `train_multi4/`.

```bash
(
set -euo pipefail
read -r -p 'Execution commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
export PROJECT_DIR="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_multiteacher_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiteacher_${COMMIT:0:8}_r1"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_multiteacher.py" results --spec "${ROOT}/campaign_spec.json"
)
```

Results authenticate saved selected validation predictions; no new inference or
test access. The comparison labels M0HLT as historical, not a freshly rerun CE
control. A full local suite and installed-Weaver miniature are recorded in HANDOFF;
Oscar resource acceptance remains the purpose of the queued setup job.

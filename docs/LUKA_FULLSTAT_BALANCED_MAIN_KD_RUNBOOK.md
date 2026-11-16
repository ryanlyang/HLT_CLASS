# Oscar balanced DE000: main-head KD

One fresh 20-epoch HLT-only DE000. Main loss is 0.25 CE + 0.75 DE050 KD at
T2; the four embedding/auxiliary KD heads and OFFLINE contrastive loss remain
unchanged. See the [plan](plans/LUKA_FULLSTAT_BALANCED_MAIN_KD_PLAN.md).

## Launch

First commit and push the complete implementation (source, tests, entrypoints
and documentation) to `origin/main`. Preserve unrelated work when staging.
Stage only this campaign's files, review already staged changes, and use
`git add -p` for its additions to the shared README/HANDOFF/plan index/source
map; this workspace also contains other work.

```powershell
git add -- src/hlt_classification/luka_fullstat_balanced_main_kd scripts/luka_fullstat_balanced_main_kd.py scripts/launch_luka_fullstat_balanced_main_kd.sh scripts/queue_luka_fullstat_balanced_main_kd.sh sbatch/run_luka_fullstat_balanced_main_kd.sh tests/test_luka_fullstat_balanced_main_kd.py tests/test_luka_fullstat_balanced_main_kd_workflow.py tests/test_luka_fullstat_balanced_main_kd_launch.py docs/plans/LUKA_FULLSTAT_BALANCED_MAIN_KD_PLAN.md docs/contracts/LUKA_FULLSTAT_BALANCED_MAIN_KD.md docs/LUKA_FULLSTAT_BALANCED_MAIN_KD_RUNBOOK.md docs/LUKA_FULLSTAT_BALANCED_MAIN_KD_VALIDATION.md
git diff --cached --stat
git commit -m "Add balanced DE000 with direct DE050 main-head KD"
git push origin main
git rev-parse HEAD
```

On Oscar paste this block and enter that pushed full commit and current HOME
headroom in whole GiB. Five original balanced references must be completed;
the original DE000 need not be complete. No preflight job is submitted.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_balanced_ff22b20f"
checkquota
read -r -p 'Pushed 40-character main-KD commit: ' COMMIT </dev/tty
read -r -p 'Available HOME quota GiB (minimum 8): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
(( QUOTA >= 8 ))
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/source/HLT_Classification_balanced_main_kd_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_balanced_main_kd_${COMMIT:0:8}_r1"
PARENT="${BASE}/checkpoints/luka_fullstat_balanced_ff22b20f_r1/campaign_spec.json"
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_balanced_main_kd.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_BALANCED_MAIN_KD_AUTOSUBMIT_V1
BASH
```

After `DETACHED` and the log path are printed, the launcher continues across
SSH disconnects. Ctrl-C stops only the attached tail. Submission is confirmed
by `LAUNCHER_EXIT=0`, `.launch/complete.json` and the saved job ID. Do not repeat
the launcher if its status is uncertain; inspect its failure record and journals.

## Progress and results

In a fresh shell set PROJECT_DIR and ROOT to the paths printed by the launch:

```bash
export PROJECT_DIR=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_balanced_main_kd_COMMIT8
ROOT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_balanced_main_kd_COMMIT8_r1
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_balanced_main_kd.py" \
    progress --spec "${ROOT}/campaign_spec.json" --stage DE000
```

Progress reads epoch report metadata, reports AUC/accuracy and balanced-anchor
recovery, and includes the original balanced DE000 once published. It performs
no inference and does not open test data. Replace `progress ... --stage DE000`
with `results --spec "${ROOT}/campaign_spec.json"` after publication for selected
metrics, R50/R30 and per-population recovery. Nonpositive accuracy gaps yield
`n/a`/null rather than misleading percentages.

## Recovery

For a confirmed terminal failed job, keep all artifacts/claims. Use a fresh
root (for example `_r2`), the same pushed source commit, and pass this new
experiment's failed `campaign_spec.json` as PARENT with `--recover` appended to
the detached launch command. The registrar authenticates terminal accounting
and committed epochs. Never pass the original balanced ladder as a recovery
parent. Full 20-epoch histories finalize without training; incomplete histories
continue from the last committed epoch. A genuinely ambiguous submission or
runtime/source change needs inspection rather than an automatic retry.

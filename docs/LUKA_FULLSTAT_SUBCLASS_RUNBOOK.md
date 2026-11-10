# Queue the FullStat subclass experiment on Oscar

One fresh HLT-only fit on3M TRAIN/500k validation for20 epochs. Preserve
U100/CONCAT/FUSION embedding prediction and pure-OFFLINE contrastive matching;
add OFFLINE soft subclasses. No new teacher or separate preflight job.
See the [plan](plans/LUKA_FULLSTAT_SUBCLASS_PLAN.md),
[contract](contracts/LUKA_FULLSTAT_SUBCLASS.md) and
[validation evidence](LUKA_FULLSTAT_SUBCLASS_VALIDATION.md).

## Commit and push locally

Execution ancestry is `d88eb82c6569312b72a3a3bccb1c39f9e357b4a4`. Its separate
complementary-teacher Hydra additions leave the Oscar donors unchanged. Use a scoped
commit: existing unrelated Hydra edits are outside this campaign. Review shared
README/HANDOFF/source-map additions separately; do not overwrite others' work.

```powershell
git add -- src/hlt_classification/luka_fullstat_subclass scripts/luka_fullstat_subclass.py scripts/launch_luka_fullstat_subclass.sh sbatch/run_luka_fullstat_subclass.sh tests/test_luka_fullstat_subclass.py tests/test_luka_fullstat_subclass_targets.py tests/test_luka_fullstat_subclass_lifecycle.py docs/plans/LUKA_FULLSTAT_SUBCLASS_PLAN.md docs/contracts/LUKA_FULLSTAT_SUBCLASS.md docs/LUKA_FULLSTAT_SUBCLASS_RUNBOOK.md docs/LUKA_FULLSTAT_SUBCLASS_VALIDATION.md
git diff --cached --stat
git commit -m "Add Oscar FullStat soft subclass supervision"
git push origin main
git rev-parse HEAD
```

## Paste once on Oscar

The completed historical Multi3 contrastive job8361672 is the parent. Enter
the pushed40-character commit and current available HOME quota (minimum8GiB).
The launcher authenticates the parent, saves a scheduler dry review and submits
one job in the background. It does not change old artifacts or running jobs.

```bash
bash <<'BASH'
set -euo pipefail
checkquota
read -r -p 'Pushed 40-character subclass commit: ' COMMIT </dev/tty
read -r -p 'Available HOME quota GiB (minimum 8): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 8 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_multi3_contrastive_a7249f49"
PARENT="${BASE}/checkpoints/luka_fullstat_multi3_contrastive_a7249f49_r1/campaign_spec.json"
PROJECT="${BASE}/source/HLT_Classification_subclass_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_subclass_${COMMIT:0:8}_r1"
test -r "${PARENT}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_subclass.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_SUBCLASS_AUTOSUBMIT_V1
BASH
```

After **DETACHED**, SSH may disconnect: the tail is only a view. Submission is
confirmed by **LAUNCH_COMPLETE** and **LAUNCHER_EXIT=0**, not DETACHED alone.
Do not launch twice. The log path is printed before tailing.

The single `lukafssc_train_subclass` job requests one L40S,6CPU,192000MiB,48h.
48h is a ceiling, not an ETA. Targets are fitted once in that allocation;
training uses cached probabilities and a small auxiliary head. Actual complete
update timing and memory admission run before the fresh scientific fit. Poor
finite metrics/cluster occupancy do not stop it; technical/resource failures do.

## Progress and completed results

Set ROOT and PROJECT to the exact paths printed by the launcher:

```bash
squeue --me -o '%.18i %.45j %.2t %.10M %.12l %R'
grep -E '^SUBCLASS (targets|epoch=|progress|measured)' "${ROOT}"/train_subclass-*.out | tail -n 35
```

After completion, saved results include all signal classes at R50/R30, accuracy,
AUC and gap recovery, compared with M0HLT/OFFLINE/U100-w5/Multi3/Multi3+contrastive:

```bash
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_subclass.py" \
    results --spec "${ROOT}/campaign_spec.json"
```

No new inference or final-test evaluation occurs in `results`.

## Explicit recovery

For a terminal failed/cancelled/timed-out exact job with completed epochs, keep
the same commit/worktree and parent; choose a fresh ROOT and append
`--resume-spec /absolute/failed/root/campaign_spec.json` to the launcher.
The original targets and epoch states are verified and copied; target fitting
is skipped and optimizer/RNG/schedule resume at the completed epoch boundary.
With20 completed epochs, only publication follows integrated technical checks.
Failures before the first epoch or requiring changed code need inspection.
Do not delete claims or retry an ambiguous submission.

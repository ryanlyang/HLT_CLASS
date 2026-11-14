# Queue the balanced FullStat ladder on Oscar

Six fresh20-epoch fits: OFFLINE, M0HLT, CONCAT, UE100, DE050, DE000.
Every model uses equal class/conditional-pT sampling over200–2000GeV, with
2,425,536 draws per epoch. The final HLT student adds OFFLINE contrastive0.3.
[Science](plans/LUKA_FULLSTAT_BALANCED_LADDER_PLAN.md),
[contract](contracts/LUKA_FULLSTAT_BALANCED_LADDER.md),
[validation](LUKA_FULLSTAT_BALANCED_LADDER_VALIDATION.md).

There is no separate preflight job. Each real allocation must pass its integrated
checks before fitting. Local readiness is not evidence that Oscar acceptance ran.
The detached launcher completes authentication, dry review and submission without
an attached SSH terminal. First three jobs are independent; Slurm resource/QOS
limits may serialize them. All later jobs depend on their completed teachers.

## Commit and push locally

The workspace contains unrelated work. Stage this campaign explicitly; inspect
already staged changes and shared-document hunks before committing. Do not use
`git add .` or reset other work. Runtime baseline used for adaptation is
`1094dea4ccfb36d08e29f740a97d90fc770335ec`.

```powershell
git add -- src/hlt_classification/luka_fullstat_balanced_ladder scripts/luka_fullstat_balanced_ladder.py scripts/launch_luka_fullstat_balanced_ladder.sh scripts/queue_luka_fullstat_balanced_ladder.sh sbatch/run_luka_fullstat_balanced_ladder.sh tests/test_luka_fullstat_balanced_ladder.py tests/test_luka_fullstat_balanced_ladder_workflow.py tests/test_luka_fullstat_balanced_ladder_launch.py docs/plans/LUKA_FULLSTAT_BALANCED_LADDER_PLAN.md docs/contracts/LUKA_FULLSTAT_BALANCED_LADDER.md docs/LUKA_FULLSTAT_BALANCED_LADDER_RUNBOOK.md docs/LUKA_FULLSTAT_BALANCED_LADDER_VALIDATION.md
git diff --cached --stat
git commit -m "Add Oscar balanced FullStat six-stage ladder"
git push origin main
git rev-parse HEAD
```

Review and stage the campaign's additions to `docs/HANDOFF.md`,
`docs/LEGACY_SOURCE_MAP.md`, `docs/plans/README.md` and `README.md` with `git add -p`
before the commit if those files also contain unrelated edits.

## Paste once on Oscar

The parent spec supplies authenticated FullStat data/runtime provenance only;
none of its old teachers or historical metrics initialize this campaign.
Require36GiB available HOME quota for new outputs; the prior dataset/models stay.
All stages request one L40S, six CPUs,192000MiB RAM and48h. This is a conservative
ceiling; measured admission can reject a stage without changing its science.

```bash
bash <<'BASH'
set -euo pipefail
checkquota
read -r -p 'Pushed 40-character balanced-ladder commit: ' COMMIT </dev/tty
read -r -p 'Available HOME quota GiB (minimum 36): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 36 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_subclass_16546c0e"
PARENT="${BASE}/checkpoints/luka_fullstat_six_teacher_memory_2fb09f38_r1/campaign_spec.json"
PROJECT="${BASE}/source/HLT_Classification_balanced_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_balanced_${COMMIT:0:8}_r1"
test -r "${PARENT}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_balanced_ladder.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_BALANCED_LADDER_AUTOSUBMIT_V1
BASH
```

After **DETACHED**, you may disconnect. Ctrl+C exits the displayed tail only.
**LAUNCH_COMPLETE** followed by **LAUNCHER_EXIT=0** confirms submission of the
exact jobs. DETACHED alone does not mean submission succeeded. Reconnect using
the printed log path; do not rerun the launch command or delete claims.
Runtime import failure is recorded before any scientific job submission.

## Inspect progress

Use the printed PROJECT and ROOT values, then:

```bash
squeue --me -o '%.18i %.45j %.2t %.10M %.12l %R'
tail -n 60 "${ROOT}.launch/launcher.log"
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_balanced_ladder.py" progress \
    --spec "${ROOT}/campaign_spec.json" --stage DE000
```

Replace DE000 with OFFLINE/M0HLT/CONCAT/UE100/DE050 as needed. This reads small
epoch reports and the log; it does not load models or run inference. Progress
estimates are at the last marker and exclude validation/export.

After all six jobs finish:

```bash
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_balanced_ladder.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

This authenticates published results and returns accuracy/AUC, R50/R30 with
passing counts, and recovery against the **new balanced** M0HLT/OFFLINE models,
for natural500k, in-range, outside-range and each100GeV bin. No inference or
final-test access. The historical recovery percentages are not the new anchors.

## Recovery

Inspect the exact failed job log and `failure.json`. Wait for accounting to
become terminal; dependent jobs may need explicit cancellation by their saved
IDs. Do not use broad job-name cancellation. Preserve old results/journals.

Use the same pushed source and a new output root, with the failed balanced
campaign's spec as PARENT. After checking36GiB **additional** available quota:

```bash
OLD_ROOT="${ROOT}"
RECOVERY_ROOT="${OLD_ROOT}_recovery1"
bash "${PROJECT}/scripts/launch_luka_fullstat_balanced_ladder.sh" \
    "${COMMIT}" "${RECOVERY_ROOT}" "${OLD_ROOT}/campaign_spec.json" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_BALANCED_LADDER_AUTOSUBMIT_V1 --recover
```

Successful stages are reused. Incomplete stages resume from their last committed
epoch, preserving optimizer/RNG/draw schedule. With20 committed epochs, recovery
repeats only selected validation/export/publication. Partial exports are rebuilt
in the fresh root. A mid-epoch failure can lose that unfinished epoch; it does
not lose earlier committed epochs. A claim prevents launching two recoveries of
the same attempt. Ambiguous sbatch responses require journal reconciliation,
not blind resubmission. Runtime-byte or resource changes are not silently accepted.

For a manual dry-plan workflow instead of detached launch, use
`scripts/queue_luka_fullstat_balanced_ladder.sh COMMIT ROOT PARENT_SPEC QUOTA_GIB`,
review `submission/command_plan.json`, then repeat with `--execute PLAN_HASH`.
Do not mix this path with an already started detached launcher.

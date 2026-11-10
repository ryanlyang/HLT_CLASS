# Queue contrastive0.6 with subclass0.5 on Oscar

Exactly one fresh20-epoch HLT fit on3M TRAIN/500k validation. Contrastive
matching rises from0.3 to0.6; subclass0.5, embeddings10 and CE1 stay fixed.
Original subclass job8761518 must have completed publication and accounting.
Its frozen subclass targets are copied exactly. No teacher retraining or
separate preflight job. See the [plan](plans/LUKA_FULLSTAT_SUBCLASS_C06_PLAN.md),
[contract](contracts/LUKA_FULLSTAT_SUBCLASS_C06.md) and
[validation](LUKA_FULLSTAT_SUBCLASS_C06_VALIDATION.md).

## Commit and push locally

Implementation baseline isffaf2997b3fa76a45e53105dae4c8ed68be3d56a. Historical
subclass source is16546c0e0196ab85cd69e23de005dc5f6f29feb5 and remains unchanged.
Stage only the scoped campaign additions. Review shared documentation edits
separately; preserve unrelated working-tree changes and temporary artifacts.

```powershell
git add -- src/hlt_classification/luka_fullstat_subclass_c06 scripts/luka_fullstat_subclass_c06.py scripts/launch_luka_fullstat_subclass_c06.sh sbatch/run_luka_fullstat_subclass_c06.sh tests/test_luka_fullstat_subclass_c06.py tests/test_luka_fullstat_subclass_c06_targets.py tests/test_luka_fullstat_subclass_c06_lifecycle.py docs/plans/LUKA_FULLSTAT_SUBCLASS_C06_PLAN.md docs/contracts/LUKA_FULLSTAT_SUBCLASS_C06.md docs/LUKA_FULLSTAT_SUBCLASS_C06_RUNBOOK.md docs/LUKA_FULLSTAT_SUBCLASS_C06_VALIDATION.md
git diff --cached --stat
git commit -m "Add Oscar contrastive0.6 with fixed subclass0.5"
git push origin main
git rev-parse HEAD
```

## Paste once on Oscar

```bash
bash <<'BASH'
set -euo pipefail
checkquota
read -r -p 'Pushed 40-character C06 commit: ' COMMIT </dev/tty
read -r -p 'Available HOME quota GiB (minimum 8): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 8 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_subclass_16546c0e"
PARENT="${BASE}/checkpoints/luka_fullstat_subclass_16546c0e_r2/campaign_spec.json"
PROJECT="${BASE}/source/HLT_Classification_subclass_c06_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_subclass_c06_${COMMIT:0:8}_r1"
test -r "${PARENT}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_subclass_c06.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_SUBCLASS_C06_AUTOSUBMIT_V1
BASH
```

After **DETACHED**, logout is safe: original-source authentication, scheduler
dry review and submission continue in the detached process. Tail is only a
view; Ctrl+C does not stop submission. **LAUNCH_COMPLETE** plus
**LAUNCHER_EXIT=0** confirms successful submission. Do not launch twice or
remove claims after a failure. No old jobs are cancelled by this command.

The job is`lukafssc6_train_subclass`: one L40S,6CPU,192000MiB,48h ceiling.
The same in-job miniature and measured admission precede fresh training;
there is no separate preflight allocation. No wall-clock speedup is promised.

## Inspect progress and results

Use the exact ROOT and PROJECT printed by the launcher:

```bash
squeue --me -o '%.18i %.45j %.2t %.10M %.12l %R'
grep -E '^SUBCLASS C06 (epoch=|progress|measured)' "${ROOT}"/train_subclass-*.out | tail -n 35
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_subclass_c06.py" results --spec "${ROOT}/campaign_spec.json"
```

Run`results` after completion. It verifies saved results without research-data
inference or final-test access. Both the34.57% original contrastive result and
the34.73% subclass+C0.3 result are included as historical references; their
values are read from verified reports, not hardcoded into metrics.

## Explicit recovery

For a terminal failed/cancelled/timed-out C06 job with completed epochs, retain
its exact commit/worktree and parent, choose a fresh ROOT and append
`--resume-spec /absolute/failed/C06/root/campaign_spec.json` to the launcher.
Frozen targets are copied without reclustering; the authenticated completed
epoch prefix restores optimizer/RNG/schedule. With20 completed epochs, only
publication follows integrated checks. C03 weights are never a valid C06 resume.

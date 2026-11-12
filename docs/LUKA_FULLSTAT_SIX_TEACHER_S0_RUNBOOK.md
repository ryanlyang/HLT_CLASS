# Queue the six-teacher subclass-off experiment

One fresh HLT-only fit, 3M TRAIN/500k validation, 20 epochs. Same six-teacher
recipe as job 8848995 with subclass weight 0. Contrastive stays 0.3, CE 1,
embedding 15 and auxiliary logit KD 1. No new control, teacher training,
teacher exports or separate preflight. See the [plan](plans/LUKA_FULLSTAT_SIX_TEACHER_S0_PLAN.md),
[contract](contracts/LUKA_FULLSTAT_SIX_TEACHER_S0.md) and
[validation](LUKA_FULLSTAT_SIX_TEACHER_S0_VALIDATION.md).

The completed control at 2fb09f38/job 8848995 is required. Keep its root, source,
frozen banks and original provenance. This is a new fit from scratch, not a
continuation of the control. Reference authentication is read-only and may take
a few minutes in the detached launcher; it does not rerun model inference.

## Commit and push the scoped addition locally

Preserve unrelated dirty work. Baseline 27a83412 is the reviewed repository state
for source admission; subsequent runtime changes outside this new namespace are
not admitted by its source guard. Review shared HANDOFF/donor-map changes
separately if those files contain other work.

```powershell
$files = @(
    'src/hlt_classification/luka_fullstat_six_teacher_s0'
    'scripts/luka_fullstat_six_teacher_s0.py'
    'scripts/launch_luka_fullstat_six_teacher_s0.sh'
    'sbatch/run_luka_fullstat_six_teacher_s0.sh'
    'tests/test_luka_fullstat_six_teacher_s0.py'
    'tests/test_luka_fullstat_six_teacher_s0_source.py'
    'tests/test_luka_fullstat_six_teacher_s0_lifecycle.py'
    'docs/plans/LUKA_FULLSTAT_SIX_TEACHER_S0_PLAN.md'
    'docs/contracts/LUKA_FULLSTAT_SIX_TEACHER_S0.md'
    'docs/LUKA_FULLSTAT_SIX_TEACHER_S0_RUNBOOK.md'
    'docs/LUKA_FULLSTAT_SIX_TEACHER_S0_VALIDATION.md'
)
git add -- $files
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat -- $files
git commit --only -m 'Add matched Oscar six-teacher subclass-off ablation' -- $files
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Paste once on Oscar

```bash
bash <<'BASH'
set -euo pipefail
checkquota
read -r -p 'Pushed 40-character subclass-off commit: ' COMMIT </dev/tty
read -r -p 'Available HOME quota GiB (minimum 12): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 12 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_six_teacher_memory_2fb09f38"
PARENT="${BASE}/checkpoints/luka_fullstat_six_teacher_memory_2fb09f38_r1/campaign_spec.json"
PROJECT="${BASE}/source/HLT_Classification_six_teacher_s0_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_six_teacher_s0_${COMMIT:0:8}_r1"
test -r "${PARENT}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_six_teacher_s0.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_SIX_TEACHER_S0_AUTOSUBMIT_V1
BASH
```

After **DETACHED**, authentication, dry review and submission continue if SSH
disconnects. Ctrl+C exits the foreground tail only. **LAUNCH_COMPLETE** and
**LAUNCHER_EXIT=0** confirm successful submission. Do not launch a second copy
or erase claims after failure; inspect the saved failure and submission journal.
Existing jobs are not cancelled or modified. Job name: `lukafs6s0_train_six`.

The new GPU allocation copies completed native TRAIN logit banks and reuses
all other frozen targets. It runs integrated acceptance and then starts the
20 scientific epochs. The requested limit is 48 hours; this is not a runtime prediction.

## Progress and completed results

Use ROOT/PROJECT printed by the launcher, or assign their exact saved paths:

```bash
squeue --me -o '%.18i %.45j %.2t %.10M %.12l %R'
tail -n 60 -f "${ROOT}.launch/launcher.log"
```

```bash
grep -E '^SIX TEACHER S0 (epoch=|progress|measured|reused)' "${ROOT}"/train_six-*.out | tail -n 30
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_six_teacher_s0.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

Results authenticate completed saved validation outputs and report accuracy,
AUC, R30/R50 and recoveries, including SIX_TEACHER and SIX_TEACHER_S0. They
do not perform new inference or open final-test data. The comparison remains
single-seed development validation, regardless of the outcome.

## Explicit recovery

After inspecting a terminal unsuccessful S0 job, use the same pushed source,
same completed control, a fresh ROOT and append
`--resume-spec OLD_S0_ROOT/campaign_spec.json` after the authorization token.
Completed epoch states preserve optimizer/RNG; all 20 completed epochs allow
publication-only recovery. Fresh target copies come from the same control.
No changed-source recovery or automatic retry is implemented. Preserve all
old roots, failure files, source worktrees and exclusive recovery claims.

# Queue the six-teacher student on Oscar

One fresh HLT-only20-epoch fit on3M TRAIN/500k validation. Six embedding heads,
six auxiliary logit heads, pure-OFFLINE contrastive0.3 and frozen subclass0.5.
No separate preflight and no teacher retraining. Two native TRAIN logit exports
run inside the student allocation. [Science](plans/LUKA_FULLSTAT_SIX_TEACHER_PLAN.md),
[contract](contracts/LUKA_FULLSTAT_SIX_TEACHER.md),
[validation](LUKA_FULLSTAT_SIX_TEACHER_VALIDATION.md).

## Commit and push locally

Runtime baseline is0a00441fee7b7974f6e7b2692e79519a65faabd3. Stage only the new
campaign files; review shared documentation separately. Preserve unrelated
dirty Hydra work. The queue guard rejects changes to historical runtime files.

```powershell
git add -- src/hlt_classification/luka_fullstat_six_teacher scripts/luka_fullstat_six_teacher.py scripts/launch_luka_fullstat_six_teacher.sh sbatch/run_luka_fullstat_six_teacher.sh tests/test_luka_fullstat_six_teacher.py tests/test_luka_fullstat_six_teacher_targets.py tests/test_luka_fullstat_six_teacher_lifecycle.py docs/plans/LUKA_FULLSTAT_SIX_TEACHER_PLAN.md docs/contracts/LUKA_FULLSTAT_SIX_TEACHER.md docs/LUKA_FULLSTAT_SIX_TEACHER_RUNBOOK.md docs/LUKA_FULLSTAT_SIX_TEACHER_VALIDATION.md
git diff --cached --stat
git commit -m "Add Oscar six-teacher subclass student"
git push origin main
git rev-parse HEAD
```

## Paste once on Oscar

```bash
bash <<'BASH'
set -euo pipefail
checkquota
read -r -p 'Pushed 40-character six-teacher commit: ' COMMIT </dev/tty
read -r -p 'Available HOME quota GiB (minimum 12): ' QUOTA </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${#QUOTA}" -le 6 ]]
(( QUOTA >= 12 ))
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_subclass_16546c0e"
PARENT="${BASE}/checkpoints/luka_fullstat_subclass_16546c0e_r2/campaign_spec.json"
LADDER="${BASE}/checkpoints/luka_fullstat_fine_tail_k1_e40_04d1f0f5_r1/campaign_spec.json"
PROJECT="${BASE}/source/HLT_Classification_six_teacher_${COMMIT:0:8}"
ROOT="${BASE}/checkpoints/luka_fullstat_six_teacher_${COMMIT:0:8}_r1"
test -r "${PARENT}"
test -r "${LADDER}"
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_six_teacher.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${LADDER}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_SIX_TEACHER_AUTOSUBMIT_V1
BASH
```

After **DETACHED**, authentication, dry review and submission continue without
the SSH session. The tail is only a view; Ctrl+C does not stop the launcher.
**LAUNCH_COMPLETE** and **LAUNCHER_EXIT=0** confirm successful submission.
Do not submit again or delete claims after a failure. No existing jobs are
cancelled. The job name is `lukafs6_train_six`.

The first phase exports frozen CONCAT/FUSION native T2 probabilities once.
This adds setup time, not two new training runs. Full20-epoch training starts
after target joins, installed-Weaver/gradient checks and complete-update timing
on the real allocation. The48-hour request is a ceiling, not a measured runtime
promise; resource admission includes preparation and validation/publication.

## Progress and results

Use the ROOT and PROJECT printed by the launcher:

```bash
squeue --me -o '%.18i %.45j %.2t %.10M %.12l %R'
tail -n 50 "${ROOT}.launch/launcher.log"
grep -E '^SIX TEACHER (epoch=|progress|measured|native KD)' "${ROOT}"/train_six-*.out | tail -n 35
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_six_teacher.py" results --spec "${ROOT}/campaign_spec.json"
```

Results authenticate saved publication and compute AUC/accuracy, R50/R30 and
recovery at the AUC-selected epoch. Historical references include subclassC03,
Multi3+contrastive, Multi3, U100-w5, M0HLT and OFFLINE. No final-test inference.

## Explicit recovery

After inspecting a terminal unsuccessful job, use the same pushed source and
parent specs, a fresh root, and append `--resume-spec OLD_ROOT/campaign_spec.json`
to the launcher command after the authorization token. A successful completed
export or at least one completed epoch is required. Complete exports copy
exactly; an incomplete export is rebuilt in the fresh root. Completed epochs
resume with their optimizer and RNG states; a20-epoch fit only republishes.
Every recovery uses an exclusive claim on the original run. Ambiguous scheduler
submission still requires inspection; do not erase the old journals/claims.

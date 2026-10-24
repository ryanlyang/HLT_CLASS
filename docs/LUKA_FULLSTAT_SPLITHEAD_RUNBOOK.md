# Queue the FullStat split-head continuation on Oscar

Three fresh fixed20 fits: DE066 -> DE033 -> DE000. Reuse completed hybrid UE100;
main CE=0.5, auxiliary-logit total=0.5 at T=2, embedding total=5. Exact per-stage
teachers: [plan](plans/LUKA_FULLSTAT_SPLITHEAD_PLAN.md);
[contract](contracts/LUKA_FULLSTAT_SPLITHEAD.md).
No standalone preflight or login controller. Each job contains three disposable
technical updates and resets before scientific training. Old jobs are not changed.

## Commit and push locally

Stage only these new files. HANDOFF/LEGACY_SOURCE_MAP have unrelated edits; use
selective staging if including their new sections. Do not use `git add .`.

```powershell
git add -- src/hlt_classification/luka_fullstat_splithead scripts/luka_fullstat_splithead.py scripts/queue_luka_fullstat_splithead.sh sbatch/run_luka_fullstat_splithead.sh tests/test_luka_fullstat_splithead.py docs/plans/LUKA_FULLSTAT_SPLITHEAD_PLAN.md docs/contracts/LUKA_FULLSTAT_SPLITHEAD.md docs/LUKA_FULLSTAT_SPLITHEAD_RUNBOOK.md
git diff --cached --stat
git commit -m "Add FullStat split-head DE066 to DE000 continuation"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before proceeding" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

Use the printed 40-character commit. Existing runtime must remain unchanged
relative to the allowed additive baseline; do not weaken the source guard.

## Review on Oscar

Completed hybrid UE000/UE100, their original pinned source and all prior referenced
artifacts must remain intact. Old DE066/DE033/DE000 need not have completed.
Run checkquota; declare 16 only if at least 16 GiB durable personal quota is free.

```bash
(
set -euo pipefail
read -r -p 'Pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="${MAIN}_fullstat_splithead_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_splithead_${COMMIT:0:8}_r1"
PREVIOUS=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_hybrid_embedding_e129075e_r1/campaign_spec.json
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
bash "${PROJECT}/scripts/queue_luka_fullstat_splithead.sh" "${COMMIT}" "${ROOT}" "${PREVIOUS}" 16
printf '\nStudy: %s\nProject: %s\n' "${ROOT}" "${PROJECT}"
)
```

No jobs are submitted here. Review jobs=3, preflight_jobs=0, setup_jobs=0,
teacher lists, 20 epochs and task walltimes. Copy PLAN_HASH. Requests: exact
accepted L40S, 6 CPUs, 180000 MiB; inherited hybrid request +60 minutes, +120
for DE066 instead. New combined runtime unmeasured; reject rather than clamp >48h.

## Submit all three jobs

```bash
(
set -euo pipefail
read -r -p 'Same pushed commit: ' COMMIT </dev/tty
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_splithead_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_splithead_${COMMIT:0:8}_r1"
PREVIOUS=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_hybrid_embedding_e129075e_r1/campaign_spec.json
read -r -p 'Authorize all three split-head fits and TRAIN exports? Type YES: ' CONFIRM </dev/tty
test "${CONFIRM}" = YES
bash "${PROJECT}/scripts/queue_luka_fullstat_splithead.sh" "${COMMIT}" "${ROOT}" "${PREVIOUS}" 16 --execute "${PLAN_HASH}"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
)
```

Expect lukafss_train_DE066, lukafss_train_DE033, lukafss_train_DE000. All queue
together, sequential afterok. Once all exact IDs are recorded you can disconnect.
No follow-up login process is needed. Technical failure blocks descendants; poor
finite science does not. Never repeat an uncertain submission: inspect the saved
submission journal and exact IDs first. No wildcard cancellation or auto cleanup.

## Progress and saved results

Logs: train_STAGE-JOBID.out. First DE066 exports M0HLT TRAIN probabilities before
epoch1. Both DE066 and DE033 export their selected native embeddings/probabilities
after epoch20; epoch20 is not yet complete-task publication. Logs show every
epoch's AUC/best epoch and every128 batches' progress/elapsed TRAIN seconds.

```bash
(
set -euo pipefail
read -r -p 'Execution commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
export PROJECT_DIR="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_splithead_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_splithead_${COMMIT:0:8}_r1"
export JC2_SITE=oscar_l40s
unset PYTHONHOME PYTHONPATH LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_splithead.py" results --spec "${ROOT}/campaign_spec.json"
)
```

Reads saved authenticated validation results, not new inference. Reports completed
stages, pending stages, historical references and R30/R50 recovery in JSON. No
test operation exists. Only final DE000 is an HLT-only deployable result. This
adaptive single-seed comparison is not an independent test or a fresh CE control.

Local installed-Weaver GPU miniature tests do not certify new Oscar throughput.
This requested no-preflight path reuses original Oscar evidence and performs
bounded checks inside each job. Any runtime/envelope mismatch fails closed.

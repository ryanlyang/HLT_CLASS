# Luka K1/E40 token-depth and teacher-diversity screen

See [registered science and ordered 20 fits](plans/LUKA_SUPERVISION_DEPTH_PLAN.md)
and [artifact contract](contracts/LUKA_SUPERVISION_DEPTH.md).

25 debug jobs: targets, 20 students, 2 OFFLINE teachers, report, cleanup. No separate
preflight. In-worker disposable safety checks remain. Only training targets can
use richer views; all 20 deployed student models remain native HLT-only.
Students are serial in declared order. Teacher controls run between students 10
and 11; results may be viewed while later rows are pending. Technical failures
block later tasks; finite poor performance does not. No old jobs are cancelled.

## Commit the scoped additions locally

Do not use git add . or force-push. Preserve unrelated edits to HANDOFF and the
donor map; stage their new entries selectively if they contain other work.

```powershell
git add -- src/hlt_classification/luka_supervision_depth scripts/luka_supervision_depth.py scripts/queue_luka_supervision_depth.sh sbatch/run_luka_supervision_depth.sh tests/test_luka_supervision_depth.py docs/plans/LUKA_SUPERVISION_DEPTH_PLAN.md docs/contracts/LUKA_SUPERVISION_DEPTH.md docs/LUKA_SUPERVISION_DEPTH_RUNBOOK.md
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Review whitespace errors before committing' }
git diff --cached --stat
git commit -m "Add matched token depth and OFFLINE diversity debug screen"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before continuing' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; stop, do not force-push' }
git rev-parse HEAD
```

## SPORC: exact pushed worktree and full dry review

Reserve at least 8 GiB personally available persistent quota. Token jobs also need
node-local disk proportional to actual real TRAIN tokens and selected depths;
they must not fall back to home/Ceph. Old OFFLINE caches may be reconstructed in
RAM for TRAIN/validation, with exact prepared fingerprints and no test access.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_depth_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_supervision_depth.sh"
BASH
```

The helper asks for the completed auxiliary-grid campaign_spec.json path (the
grid containing k1_e40_plain), quota, displayed PLAN_HASH and YES. It authenticates
the original parent source/accounting, creates a fresh immutable spec, performs
full dry review, then submits the dependency graph in a detached process.
It does not retrain any c50 ladder teacher. The two new OFFLINE fits are explicit
diversity controls, not replacements for the frozen KD teachers.
Submission PID alone is not success: inspect the printed log and accepted IDs.
Do not blindly rerun after interruption or a partial submission; preserve journal.

## Results

Use the exact ROOT and PROJECT printed by the helper in the environment activated
by the same pinned worktree (atlas_kd_sporc, PYTHONPATH=PROJECT/src, NOUSERSITE,
LD_LIBRARY_PATH prepended, single-thread BLAS, CUBLAS=:4096:8):

```bash
python -u -B -s "${PROJECT}/scripts/luka_supervision_depth.py" results \
  --spec "${ROOT}/campaign_spec.json"
squeue --me -o "%.18i %.12P %.40j %.2t %.10M %R"
```

Histories and native selected checkpoints remain in ROOT/<arm>/fit. All 30 epochs
run; AUC uses native CE head only. New teacher rows are OFFLINE ORACLE controls.
No calibrated confidence from this adaptive development screen is implied.

## Code map and verification boundary

`contracts.py` freezes all 25 tasks/recipes; `model.py` taps native blocks and
implements common-assignment set loss and SupCon; `banks.py` owns TRAIN targets;
`training.py` records loss components and selects native checkpoints; `worker.py`
exports/fits/reports/retires; `source.py` authenticates original lineage;
`submission.py` generates the hash-approved exact debug DAG.

No separate GPU preflight is submitted. Inherited A100 acceptance is not a new
measurement of these objectives; 12h GPU allowances are conservative, not a
runtime promise. Real SPORC execution is still required to establish full-scale
resource use. Local synthetic installed-Weaver checks do not claim remote data
or scheduler validation. Fresh_check, datasets and old campaigns are untouched.

Local release evidence (2026-10-27): 52 new tests passed, one Windows symlink
test skipped; 106 existing grid/token/joint tests passed. The synthetic CUDA
miniature exercised every worker, all 22 scientific 30-epoch fits, native
checkpoint readback and reporting after target retirement. The exact baseline
also reproduces the old K1/E40 30-epoch history and selected native state.

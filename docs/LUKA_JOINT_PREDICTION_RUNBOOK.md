# Queue CE + jet4 + tokens3 on SPORC debug

Two NEW fits: aligned and unordered. No new preflight job, no technical epoch.
Uses the completed token-study resource evidence and original frozen U100 teacher.
Both128-D heads disappear at inference. Full recipe: [plan](plans/LUKA_JOINT_PREDICTION_PLAN.md).

## Commit only the new experiment files (local PowerShell)

```powershell
git add -- src/hlt_classification/luka_joint_prediction scripts/luka_joint_prediction.py scripts/queue_luka_joint_prediction.sh sbatch/run_luka_joint_prediction.sh tests/test_luka_joint_prediction.py docs/plans/LUKA_JOINT_PREDICTION_PLAN.md docs/contracts/LUKA_JOINT_PREDICTION.md docs/LUKA_JOINT_PREDICTION_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Review staged changes' }
git commit -m "Add joint jet4 token3 prediction direct debug trials"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before continuing' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push' }
git rev-parse HEAD
```

Inspect the index for unrelated staged work before committing. HANDOFF and donor
map updates coexist with unrelated local edits; selectively stage those hunks if
desired. Do not add all repository changes to make a source guard pass.

## SPORC: fresh pinned worktree, review, submit

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the NEW pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_jointpred_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_joint_prediction.sh"
BASH
```

Reserve at least2GiB personally available persistent space. Paste the actual
printed64-character PLAN_HASH, then YES. Helper authenticates the completed
`luka_particle_prediction_d8dba242_debug_r1` in its original worktree, renders
two exact `lukajp_aligned`/`lukajp_set` debug jobs, checks Slurm dry admission,
then submits only with your hash authorization. No jobs are cancelled or moved.
Missing prior evidence is an error, not permission to silently bypass validation.

Requests inherit measured token resources with25% time margin plus30min and a
4h minimum, capped by rejection at24h; this is NOT a measurement of the new
combination. Both jobs need sufficient node-local /tmp disk. No fallback to home.
Creation/source checks may take minutes; watch the helper's printed log for both
accepted job IDs. Preserve partial roots/journals on error; don't blindly rerun.

## Results, including partial completion

Use the NEW exact project/root printed by the helper. Activate atlas_kd_sporc,
set PYTHONNOUSERSITE=1, PYTHONDONTWRITEBYTECODE=1, PYTHONPATH to that project's
src, and prepend CONDA_PREFIX/lib to LD_LIBRARY_PATH as in the worker. Then:

```bash
python -u -B -s "${PROJECT}/scripts/luka_joint_prediction.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

Each arm retains fit/training_report.json (all30 epochs, CE/jet/token raw and
weighted losses), native selected.pt, validation_predictions.npz, receipts,
reference predictions, target moments/hashes and phase timings. No scratch banks
are required for reporting. No independent test: original50k is report-only.
No commit, push or cluster submission is performed by local implementation tests.

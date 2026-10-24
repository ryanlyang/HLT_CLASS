# Four trial hybrid ladders: direct debug queue

Authority: [plan](plans/LUKA_TRIAL_HYBRID_PLAN.md), [contract](contracts/LUKA_TRIAL_HYBRID.md).

Four complete from-scratch UE000 -> UE100 -> DE066 -> DE033 -> DE000 chains:

| Arm | CE / KD coefficients |
| --- | --- |
| c25 | 0.25 / 0.75 throughout |
| c50 | 0.50 / 0.50 throughout |
| c75 | 0.75 / 0.25 throughout |
| anneal | KD decreases linearly from 0.75 at epoch 1 to zero at epoch 25; CE = 1 - KD |

The annealed arm stays at CE-only task loss through epoch 30, while embedding
weight remains 5 throughout. The schedule resets at each rung. Teachers are
branch-specific; no existing learned U/D checkpoint substitutes for a new rung.
Use our 299,234 / 74,895 trial split, not 3M FullStat. There is no independent
test; original50k is report-only.

Exactly 22 jobs on debug: `lukahy_targets`, 20 `lukahy_<arm>_<stage>`, then
`lukahy_report`. No new preflight, stress steps, technical epoch or controller.
Existing acceptance/source checks and scheduler `sbatch --test-only` remain.
Old jobs, Fresh_check, datasets and old caches are never altered.

## Commit locally

Only stage the new package, scripts, test and three docs; shared HANDOFF and
LEGACY_SOURCE_MAP contain unrelated work and should be reviewed/staged by hunk.

```powershell
& {
    Set-Location C:\Users\22rya\ComputerScience\CERN\HLT_Classification
    $staged = git diff --cached --name-only
    if ($LASTEXITCODE -ne 0 -or $staged) { throw "Review the existing index first." }
    git add -- src/hlt_classification/luka_trial_hybrid scripts/luka_trial_hybrid.py scripts/queue_luka_trial_hybrid.sh sbatch/run_luka_trial_hybrid.sh tests/test_luka_trial_hybrid.py docs/plans/LUKA_TRIAL_HYBRID_PLAN.md docs/contracts/LUKA_TRIAL_HYBRID.md docs/LUKA_TRIAL_HYBRID_RUNBOOK.md
    if ($LASTEXITCODE -ne 0) { throw "Stage failed" }
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw "Review staged whitespace" }
    git diff --cached --stat
    if ((Read-Host 'Commit and push these changes? Type YES') -ne 'YES') { return }
    git commit -m "Add four trial hybrid KD embedding ladders in debug"
    if ($LASTEXITCODE -ne 0) { throw "Commit failed; stop" }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw "Push failed; do not force-push" }
    git rev-parse HEAD
}
```

## Queue on sporcsubmit

The exact completed source study must remain available:
`checkpoints/luka_particle_prediction_d8dba242_debug_r1/campaign_spec.json`.
It authenticates the original native/KD and embedding resources, populations,
models and comparator predictions in the original pinned worktrees. Missing
evidence stops creation; never substitute a different teacher or bypass checks.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the new pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_trialhybrid_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_trial_hybrid.sh"
BASH
```

Enter personally available quota >=8 GiB, review the full 22-job plan, paste the
64-character hash VALUE, then YES. The helper submits the whole dependency graph
in a disconnect-safe process. Wait for its successful submission ledger; a PID
alone is not proof that jobs submitted. Do not submit a second copy.

6 CPU / 90,000 MiB / A100 fit requests use inherited projection * 1.5 + 120 min,
minimum 8 hours; reject over 24 hours. Target export: 8 hours. CPU report:
1 CPU / 16 GiB / 2 hours. New objective is not
remotely measured: these are conservative allowances, not timing guarantees.
All 30 epochs run regardless of performance. No training recipe changes on OOM.

## Progress and results

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the same pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_trialhybrid_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_trial_hybrid_${COMMIT:0:8}_debug_r1"
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
test -f "${ROOT}/campaign_spec.json"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
squeue --me -o "%.18i %.12P %.35j %.2t %.10M %R"
python -u -B -s "${PROJECT}/scripts/luka_trial_hybrid.py" results --spec "${ROOT}/campaign_spec.json"
printf '\nLogs: %s/<arm>_<stage>-<JOBID>.out\n' "${ROOT}"
BASH
```

Results are available after shared export; incomplete fits are explicit pending
rows. Compare final DE000 across the four mixtures, M0HLT/OFFLINE and frozen
historical OFFLINE-w5/U100-w5/token controls. Intermediate results are oracles.
Final report persists `report/results.json`, then deletes only 18 new compact
TRAIN-target NPZ payloads, retaining normalization moments, manifests and hashes.
All 20 native selected checkpoints, predictions, full histories and logs remain.
There is no backup of retired targets; rebuilding requires re-export from kept
models/views. After cleanup, results authenticate retained data and descriptors,
not missing target bytes. An interrupted cleanup requires explicit recovery.

## Code map

`contracts.py`: exact four mixtures/teacher graph; `training.py`: fixed30 losses
and selection; `model.py`: native multihead adapter; `banks.py`: frozen native
targets/branch joins; `source.py`: inherited acceptance; `campaign.py`: immutable
scope/resources; `submission.py`: dry/live exact DAG; `worker.py`: scientific
execution, result authentication and owned retirement. Thin CLI/queue/Slurm
scripts never redefine physics. See HANDOFF for local test evidence and limits.

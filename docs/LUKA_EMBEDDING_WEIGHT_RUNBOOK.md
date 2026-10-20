# Three no-KD embedding weights on SPORC tier3

Authority: [plan](plans/LUKA_EMBEDDING_WEIGHT_PLAN.md),
[contracts](contracts/LUKA_EMBEDDING_WEIGHT.md).

Three fresh single-HLT models: `CE + lambda * SmoothL1`, lambda = 0.75, 1.5, 10.
No KD or ladder. Same trial ALLJETS 299,234 / 74,895, batch 128, fixed 30 epochs,
seeds, teacher embeddings and full-val selection as the existing 0.3 experiment.
No independent test. Reserve 2 GiB total, without duplicating the shared bank.
Both the shared preflight and all three scientific jobs use **tier3**.
Create a fresh tier3 root; do not edit or reuse an old debug gate or plan.

Local verification: **53 tests passed**, covering the new sweep plus unchanged
single-stage and ladder regressions, including installed local CUDA/Weaver
miniature workers. CLI, shell syntax and documentation links were checked.
Tier3 follow-up: **18 focused tests passed in 31.21s**, including exact tier3
submission commands, rejection of old debug specifications and the source pin.
Actual SPORC resource acceptance still requires the new preflight below.
No jobs have been submitted by this implementation.

## 1. Commit/push on Windows

Stage only this experiment, not unrelated work. Review the staged diff.

```powershell
git add -- src/hlt_classification/luka_embedding_weight `
  scripts/luka_embedding_weight.py scripts/queue_luka_embedding_weight.sh `
  sbatch/run_luka_embedding_weight.sh tests/test_luka_embedding_weight.py `
  docs/plans/LUKA_EMBEDDING_WEIGHT_PLAN.md docs/contracts/LUKA_EMBEDDING_WEIGHT.md `
  docs/LUKA_EMBEDDING_WEIGHT_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw "Stage failed" }
git diff --cached --stat
# Optionally stage only the new HANDOFF/LEGACY_SOURCE_MAP sections with git add -p.
git commit -m "Add matched no-KD embedding weights 0.75 1.5 and 10"
if ($LASTEXITCODE -ne 0) { throw "Inspect status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Stop; do not force-push" }
git rev-parse HEAD
```

## 2. Create and queue one shared preflight on sporcsubmit

Keep the original target bank and its pinned source worktree intact. This uses
the completed `luka_embedding_cfb3a158_debug_r1` run, not the embedding ladder.
That completed reference stays at its original debug path. The new source must
preserve existing runtime relative to commit ed129fa9, which adds the unrelated
FullStat namespace without changing this experiment's scientific dependencies.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve artifacts and paste the error." >&2' ERR
read -r -p 'Paste the NEW pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_embedweight_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_embedding_weight_${COMMIT:0:8}_tier3_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
EMBED="${MAIN}/checkpoints/luka_embedding_cfb3a158_debug_r1/science/campaign_spec.json"
test -f "${PARENT}"
test -f "${EMBED}"
test ! -e "${ROOT}"
df -h /home/ryreu
read -r -p 'Personally available GiB reserved for all three runs (at least 2): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT}/scripts/luka_embedding_weight.py" create \
    --root "${ROOT}" --parent "${PARENT}" --embedding "${EMBED}" \
    --expected-commit "${COMMIT}" --quota-gib "${QUOTA}"
printf '\nPROJECT=%s\nROOT=%s\n' "${PROJECT}" "${ROOT}"
bash "${PROJECT}/scripts/queue_luka_embedding_weight.sh" "${PROJECT}" "${ROOT}" gate
BASH
```

Inspect the dry plan, paste its actual 64-character hash and confirm YES.
The helper prints a log; confirm a real Slurm job ID (name `lukaew_preflight`),
not just the helper PID. The preflight checks all three weights. Nothing
automatically submits science.

## 3. After successful preflight, queue all three scientific fits

Set `PROJECT` and `ROOT` to the exact values printed above. They are not
preserved outside the previous Bash block.

```bash
bash "${PROJECT}/scripts/queue_luka_embedding_weight.sh" "${PROJECT}" "${ROOT}" science
```

This authenticates COMPLETED/0:0 preflight evidence, creates a science spec,
dry-reviews **three** commands, then asks for its hash and YES. All use tier3,
one A100, 6 CPUs, 90000 MiB and independently measured walltime capped at 24h.
They can run independently as resources allow. Names: `lukaew_w075`,
`lukaew_w15`, `lukaew_w10`. An uncertain/partial submission must be inspected;
do not create another campaign to bypass a journal.

## 4. Compare completed weights, CE and existing 0.3

In the same activated environment (or reactivate as in step 2):

```bash
python -u -B -s "${PROJECT}/scripts/luka_embedding_weight.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

Incomplete arms print PENDING/INCOMPLETE. Per-epoch CE, auxiliary loss and
validation metrics are in `science/<arm>/fit/training_report.json`; live epoch
lines appear in `science/<arm>-<jobid>.out`. Results never load a model or need
the old particle caches. Preserve the new preflight/report files, reference
snapshots and selected checkpoints. No old or shared artifacts are deleted.

This is an exploratory three-weight sweep with one seed, not a statistically
independent test. Report all weights, including poor outcomes.

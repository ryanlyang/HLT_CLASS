# U100 embedding pretraining20 -> CE+5*embedding30, debug

Authority: [plan](plans/LUKA_U100_PRETRAIN_PLAN.md) and
[v1 contracts](contracts/LUKA_U100_PRETRAIN.md). Trial299234/74895, not FullStat.
No independent test; original50k is report-only. One new HLT model, two stages;
no extra controls trained. Historical OFFLINE-w5 changes target AND compute.

Stage1 freezes the classifier and learns only embedding prediction. Select
lowest full-validation embedding loss, keep backbone+predictor. Stage2 resets
AdamW/schedule and trains CE+5*embedding with the original fixed30 schedule.
Select AUC as usual. Frozen original regular COARSE_U100 is never retrained.
Normalization fits TRAIN only; validation embeddings only select stage1.

Three debug jobs are submitted together: `lukaep_setup` (8h),
`lukaep_pretrain` (6h), `lukaep_finetune` (6h). All oneA100/6CPU/90000MiB.
Both science stages remain dependent on successful setup measurement/admission;
finetuning also depends on pretraining. The workers recheck exact receipt and
Slurm accounting, not just afterok. No manual follow-up or login controller.
Measured projections must fit these caps; do not shorten walltime blindly.
These are conservative request ceilings, not runtime predictions.

## Windows: scoped commit and push

From the repository, inspect existing staged work first. Do not add the whole
dirty HANDOFF or LEGACY_SOURCE_MAP; optionally stage only this task's hunks.

```powershell
& {
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { throw "Inspect existing staged changes first" }
    git add -- src/hlt_classification/luka_u100_pretrain scripts/luka_u100_pretrain.py scripts/queue_luka_u100_pretrain.sh sbatch/run_luka_u100_pretrain.sh tests/test_luka_u100_pretrain.py docs/plans/LUKA_U100_PRETRAIN_PLAN.md docs/contracts/LUKA_U100_PRETRAIN.md docs/LUKA_U100_PRETRAIN_RUNBOOK.md
    if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
    git diff --cached --stat
    if ((Read-Host 'Commit and push? Type YES') -ne 'YES') { throw 'Stopped; staged changes preserved' }
    git commit -m "Add debug U100 embedding pretrain20 then CE embedding30"
    if ($LASTEXITCODE -ne 0) { throw "Commit failed" }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw "Push failed; do not force-push" }
    git rev-parse HEAD
}
```

## SPORC: new pinned worktree and full dry review

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve everything and paste the error." >&2' ERR
MAIN=/home/ryreu/atlas/HLT_Classification
read -r -p 'Paste the new pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_u100pre_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_pretrain.sh"
BASH
```

Reserve at least 2 GiB, inspect all three commands, paste the actual plan hash
and YES. Inspect the printed submission log for all three IDs. A PID alone
does not establish submission. No old debug holds or tier3 jobs are changed.
An interrupted/ambiguous submission requires journal inspection, not a retry.

## Progress, results and retention

Set ROOT to the exact Study path printed by the helper. Epoch lines are in
`pretrain-JOBID.out` and `finetune-JOBID.out`, prefix `U100_PRETRAIN stage=`.
Setup includes15-second phase heartbeats and per-role target export progress.
Stage1 reports val_embed, not meaningful classifier AUC. Stage2 reports both.

After completion, use the same pushed commit as the queue helper:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the 40-character source commit for this run: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_u100pre_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_u100_pretrain_${COMMIT:0:8}_debug_r1"
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_u100_pretrain.py" results --spec "${ROOT}/campaign_spec.json"
BASH
```

`pretrain/fit/selected.pt` includes predictor and native state. It is a training
transition checkpoint, not the deployable endpoint. `finetune/fit/selected.pt`
is native-only. Keep both selected states, compact TRAIN+validation targets,
normalization, reports, final predictions, acceptance evidence and journals.
There are no per-epoch checkpoints, new particle caches or attention banks.
No deletion is automatic. Full-model/input/source hashes are required for reuse;
file existence is only an early diagnostic. Missing U100 or w5 blocks the run;
never substitute a different teacher or mix FullStat artifacts.

## Implementation verification

Focused new and donor tests: **183 passed** (35 new), including actual local
installed-Weaver/CUDA parity at production configuration and a reduced-model
setup -> 20-epoch pretrain -> 30-epoch fine-tune worker miniature. Shell syntax,
runbook PowerShell/Bash parsing, CLI help and whitespace checks passed. No real
SPORC job has been submitted by this implementation. Genuine dataset/resource
acceptance is performed by the first queued setup job; both dependent science
jobs refuse to train if it fails.

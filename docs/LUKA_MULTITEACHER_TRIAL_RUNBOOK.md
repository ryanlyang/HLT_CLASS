# Five multi-teacher embedding trials on SPORC tier3

This is the smaller Luka sample: **299,234 TRAIN / 74,895 validation**, batch128,
30epochs, fresh HLT-only students. It is not the Oscar FullStat experiment.
No independent test remains; original50k is a report-only subset.

The five fits are `multi_w25`, `multi_w5`, `multi_w10`, `u100x4_w5`, and
`multi_w5_rel`. The first three use total embedding weights2.5,5,10; each is
divided equally over four independent heads. The control uses four heads
predicting the same U100 representation. The relational arm adds a TRAIN-only
neighbourhood-prediction loss to the four-teacher weight5 objective.
All use CE, no logit KD; inference removes every auxiliary head.

Authority: [active plan](plans/LUKA_MULTITEACHER_TRIAL_PLAN.md),
[v1 contract](contracts/LUKA_MULTITEACHER_TRIAL.md).

## What must still exist

The helper authenticates the original completed results and the actual selected
native checkpoints, not just path presence:

- `luka_fixed30_74aa58c5_r1/science/train_OFFLINE/selected.pt`
- `luka_fixed30_74aa58c5_r1/science/train_COARSE_U100/selected.pt`
- `luka_attention_94f56025_r1/science/train_LOGIT_D066/selected.pt`
- `luka_attention_94f56025_r1/science/train_LOGIT_D033/selected.pt`

These roots are under `/home/ryreu/atlas/HLT_Classification/checkpoints`.
Keep their original pinned worktrees, specs, retained reports/predictions and
receipts. The older attention export banks are NOT required. The required
OFFLINE-w5 reference is
`luka_embedding_sweep_007885f7_tier3_r1/science_debug/campaign_spec.json`;
its accepted resource evidence is inherited, not rerun or relabelled.
Optionally supply the completed smaller-data direct U100-w5 spec for reporting.
Enter alone omits that optional reference; it never omits any teacher.

The prepared D000/U100/D066/D033 TRAIN caches and D000 validation cache must
survive. OFFLINE TRAIN is reconstructed in RAM from the original authenticated
ROOT files if its old cache was deleted. No replacement teachers or model
retraining happen silently. Missing evidence stops before submission.

## Commit and push on Windows

From the repository PowerShell terminal, stage ONLY these new implementation
files. Existing unrelated edits, datasets and temporary directories stay out.
Review HANDOFF/LEGACY_SOURCE_MAP separately with `git add -p` if desired.

```powershell
$Files = @(
    'src/hlt_classification/luka_multiteacher_trial',
    'scripts/luka_multiteacher_trial.py',
    'scripts/queue_luka_multiteacher_trial.sh',
    'sbatch/run_luka_multiteacher_trial.sh',
    'tests/test_luka_multiteacher_trial.py',
    'docs/plans/LUKA_MULTITEACHER_TRIAL_PLAN.md',
    'docs/contracts/LUKA_MULTITEACHER_TRIAL.md',
    'docs/LUKA_MULTITEACHER_TRIAL_RUNBOOK.md'
)
git add -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Diff check failed' }
if ((Read-Host 'Commit and push the reviewed files? Type YES') -ne 'YES') { throw 'Stopped' }
git commit -m 'Add five direct tier3 multi-teacher embedding trials'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect status before retrying' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; stop and paste output. Do not force-push.' }
git rev-parse HEAD
```

Do not merge arbitrary runtime changes merely to satisfy a source guard. The
new implementation is additive relative to `d98359f76abe3a46c70f948aa4b99d51c82787e9`.
If origin advanced, reconcile and reverify the integration explicitly.

## Queue on sporcsubmit

There are **six jobs and zero preflight jobs**: one actual TRAIN-target export,
then five independent fits, all submitted now. Each fit has an `afterok`
dependency on export. No second manual queue step or login controller is needed.
Export authenticates the teachers, creates the four shared compact target banks
and snapshots comparison predictions; it does not run a training benchmark.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_multi5_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_multiteacher_trial.sh"
BASH
```

Reserve at least **4GiB personally available persistent space for the entire
campaign**, not4GiB per arm. The helper performs a dry review (including Slurm
`--test-only` resource checks, which do not queue jobs). Paste the printed
64-character `PLAN_HASH` **value**, then type YES to authorize export and all
five fits. Watch the printed log until all six accepted IDs appear. A helper
PID alone is not confirmation of submission. Do not rerun after a partial or
ambiguous submission; preserve the exact journal and inspect it first.

```bash
squeue --me -o "%.18i %.12P %.32j %.2t %.10M %R"
```

Jobs are named `lukamt_export_targets`, `lukamt_multi_w25`, `lukamt_multi_w5`,
`lukamt_multi_w10`, `lukamt_u100x4_w5`, `lukamt_multi_w5_rel`. Existing jobs,
holds, debug campaigns and Fresh_check are untouched. Slurm may run the fits
concurrently or serially according to available resources.

All jobs request tier3, account reu-aisocial, qos_tier3, one A100,6CPU,90000MiB.
Export requests6h; fits request max(6h, twice the inherited w5 request plus1h),
bounded by24h. These are conservative estimates using accepted SPORC evidence,
NOT new measured multi-head/relational timings. Without a new preflight there
is additional resource/runtime risk. Input, environment, numerical and native
checkpoint-readback checks still fail closed inside production workers.

## Results, including partial completion

Use the same commit as submission:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Campaign 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_multi5_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_multi5_${COMMIT:0:8}_tier3_r1"
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_multiteacher_trial.py" results --spec "${ROOT}/campaign_spec.json"
BASH
```

This reads saved authenticated artifacts only, not particle arrays, inference
or training. It lists incomplete fits separately. For live progress, tail the
specific arm's printed Slurm output: each epoch logs validation AUC, running
best/epoch, CE, each head's raw embedding loss and relational loss/ramp.

All AUC/accuracy/recovery/R30/R50 results use the same full-validation-selected
checkpoint. Rejections include QCD passing counts; zero passing is censored,
not infinity. Original50k never selects a checkpoint. M0HLT/OFFLINE/OFFLINE-w5
and optional U100-w5 are historical matched references, not new fits.
This is an adaptive single-seed validation screen; improvement is not guaranteed.

Keep shared target banks (~0.57GiB vector payload), five selected native models,
predictions, histories and receipts until comparisons are done. No optimizer
snapshots, duplicate particle caches, attention matrices or automatic deletions
are created. Deleting hashed outputs would break result authentication.

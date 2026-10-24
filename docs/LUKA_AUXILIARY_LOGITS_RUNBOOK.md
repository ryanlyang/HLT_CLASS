# c50 auxiliary-logit follow-up: debug

[Plan](plans/LUKA_AUXILIARY_LOGITS_PLAN.md) |
[Contract](contracts/LUKA_AUXILIARY_LOGITS.md).

Four new from-scratch HLT-only DE000 students; no earlier rung retraining.
`k0/k025/k05/k1`: total auxiliary-logit weights 0 / 0.25 / 0.5 / 1,
main CE weight 0.5, and total embedding weight 5. Teachers are OFFLINE and
selected **c50** UE100/DE066/DE033.
The main head receives no KD. All auxiliary heads are training-only.
30 epochs, physical batch 128; 299,234 train / 74,895 validation, no independent test.

The CE coefficient stays at 0.5 to match the parent c50 experiment; "CE-only"
means no KD acts on that head, not that its coefficient is 1. The zero-logit
arm still predicts all four embeddings, making it the matched control for
adding auxiliary logits. The original c50 result is a historical reference.

## Before queueing

Parent c50 DE000 must be complete, along with all c50 ancestors. c25/c75/anneal
need not finish. Parent selected native models and metadata must remain intact.
Original TRAIN target retirement is supported; the new campaign exports its
own banks. No old jobs/files need cancelling, moving, editing or deleting.
Reserve at least 3 GiB personally available home quota. No separate preflight.
The shared export includes a short disposable safety check, not an extra epoch.
Reuse existing measured c50 acceptance; new full-population timing is unmeasured.

## Commit and push locally

Use selective staging; preserve unrelated dirty work. PowerShell:

```powershell
$paths = @(
  'src/hlt_classification/luka_auxiliary_logits',
  'scripts/luka_auxiliary_logits.py',
  'scripts/queue_luka_auxiliary_logits.sh',
  'sbatch/run_luka_auxiliary_logits.sh',
  'tests/test_luka_auxiliary_logits.py',
  'docs/plans/LUKA_AUXILIARY_LOGITS_PLAN.md',
  'docs/contracts/LUKA_AUXILIARY_LOGITS.md',
  'docs/LUKA_AUXILIARY_LOGITS_RUNBOOK.md'
)
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Review staged whitespace' }
git diff --cached --stat
# Inspect before committing. Shared HANDOFF/LEGACY_SOURCE_MAP contain unrelated
# work: stage only the intended new sections separately with git add -p.
git commit -m 'Add c50 auxiliary-logit DE000 follow-up'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before retrying' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; stop, do not force-push' }
git rev-parse HEAD
```

## Queue from sporcsubmit

Use the new 40-character pushed commit. This creates a new clean pinned worktree;
does not update old worktrees or jobs. All prompts read `/dev/tty`.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_auxlogit_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_auxiliary_logits.sh"
BASH
```

Helper defaults to parent
`checkpoints/luka_trial_hybrid_72490a9d_debug_r1/campaign_spec.json`.
If its directory differs, pass the exact parent spec as the helper's first
argument; never select a different model chain to bypass a missing artifact.
Review the six-job dry plan, paste its printed 64-character PLAN_HASH, and approve YES.
The nohup helper submits the exact entire DAG, then exits. Fits depend on
successful targets/safety publication; report depends on all four fits.
No manual follow-up or persistent login controller is needed.

Submission order: `lukaal_targets`, `lukaal_k1`, `lukaal_k05`, `lukaal_k025`, `lukaal_k0`,
`lukaal_report`. GPU jobs: 8h minimum, 6 CPU / 90,000 MiB / A100 / debug;
report: 2h / 16 GiB CPU.
The four weights are submitted in descending order: **1, 0.5, 0.25, 0**.
They all depend only on targets; this does not serialize the fits or guarantee
Slurm's start order. Available resources and scheduler priority still apply.
Requests are conservative allowances, not a runtime guarantee. Do not change
partition, batch, source, teacher, memory or loss weights in a queued spec.
Do not blindly repeat after partial/ambiguous submission; preserve journals.

## Progress/results

Each fit prints `AUXLOGIT arm=... epoch=... auc=... best_auc=... best_epoch=...`
plus raw CE, weighted auxiliary sums and batch progress. Full per-teacher
losses remain in `<arm>/fit/training_report.json`. Main-head validation only.

The helper runs in its own shell, so its environment does not persist in your
login shell. To display results later, use:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the new study path printed during queueing: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
test -s "${SPEC}"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
PROJECT="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_auxiliary_logits.py" results --spec "${SPEC}"
BASH
```

Results work during fitting after targets complete, with pending arms explicit.
Reports retain full-validation/original50k metrics and recoveries, selected
epochs, and delta versus original c50 DE000. No test-selected winner claims.

After all four consumers finish, report publishes results and retires ONLY
the four new TRAIN payloads, preserving manifests, models and predictions.
No backup of those reproducible payloads. A partial-retirement error must be
investigated, not bypassed by deleting the journal or making a fake receipt.

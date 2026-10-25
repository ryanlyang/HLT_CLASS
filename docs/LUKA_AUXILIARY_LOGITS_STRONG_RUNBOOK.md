# Queue k5 and k10 in debug, without a separate preflight

[Plan](plans/LUKA_AUXILIARY_LOGITS_STRONG_PLAN.md) |
[Contract](contracts/LUKA_AUXILIARY_LOGITS_STRONG.md).

Exactly two new fresh HLT DE000 fits. CE0.5, embedding total5, auxiliary KD
total5 or10 (1.25 or2.5 per teacher), T2. Same frozen c50 teachers, seeds,
batch128 and30 epochs. This does NOT implement the frozen-classifier strategy.
Current jobs remain untouched. No new standalone preflight.

There are four scheduler jobs: shared target export, k5, k10, CPU report.
The export includes short disposable numerical/resource checks. Both fits
automatically start after its successful authenticated completion; neither
depends on the other. Submit k5 first; Slurm start order is not guaranteed.

## 1. Commit and push locally

Use PowerShell. Preserve unrelated dirty/staged work; do not use git add -A.
HANDOFF and LEGACY_SOURCE_MAP contain unrelated edits: their new sections may
be staged separately with git add -p after review.

```powershell
& {
    $alreadyStaged = @(git diff --cached --name-only)
    if ($LASTEXITCODE -ne 0 -or $alreadyStaged.Count -gt 0) {
        throw 'Review existing staged work before continuing.'
    }
    $paths = @(
        'src/hlt_classification/luka_auxiliary_logits_strong',
        'scripts/luka_auxiliary_logits_strong.py',
        'scripts/queue_luka_auxiliary_logits_strong.sh',
        'sbatch/run_luka_auxiliary_logits_strong.sh',
        'tests/test_luka_auxiliary_logits_strong.py',
        'docs/plans/LUKA_AUXILIARY_LOGITS_STRONG_PLAN.md',
        'docs/contracts/LUKA_AUXILIARY_LOGITS_STRONG.md',
        'docs/LUKA_AUXILIARY_LOGITS_STRONG_RUNBOOK.md'
    )
    git add -- $paths
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Review staged whitespace' }
    git diff --cached --stat
    $answer = Read-Host 'Commit and push these changes? Type YES'
    if ($answer -ne 'YES') { return }
    git commit -m 'Add direct debug auxiliary-logit weights 5 and 10'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before retrying' }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; stop, do not force-push' }
    git rev-parse HEAD
}
```

## 2. Queue from sporcsubmit

Keep original c50 selected teachers, reports and pinned source intact. Its other
arms need not finish. No current auxiliary-logit result/bank is a dependency.
Reserve at least3GiB personal home quota; export independently creates new banks.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve everything and paste the error." >&2' ERR
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_auxstrong_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_auxiliary_logits_strong.sh"
BASH
```

The helper creates `checkpoints/luka_auxlogit_strong_<commit8>_debug_r1`.
Review the printed four-job plan; paste the actual64-character PLAN_HASH value,
not its label, then YES. Prompts read /dev/tty. The detached helper submits the
whole DAG and exits; no manual post-export submission or long-lived controller.
Preserve journals after an interrupted/ambiguous attempt; do not blindly retry.

Defaults: original parent
`/home/ryreu/atlas/HLT_Classification/checkpoints/luka_trial_hybrid_72490a9d_debug_r1/campaign_spec.json`.
Pass a different exact location as the helper's first argument only if this same
immutable campaign was relocated; never substitute another teacher chain.

Expected jobs: `lukaas_targets`, `lukaas_k5`, `lukaas_k10`, `lukaas_report`.
GPU jobs: debug/A100/6CPU/90000MiB,8h minimum (or larger inherited allowance).
Report:1CPU/16GiB/2h. These are inherited conservative allowances, not new
remote throughput measurements. Do not edit registered batch/partition/losses.

## 3. Progress and results

Fit logs print `AUXLOGIT_STRONG arm=k5 epoch=... auc=... best_auc=...` (or k10).
All30 epochs/per-teacher losses remain in `<arm>/fit/training_report.json`.
For saved results, run on SPORC:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Study path printed during queueing: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
test -s "${SPEC}"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
PROJECT="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_auxiliary_logits_strong.py" results --spec "${SPEC}"
BASH
```

Results include c50 DE000 and historical CE/OFFLINE/embedding references;
pending new arms are explicit. k1 remains in its separate original campaign.
Full74,895 validation selects models; original50k report-only; no independent test.

After both fits and the report validate, the report retires only four new TRAIN
target payloads. No backup; they require re-export if ever needed again. Native
models, predictions, histories, manifests/moments and logs remain. Old jobs,
campaign payloads, datasets and Fresh_check are untouched.

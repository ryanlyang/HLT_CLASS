# Native embedding prediction ladder — SPORC debug

Plan: [LUKA_EMBEDDING_PREDICTION_PLAN.md](plans/LUKA_EMBEDDING_PREDICTION_PLAN.md).
Contract: [LUKA_EMBEDDING_PREDICTION.md](contracts/LUKA_EMBEDDING_PREDICTION.md).

Frozen original U000 → new U100 → new D066 → new D033 → new D000.
Each fresh student trains CE + **5 × embedding prediction loss** for 30 epochs,
batch128, the usual fresh-training 3e-4 warmup/cosine schedule. A small head
predicts its predecessor's selected native 128-D pre-classifier embedding.
The next stage targets the student's native embedding, **not its predictions**.
There is no logit KD, warm start, U050 fit or fixed-OFFLINE target bank.
Every teacher is frozen and every target bank is normalized on TRAIN only.

Split: 299,234 train / 74,895 validation; original50k is report-only. Historical
test was retired; this is single-seed exploratory development, not an independent
test. Only final D000 is deployable HLT-only; intermediate richer views are oracle
diagnostics. Historical M0HLT/OFFLINE/U000 are references, not new causal controls.

## Commit and push locally

Donor baseline: `118aa375eea423e5e99338cd9a6eda198225ef21`; reviewed integration
baseline: `b21c2d4e01211a31e57d879c5a86ca90aab2ca0b` (only a separate FullStat
package added; scientific donors unchanged). Existing runtime must remain
unchanged. Review staged work first; don't use `git add .` or
force-push. Select HANDOFF/source-map hunks separately if desired.

```powershell
$paths = @(
  'src/hlt_classification/luka_embedding_prediction',
  'scripts/luka_embedding_prediction.py',
  'scripts/queue_luka_embedding_prediction.sh',
  'sbatch/run_luka_embedding_prediction.sh',
  'tests/test_luka_embedding_prediction.py',
  'docs/plans/LUKA_EMBEDDING_PREDICTION_PLAN.md',
  'docs/contracts/LUKA_EMBEDDING_PREDICTION.md',
  'docs/LUKA_EMBEDDING_PREDICTION_RUNBOOK.md'
)
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
# Review before committing; preserve unrelated staged work.
git commit -m "Add immediate-predecessor native embedding prediction ladder"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect output' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push' }
git rev-parse HEAD
```

## Queue once on sporcsubmit

Requires the original fixed30 campaign, its pinned worktree/receipts and
selected U000/M0HLT/OFFLINE models, ALLJETS metadata/assignments/raw data and
prepared U100/D066/D033/D000 caches. Fully deleted U000 cache blocks are rebuilt
in RAM using the original assignments/fingerprints. No new particle cache is
written. Partial/corrupt caches are not silently rebuilt. No original U100
model, older attention bank, or completion of another experiment is required.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_embedpred_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_embedding_prediction.sh"
BASH
```

The helper creates `checkpoints/luka_embedding_prediction_<commit8>_debug_r1`,
authenticates inputs, asks for at least 2 GiB personally available space, shows
the full dry plan, and asks for the actual 64-character PLAN_HASH and YES.
It submits **five jobs**: `lukaepl_setup`, `lukaepl_U100`, `lukaepl_D066`,
`lukaepl_D033`, `lukaepl_D000`. Every fit depends on setup, and later stages
also depend on their predecessor. No second manual submission/login controller
is needed. Wait for all five exact job IDs in the printed submission log.
Never blindly retry a partial/uncertain submission or edit its immutable plan.

Setup runs genuine native/installed-Weaver parity, U000 validation readback,
TRAIN-only target export, maximum-support stresses and a disposable full-
population one-epoch miniature of all four stages. It does NOT initialize
scientific students. Science uses newly selected scientific teachers only.
All jobs request 6 CPUs, 90,000 MiB, one A100 on debug; setup cap8h, each fit
cap6h. Measured time/memory margins must pass; poor scores never fail admission.
Local CUDA tests do not replace this genuine SPORC setup.

No existing jobs, old campaigns, shared files or Fresh_check are changed.

## Progress and results

```bash
squeue --me -o "%.18i %.12P %.32j %.2t %.10M %R"
```

Logs in the printed root are `U100-<job>.out`, `D066-<job>.out`, etc. They print
`NATIVE_EMBED node=... epoch=... auc=... best_auc=... best_epoch=...` every epoch.
Full histories and selected predictions live in `<coordinate>/fit/`.
After setup passes, set the exact PROJECT and ROOT printed above, then:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_embedding_prediction.py" results --spec "${ROOT}/campaign_spec.json"
```

Reporting authenticates retained hashes and predictions without loading a model;
prints completed stages, pending stages, accuracy, AUC, recovery and rejection50
on both validation populations. A running stage's log supplies provisional AUCs;
it is not a completed authenticated result until its receipt exists.

Storage is compact (~1.1 GiB of seven TRAIN banks before compression, plus small
selected models/predictions/metadata; reserved2GiB). No per-epoch models or
attention tensors. These small lineage-bound artifacts remain available for
reporting; don't delete them midway. No broad automatic cleanup is performed.

## Code map

`src/hlt_classification/luka_embedding_prediction/`: `contracts.py` registers
the exact recipe/chain; `source.py` authenticates the original U000 anchor;
`data.py` loads exact inputs; `targets.py` exports native TRAIN representations;
`training.py` implements fresh CE+5 prediction; `worker.py` measures setup,
reads back selected native checkpoints, exports the next targets and reports;
`campaign.py` binds immutable inputs; `submission.py` renders the exact DAG.
The CLI and Slurm wrapper are thin. New v1 artifacts are distinct from the older
fixed-OFFLINE/logit-KD embedding ladder. Focused tests are in
`tests/test_luka_embedding_prediction.py`.

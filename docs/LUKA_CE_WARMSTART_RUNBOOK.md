# Debug: matched CE warm start, with and without U100 embedding loss

Plan: [LUKA_CE_WARMSTART_PLAN.md](plans/LUKA_CE_WARMSTART_PLAN.md).
Contract: [LUKA_CE_WARMSTART.md](contracts/LUKA_CE_WARMSTART.md).

Both new fits start independently from the existing selected fixed30 M0HLT
checkpoint, including its classifier. CE_CONTINUE uses CE only;
CE_EMBED_U100_W5 uses CE + 5 * frozen COARSE_U100 embedding prediction.
Both train all native parameters for another 30 epochs, batch128, fresh AdamW,
3e-5 peak LR / 1.5e-6 floor, existing warmup3/cosine schedule scaled by 0.1.
This is NOT the earlier embedding-only20 -> finetune30 experiment.

The primary comparison is against the new continued-CE control. Historical
M0HLT, OFFLINE, OFFLINE-w5 and U100 are context, not extra-training-matched
controls. The split is 299,234/74,895, with original50k report-only and no
independent test. No teacher retraining or logit KD. Inference remains native
HLT-only; the auxiliary head is discarded.

## Commit and publish locally

Implementation is additive relative to ed6d0af4be65f059356543f90583b30185702615.
Do not use `git add .`: this workspace contains unrelated edits. Review and
commit only these new implementation/documents/tests (plus any deliberately
selected handoff/source-map hunks). Do not force-push on a rejection.

```powershell
$paths = @(
  'src/hlt_classification/luka_ce_warmstart',
  'scripts/luka_ce_warmstart.py',
  'scripts/queue_luka_ce_warmstart.sh',
  'sbatch/run_luka_ce_warmstart.sh',
  'tests/test_luka_ce_warmstart.py',
  'docs/plans/LUKA_CE_WARMSTART_PLAN.md',
  'docs/contracts/LUKA_CE_WARMSTART.md',
  'docs/LUKA_CE_WARMSTART_RUNBOOK.md'
)
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
# Inspect the staged files before committing. Preserve unrelated staged work.
git commit -m "Add matched CE warmstart and U100 embedding debug comparison"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect output' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push' }
git rev-parse HEAD
```

## Queue once on SPORC

Prerequisites: parent fixed30 campaign and native M0HLT/COARSE_U100 selected
checkpoints, original pinned source worktrees and receipts, ALLJETS prepared
D000 and U100 train/validation caches, historical OFFLINE-w5 saved results.
Do not delete those dependencies. This does not depend on completion of the
earlier U10020->30 or direct-tier3 U100-w5 experiments.

Paste on **sporcsubmit**, with the new full pushed commit when prompted:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_cewarm_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_ce_warmstart.sh"
BASH
```

The helper authenticates inputs, reserves a declared minimum 2 GiB, creates a
fresh `checkpoints/luka_ce_warmstart_<commit8>_debug_r1`, prints the complete dry
plan and requests its actual 64-character PLAN_HASH and YES. Three jobs are
submitted: `lukace_setup`, `lukace_ce`, `lukace_embed`. Both science jobs have
`afterok` on setup, plus worker-side receipt/resource/accounting validation.
No persistent login controller or manual science follow-up is needed. Check
the printed submission log for all three exact job IDs before disconnecting.
Never blindly rerun after a partial/uncertain submission.

Setup is a genuine production-worker miniature, not a scientific fit: full
teacher/CE probability readbacks, TRAIN-only target export, installed parity,
maximum-support stresses, and one disposable full-population epoch per arm.
Neither science fit inherits probe weights. Setup is capped at 8h, each fit at
6h, all on debug with 6 CPUs, 90,000 MiB and one A100. Admission uses measured
memory/time with margins; scientific performance cannot fail acceptance.
Do not manually shorten/move jobs without reviewing their immutable plan.

No current jobs are canceled or held. No Fresh_check paths are touched.

## Progress and saved results

```bash
squeue --me -o "%.18i %.12P %.32j %.2t %.10M %R"
```

Each science log prints `CE_WARMSTART arm=... epoch=... auc=... best_auc=...`
for every epoch. The root contains `ce-<job>.out` and `embed-<job>.out`.
Completed reports/predictions are under `ce/fit` and `embed/fit`; one selected
native checkpoint per arm, no per-epoch checkpoints or attention caches.

After setup passes, use the exact project/root printed by the helper:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
# Set PROJECT and ROOT to the exact paths printed for this campaign.
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_ce_warmstart.py" results --spec "${ROOT}/campaign_spec.json"
```

Reporting is read-only, validates retained hashes/predictions and displays
completed arms while marking others pending. It prints accuracy, AUC, recovery
and AUC delta against continued CE for full validation and original50k. Do not
interpret repeated single-seed development selection as independent-test proof.

## Code and testing

`luka_ce_warmstart/contracts.py` registers science; `campaign.py` and `source.py`
bind inputs; `targets.py` imports the unchanged TRAIN-only U100 bank contract;
`training.py` restores native weights, resets optimization and isolates arms;
`worker.py` measures setup, runs fits and reports; `submission.py` publishes the
exact three-job DAG. CLI/Slurm workers are thin. No old runtime files changed.

Focused tests: `tests/test_luka_ce_warmstart.py`, plus the unchanged U100 direct
and pretraining suites. Local CUDA tests are installed-Weaver parity and a
synthetic 22-row production-worker miniature; they do not replace the real
SPORC population/setup and are not a claim of cluster acceptance.

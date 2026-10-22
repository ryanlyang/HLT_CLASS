# U100 embedding prediction: direct tier3, weight 5

One fresh HLT fit: **30 epochs CE + 5 x frozen regular COARSE_U100 embedding
prediction from epoch 1**. No logit KD, no embedding-only pretraining, no warm
start, no new preflight. This does not replace or cancel the debug20+30 run or
D066 run. All scheduling holds are left alone.

One A100 / 6 CPUs / 90,000 MiB on tier3 (`reu-aisocial`, `qos_tier3`). Walltime
is inherited OFFLINE-w5 request + 120-minute setup allowance, at least 6 hours
and at most 24 hours; this is not a new U100 measurement. Reserve 2 GiB
persistent storage. Shared prepared U100 and D000 caches must still exist.

## Commit and push locally

Only stage this additive implementation; do not use `git add .`. Existing
unrelated HANDOFF and other edits should be reviewed separately.

```powershell
& {
    $files = @(
        'src/hlt_classification/luka_u100_embedding',
        'scripts/luka_u100_embedding.py',
        'scripts/queue_luka_u100_embedding.sh',
        'sbatch/run_luka_u100_embedding.sh',
        'tests/test_luka_u100_embedding.py',
        'docs/plans/LUKA_U100_EMBEDDING_PLAN.md',
        'docs/contracts/LUKA_U100_EMBEDDING.md',
        'docs/LUKA_U100_EMBEDDING_RUNBOOK.md'
    )
    git add -- $files
    if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
    git diff --cached --stat
    if ((Read-Host 'Commit and push the reviewed staged files? Type YES') -cne 'YES') { return }
    git commit -m 'Add direct tier3 U100 embedding weight5 without new preflight'
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed; stop and inspect' }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push. Preserve and inspect.' }
    git rev-parse HEAD
}
```

## Queue from SPORC

Use the new pushed commit, not dd51e8e2 (the earlier debug two-stage code).
This creates a separate root and shows a dry plan. Enter the printed
64-character `DIRECT_PLAN_HASH` value and confirm one real job.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve logs and journals." >&2' ERR
read -r -p 'Paste the NEW pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_u100embed_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_embedding.sh"
BASH
```

The helper prints the submission log and study root
`checkpoints/luka_u100_direct_<commit8>_tier3_r1`. A helper PID alone is NOT proof
of submission; inspect its log for the job ID and `squeue --me`. The job is
named `lukau1e_direct`. No follow-up controller is necessary.

Authentication checks the retained regular COARSE_U100 checkpoint in
`luka_fixed30_74aa58c5_r1/science`, the unchanged parent controls, and completed
OFFLINE-w5 in `luka_embedding_sweep_007885f7_tier3_r1/science_debug` using their
original source worktrees. No D066 checkpoint is needed. A missing teacher or
reference fails closed; it must never silently select another model.

## Progress and saved results

The log emits `U100_EMBED weight=5 epoch=... auc=... best_auc=... best_epoch=...`
for every finished epoch, plus phase heartbeats during setup. The native HLT
classifier has no prediction head at inference. After completion, in the same
environment (fill in exact paths printed by the helper):

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Pinned PROJECT directory: ' PROJECT </dev/tty
read -r -p 'Study ROOT directory: ' ROOT </dev/tty
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -B -s "${PROJECT}/scripts/luka_u100_embedding.py" results \
    --spec "${ROOT}/campaign_spec.json"
BASH
```

Results show CE, OFFLINE oracle, OFFLINE-w5, U100 oracle and the new HLT student,
including AUC recovery. Full validation selects the checkpoint; original50k is
report-only. This is an exploratory single-seed comparison, not an independent
test. Retain compact targets, selected native checkpoint, predictions and
receipts; no automatic deletion is performed. Do not restart a partial run or
change its immutable plan: preserve the output and diagnose first.

## Donors

At dd51e8e25770482134809dbb34d4766f8dac74b9: D066 `direct.py`, `contracts.py`,
`targets.py`, `training.py`, `worker.py`, `preflight.py:teacher_readback`, queue
helper and wrapper; U100 pretrain `source.py`; native auxiliary head, standardizer,
fixed30 recipe, original-source replay and exact-DAG submission utilities.
These are same-repository donors, not a new Fresh_check migration.

Local verification: 40 new tests passed, including production-architecture
installed-Weaver/CUDA parity and a reduced-model target-export/30-epoch/native
readback miniature. Combined new/D066/pretrain regression: 149 passed, 1
deselected in 72.13s. The deselected pre-existing D066 committed-tree test
rejects the already-committed U100_PRETRAIN addition at dd51e8e2; that old
source guard was not changed. Bash and PowerShell runbook syntax were checked.
No new remote validation or submission was performed.

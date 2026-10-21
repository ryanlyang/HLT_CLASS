# Eight additional no-KD embedding weights on SPORC tier3

Authority: [plan](plans/LUKA_EMBEDDING_SWEEP_PLAN.md),
[contracts](contracts/LUKA_EMBEDDING_SWEEP.md).

Eight fresh single-HLT models: `CE + lambda * SmoothL1`, lambda =
**3, 5, 20, 50, 100, 200, 500, 1000**. No repeats of 1.5 or 10.
No KD or ladder. Same trial ALLJETS 299,234 / 74,895, batch 128, fixed 30 epochs,
seeds, teacher embeddings and full-val selection as the existing 0.3 experiment.
No independent test. Reserve 2 GiB total, without duplicating the shared bank.
Both the shared preflight and all eight scientific jobs use **tier3**.
Create a fresh tier3 root; do not edit or reuse an old debug gate or plan.
Recommended: opt into automatic follow-up once. The full preflight runs first;
after authenticated success the controller queues all eight fits without you
waiting at the terminal. It does not skip any gate checks.

Local verification with automatic follow-up: **142 tests passed in 215.26s**
(33 sweep,31 new follow-up,23 donor follow-up,18 original weight,22 auxiliary,
15 ladder). This includes installed local Weaver/CUDA
production-model preflight and all eight full30 miniature scientific fits,
source/resource/storage/corruption guards, exact tier3 submission fixtures and
failed/missing/ambiguous-gate, duplicate-controller and partial-submit cases.
Shell/runbook syntax, CLI help and documentation links were also checked.
Local tests and genuine cluster preflight acceptance are separate. Do not queue
science until the new eight-weight preflight is accepted. No old jobs or runs
are overwritten. See HANDOFF for detailed evidence.

## 1. Commit/push on Windows

Stage only this experiment, not unrelated work. Review the staged diff.

```powershell
git add -- src/hlt_classification/luka_embedding_sweep `
  scripts/luka_embedding_sweep.py scripts/queue_luka_embedding_sweep.sh `
  sbatch/run_luka_embedding_sweep.sh tests/test_luka_embedding_sweep.py `
  tests/test_luka_embedding_sweep_followup.py `
  docs/plans/LUKA_EMBEDDING_SWEEP_PLAN.md docs/contracts/LUKA_EMBEDDING_SWEEP.md `
  docs/LUKA_EMBEDDING_SWEEP_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw "Stage failed" }
git diff --cached --stat
# Optionally stage only the new HANDOFF/LEGACY_SOURCE_MAP sections with git add -p.
git commit -m "Add tier3 embedding sweep with automatic verified preflight followup"
if ($LASTEXITCODE -ne 0) { throw "Inspect status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Stop; do not force-push" }
git rev-parse HEAD
```

## 2. Create and arm preflight plus automatic science on sporcsubmit

Keep the original target bank and its pinned source worktree intact. This uses
the completed `luka_embedding_cfb3a158_debug_r1` run, not the embedding ladder.
That completed reference stays at its original debug path. The new source must
preserve existing runtime relative to donor/integration commit
`bf360343974121397a568fdb635a2c1c1011bc90`. The original three-weight campaign
is not a required parent and must stay on its own unchanged worktree.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve artifacts and paste the error." >&2' ERR
read -r -p 'Paste the NEW pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_embedsweep_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_embedding_sweep_${COMMIT:0:8}_tier3_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
EMBED="${MAIN}/checkpoints/luka_embedding_cfb3a158_debug_r1/science/campaign_spec.json"
test -f "${PARENT}"
test -f "${EMBED}"
test ! -e "${ROOT}"
df -h /home/ryreu
read -r -p 'Personally available GiB reserved for all eight runs (at least 2): ' QUOTA </dev/tty
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
python -u -B -s "${PROJECT}/scripts/luka_embedding_sweep.py" create \
    --root "${ROOT}" --parent "${PARENT}" --embedding "${EMBED}" \
    --expected-commit "${COMMIT}" --quota-gib "${QUOTA}"
printf '\nPROJECT=%s\nROOT=%s\n' "${PROJECT}" "${ROOT}"
bash "${PROJECT}/scripts/queue_luka_embedding_sweep.sh" "${PROJECT}" "${ROOT}" auto
BASH
```

Inspect the dry follow-up plan: one full preflight and eight tier3 fits, with
24h per-fit resource ceilings. Paste the actual 64-character
`FOLLOWUP_PLAN_HASH` value (not that literal label), then confirm YES. The helper
prints a log; run its `tail -f` command and wait for:

```text
ARMED: preflight <jobid>; after verified success, all 8 tier3 science jobs will submit automatically.
```

Only the preflight is initially in squeue. After its COMPLETED/0:0 accounting
and full evidence validation, the controller saves the measured exact dry plan
and submits all eight fits. It logs `ALL SCIENCE JOBS QUEUED` with their IDs;
this means submitted, not finished training. Ctrl-C stops tail only. Once ARMED,
you can disconnect SSH: nohup continues on the submit host. A reboot or killed
controller is not automatically recovered. Polling is every 60s, bounded at 14 days.

**Do not also submit science manually or start a second controller.** A failed
gate, corrupt receipt, changed source or resource failure stops follow-up.
Poor validation scores do not. Inspect `followup/failure.json` and the printed
log if it stops. All accepted IDs remain in exact submission journals; never
blindly rerun after an uncertain/partial submission. No jobs are cancelled.

If the dry review was already created but no live controller launched, do not
rerun the fresh-root creation block. Set PROJECT/ROOT to their printed exact
values and rerun only the helper's `auto` command. An existing gate can be
bound only through its exact successful submission journal, not by guessing
an ID. An already-armed or partially submitted controller needs inspection.

## 3. Manual alternative only (not needed with auto)

If you prefer two approvals, use `gate` instead of `auto` in step2, then use
`science` after that gate passes. Do not use this section if auto is armed.
Set `PROJECT` and `ROOT` to the exact values printed above; they are not
preserved outside the previous Bash block.

```bash
bash "${PROJECT}/scripts/queue_luka_embedding_sweep.sh" "${PROJECT}" "${ROOT}" science
```

This authenticates COMPLETED/0:0 preflight evidence, creates a science spec,
dry-reviews **eight** commands, then asks for its hash and YES. All use tier3,
one A100, 6 CPUs, 90000 MiB and independently measured walltime capped at 24h.
They can run independently as resources allow. Names: `lukaes_w3`, `lukaes_w5`, `lukaes_w20`, `lukaes_w50`, `lukaes_w100`,
`lukaes_w200`, `lukaes_w500`, `lukaes_w1000`. An uncertain/partial submission must be inspected;
do not create another campaign to bypass a journal.

## 4. Compare completed weights, recoveries, CE and existing 0.3

In the same activated environment (or reactivate as in step 2):

```bash
python -u -B -s "${PROJECT}/scripts/luka_embedding_sweep.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

Incomplete arms print PENDING/INCOMPLETE. Per-epoch CE, auxiliary loss and
validation metrics are in `science/<arm>/fit/training_report.json`; live epoch
lines appear in `science/<arm>-<jobid>.out`. Results never load a model or need
the old particle caches. Preserve the new preflight/report files, reference
snapshots and selected checkpoints. No old or shared artifacts are deleted.

This is an exploratory eight-weight sweep with one seed, not a statistically
independent test. Report all weights, including poor outcomes.

## 5. Read live epoch progress

Every scientific epoch logs AUC, best AUC/epoch so far, CE, unweighted auxiliary
loss and weighted auxiliary loss. The live best is provisional, not authenticated
completion. After setting ROOT to the new root above:

```bash
for ARM in w3 w5 w20 w50 w100 w200 w500 w1000; do
    printf '\n=== %s ===\n' "${ARM}"
    grep -h '^EMBED weight=' "${ROOT}/science/${ARM}-"*.out 2>/dev/null | tail -n 1 || true
done
```

The results command reports all completed arms while others are pending. Keep
the selected epoch even if epoch30 is worse; never select on original50k.
Use the original three-weight results command/worktree separately for 0.75,
1.5 and 10. Nothing here imports provisional log scores as scientific evidence.

## 6. Storage and safe handoff

Reserve 2 GiB for this entire new sweep, not per arm. It reuses the same authenticated
target bank and prepared HLT inputs read-only. Keep the original 0.3 source,
target bank and matched control provenance intact. The new campaign saves
only one selected native checkpoint per fit plus compact predictions/history
and preflight evidence, with no per-epoch state or new particle/target cache.
No cleanup commands run automatically. A missing shared parent must be inspected,
not silently rebuilt under a different scientific identity.

Only the listed new files plus the corresponding HANDOFF/LEGACY_SOURCE_MAP
sections should be committed. Do not stage unrelated repository work. A rejected
push needs review/integration, never force-push. Queue helper PID alone is not
Slurm acceptance; for auto wait for ARMED, then inspect the printed log and
exact submission ledger. Automatic continuation adds only compact metadata,
not another GPU allocation, bank copy or particle cache.

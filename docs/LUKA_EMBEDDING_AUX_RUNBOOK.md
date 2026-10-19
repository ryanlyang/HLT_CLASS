# Queue the single-stage HLT offline-embedding prediction test

Authority: [plan](plans/LUKA_EMBEDDING_AUX_PLAN.md) and
[v1 contracts](contracts/LUKA_EMBEDDING_AUX.md).

This is the **Luka trial ALLJETS** split, 299,234 TRAIN / 74,895 validation,
not Oscar FullStat. One new HLT model gets CE plus 0.3 times a small-head
offline-embedding prediction loss. No ladder, logit KD, attention KD or warm
start. The matched completed fixed30 M0HLT is the CE reference. No independent
test remains; original50k validation is a secondary report, not a selector.

Default: SPORC **tier3**, one A100, six CPUs, 90,000 MiB. Explicit `debug`
selection is supported, but the preflight and science must use the same site.
Reserve 2 GiB of personally available persistent storage. No shared artifacts
are deleted. Compact TRAIN targets (~146 MiB before identities/compression),
predictions, reports and the selected native HLT checkpoint are retained.

## 1. Commit and push only this implementation

In local PowerShell, review the changes before staging:

```powershell
git diff --stat
git status --short -- src/hlt_classification/luka_embedding_aux scripts/luka_embedding_aux.py scripts/queue_luka_embedding_aux.sh sbatch/run_luka_embedding_aux.sh tests/test_luka_embedding_aux.py docs/plans/LUKA_EMBEDDING_AUX_PLAN.md docs/contracts/LUKA_EMBEDDING_AUX.md docs/LUKA_EMBEDDING_AUX_RUNBOOK.md

git add -- src/hlt_classification/luka_embedding_aux scripts/luka_embedding_aux.py scripts/queue_luka_embedding_aux.sh sbatch/run_luka_embedding_aux.sh tests/test_luka_embedding_aux.py docs/plans/LUKA_EMBEDDING_AUX_PLAN.md docs/contracts/LUKA_EMBEDDING_AUX.md docs/LUKA_EMBEDDING_AUX_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
```

`docs/HANDOFF.md` and `docs/LEGACY_SOURCE_MAP.md` also contain new entries and
unrelated work. Use `git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md` to
stage only the new embedding entries after inspection. Do not use `git add .`.
Then:

```powershell
git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    git commit -m "Add matched HLT auxiliary offline embedding experiment"
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed; stop' }
} elseif ($LASTEXITCODE -ne 0) {
    throw 'Cannot inspect staged changes'
}
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; stop and resolve without force-pushing' }
git rev-parse HEAD
```

Do not guess a commit. Use the **new 40-character pushed hash**. Source guards
freeze existing scientific runtime against donor
`62c9b2df91eb3bda4fd4df4edc75c5f810a48c55`; a later conflicting runtime change
requires compatibility review, not bypassing the guard.

## 2. Create a fresh SPORC gate

Run on **sporcsubmit**, not Tigris. This creates metadata but submits no jobs.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_lukaembed_${COMMIT:0:8}"
PARTITION=tier3
# For explicit debug, change the line above BEFORE creating the gate.
ROOT="${MAIN}/checkpoints/luka_embedding_${COMMIT:0:8}_${PARTITION}_r1"
PARENT="${MAIN}/checkpoints/luka_fixed30_74aa58c5_r1/science/campaign_spec.json"
test -f "${PARENT}"
test ! -e "${ROOT}"
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
df -h /home/ryreu
read -r -p 'Personally available GiB reserved for this run (at least 2): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
python -u -B -s "${PROJECT}/scripts/luka_embedding_aux.py" create \
    --root "${ROOT}" --parent "${PARENT}" --expected-commit "${COMMIT}" \
    --quota-gib "${QUOTA}" --partition "${PARTITION}"
printf '\nGate created. Queue with:\nbash "%s/scripts/queue_luka_embedding_aux.sh" "%s" "%s" gate\n' \
    "${PROJECT}" "${PROJECT}" "${ROOT}"
BASH
```

M0HLT and OFFLINE must still have their original selected checkpoints, training
reports, validation predictions and receipts. They are authenticated in the
original pinned worktree. The deleted OFFLINE input caches and old attention
banks are **not** required. Raw ROOT files, prepared metadata and D000 caches
must remain. Do not delete any parent or new-run paths to overcome an error.

## 3. Queue the preflight

Run the exact `bash ... gate` command printed above. Inspect the dry plan, paste
the **64-character value** after `PLAN_HASH=`, then type `YES`. The helper starts
a disconnect-safe submission process, not a persistent campaign controller.
Watch its log until an exact Slurm job ID/submission ledger appears. A helper
PID or `sbatch --test-only` estimate is not a queued job.

The one preflight job reconstructs only OFFLINE TRAIN in RAM, replays its input
fingerprint, exports compact frozen-teacher targets, checks the installed
model and auxiliary gradients, stresses the longest TRAIN/validation support,
and runs one full-population technical epoch. Technical weights never seed the
scientific fit. Poor metrics do not fail this gate. Check:

```bash
squeue --me -o '%.18i %.12P %.35j %.2t %.10M %R'
# Replace JOB_ID with the exact returned preflight ID.
sacct -j JOB_ID -X -P -o JobID,JobName%40,State,ExitCode,Elapsed
```

## 4. Queue the single scientific fit after successful preflight

Reuse the exact PROJECT and ROOT paths printed in step 2:

```bash
bash "${PROJECT}/scripts/queue_luka_embedding_aux.sh" "${PROJECT}" "${ROOT}" science
```

The helper authenticates the preflight report, target payloads, exact submission
journal and COMPLETED/0:0 accounting before creating the science spec. It then
prints a fresh dry plan and asks for its hash/YES. This submits **one new fit**,
not a rerun of M0HLT/OFFLINE. Runtime is a measured conservative request capped
at 24 hours. If that bound fails, preserve evidence; do not change epochs or
limits in an existing spec. No automatic retry/resume is implemented.

## 5. Progress and results

`tail -f "${ROOT}/science/fit-JOB_ID.out"` shows every epoch's validation AUC,
CE loss, auxiliary loss and train time. After successful completion, with the
same Conda/PYTHONPATH environment as step 2:

```bash
python -u -B -s "${PROJECT}/scripts/luka_embedding_aux.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

The saved result has M0HLT, OFFLINE oracle and HLT_EMBED_AUX, full validation and
original50k metrics, QCD rejection, absolute deltas and recovery. The printed
summary emphasizes the matched CE comparison. All metrics are recomputed from
authenticated saved predictions; no model inference or raw data is needed.
The native deployable checkpoint is `science/fit/selected.pt`; it contains no
prediction-head parameters. Single-seed development gains are exploratory,
not independent-test evidence or a guarantee of improvement.

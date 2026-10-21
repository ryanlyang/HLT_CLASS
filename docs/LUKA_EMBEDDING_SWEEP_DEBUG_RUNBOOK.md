# Move the pending trial embedding sweep to debug

This is the eight-weight **3/5/20/50/100/200/500/1000** trial sweep, not the
embedding ladder or the Oscar FullStat sweep. It reuses the accepted tier3
preflight, same A100/environment, 6 CPUs, 90000 MiB and measured per-fit times.
No extra GPU preflight, particle preparation, target copies or training changes.
See the [plan](plans/LUKA_EMBEDDING_SWEEP_DEBUG_PLAN.md) and
[contract](contracts/LUKA_EMBEDDING_SWEEP_DEBUG.md).

The helper cancels only journal-bound jobs from these two specified campaigns:

- Eight original pending sweep jobs, 21937191 through 21937198 (each exact ID
  is obtained from the authenticated journal, never a numeric cancellation range).
- Remaining pending/running `lukaag` alternating-gate jobs in
  `luka_alternating_bf360343_r1`. Completed jobs and all files are preserved.

The unfinished running alternating fit is lost. No `evmsg`, Fresh_check, other
campaign, dataset or model file is cancelled/deleted. If a sweep fit has started
or has output, stop for a separate decision; do not delete it to bypass checks.
Debug admission does not guarantee immediate start or exclusive use of a node.
The old sweep uses exact IDs plus Slurm's controller-side pending-state filter;
see the [official scancel options](https://slurm.schedmd.com/scancel.html).

## 1. Windows: commit and push this additive transfer

Do not use `git add .`. Preserve unrelated staged/unstaged work. Review the
existing index first; if it contains unrelated paths, resolve that separately.
The listed files are the complete required queue implementation; the new
HANDOFF/LEGACY_SOURCE_MAP entries may be staged separately with `git add -p`.

```powershell
git diff --cached --name-only
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect index' }
if (git diff --cached --name-only) { throw 'Index is not empty; inspect staged work first' }
git add -- src/hlt_classification/luka_embedding_sweep/debug.py src/hlt_classification/luka_embedding_sweep/debug_retirement.py scripts/luka_embedding_sweep_debug.py scripts/queue_luka_embedding_sweep_debug.sh sbatch/run_luka_embedding_sweep_debug.sh tests/test_luka_embedding_sweep_debug.py docs/plans/LUKA_EMBEDDING_SWEEP_DEBUG_PLAN.md docs/contracts/LUKA_EMBEDDING_SWEEP_DEBUG.md docs/LUKA_EMBEDDING_SWEEP_DEBUG_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
git commit -m "Add authenticated debug transfer for pending embedding sweep"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before proceeding' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; preserve work and do not force-push' }
git rev-parse HEAD
```

## 2. SPORC submit host: review, cancel exact old jobs, queue replacements

Run on **sporcsubmit**, not Tigris. Paste the new full pushed commit. The helper
first authenticates the original gate and all eight submissions, creates a fresh
`science_debug` specification, and dry-reviews the debug plan. No existing
specification or journal is rewritten. Next it prints a separate cancellation
plan with live identity checks. Read it before typing its hash and confirmation.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character transfer commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_embedsweep_debug_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_embedding_sweep_007885f7_tier3_r1"
ALT="${MAIN}/checkpoints/luka_alternating_bf360343_r1/science/campaign_spec.json"
test -f "${ROOT}/science/campaign_spec.json"
test -f "${ALT}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_embedding_sweep_debug.sh" "${PROJECT}" "${ROOT}" "${ALT}"
BASH
```

There are two independent approvals: `CANCEL REVIEWED JOBS` for cancellation,
then `YES` for eight new debug submissions. The helper uses `/dev/tty`; it does
not call Python `input()` on the piped script. It prints a detached submission
log. Confirm **eight numeric Slurm IDs**, not only a helper PID. New names are
`lukaesd_w3` etc. Never also submit the original tier3 campaign manually.

## Accounting delay after successful cancellation

`RETIRE_COMPLETE` records successful cancellation requests, not proof that
termination has reached accounting. If the helper stops specifically because
confirmation is unavailable, inspect `retirement/receipt.json` and `sacct`.
Do not repeat cancellation. After accounting catches up, use this read-only
confirmation plus fresh dry-review/submission block. Missing or ambiguous live
submission claims require inspection; this is not a retry for a failed sbatch.

```bash
bash <<'BASH'
set -euo pipefail
ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/luka_embedding_sweep_007885f7_tier3_r1
SPEC="${ROOT}/science_debug/campaign_spec.json"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
PROJECT="$("${PYTHON}" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_embedding_sweep_debug.py"
python -u -B -s "${CLI}" confirm --spec "${SPEC}"
python -u -B -s "${CLI}" submit --spec "${SPEC}"
read -r -p 'Printed PLAN_HASH (64 hex): ' HASH </dev/tty
[[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit the eight debug replacements? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/science_debug/submit.XXXXXX.log")"
nohup python -u -B -s "${CLI}" submit --spec "${SPEC}" --execute \
    --plan-hash "${HASH}" --authorization AUTHORIZE_LUKA_EMBEDDING_SWEEP_DEBUG_V1 \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nWatch: tail -f "%s"\n' "$!" "${LOG}"
BASH
```

If cancellation partially failed or has an intent without a complete receipt,
preserve the directory and inspect the exact IDs/logs; no automatic retry.
If an original started racing cancellation, the replacement is blocked even
if the original ended CANCELLED. Ask before restarting or changing this policy.

## Results

In the same activated environment and with PROJECT/SPEC from above:

```bash
python -u -B -s "${PROJECT}/scripts/luka_embedding_sweep_debug.py" results --spec "${SPEC}"
```

Prints completed weights against matched CE/OFFLINE/original0.3, plus pending
arms, AUC deltas and recovery on validation and overlapping original50k.
No model inference or independent test evaluation is performed by reporting.
Only one selected native state, predictions, epoch histories and compact
receipts per new fit are kept. The original 2 GiB total allowance also covers
the new subtree; no automatic deletion is performed.

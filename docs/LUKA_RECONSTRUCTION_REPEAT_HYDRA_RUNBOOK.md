# Queue eighteen reconstruction repeats on Hydra

This campaign runs **C, R24, R20, R22, R28 and D20 at fresh seed indices4,5,6**.
Every fit uses4 GTX1080 Ti GPUs,12 CPUs,64000MiB, FP32, global batch128 and30
training epochs. CE + U100-w5 stays active throughout. See the
[scientific plan](plans/LUKA_RECONSTRUCTION_REPEAT_HYDRA_PLAN.md).

Bootstrap submits two jobs: `setup_pascal` and its dependent CPU
`release_pascal`. After successful qualification, release automatically queues
all18 independent fits plus an afterany report. You can log out once both
bootstrap IDs are saved. Setup is capped at6h; fit walltimes are measured and
capped at48h. No A6000 jobs or new data exports are needed.

## Commit and push locally

Review existing staged changes first. Stage only this new campaign; shared
README/handoff/source-map hunks can be reviewed separately because they contain
unrelated work. In PowerShell, run this as one block so a failed check stops
before commit/push:

```powershell
& {
    git add -- src/hlt_classification/luka_reconstruction_repeat_hydra scripts/luka_reconstruction_repeat_hydra.py scripts/queue_luka_reconstruction_repeat_hydra.sh sbatch/run_luka_reconstruction_repeat_hydra.sh tests/test_luka_reconstruction_repeat_hydra.py docs/plans/LUKA_RECONSTRUCTION_REPEAT_HYDRA_PLAN.md docs/contracts/LUKA_RECONSTRUCTION_REPEAT_HYDRA.md docs/LUKA_RECONSTRUCTION_REPEAT_HYDRA_RUNBOOK.md docs/LUKA_RECONSTRUCTION_REPEAT_HYDRA_VALIDATION.md
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
    git diff --cached --stat
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged checks failed' }
    git commit -m "Add eighteen paired reconstruction repeats on four GTX1080 Ti GPUs"
    if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
    git push origin main
    if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
    git rev-parse HEAD
}
```

Keep the old scientific checkout58d0c0ed, benchmark1094dea4 and all their
source/data/results/environment directories. The new CLI loads the original
scientific modules from that immutable checkout and its own runtime separately.

## Register and queue on Brown

Paste the following with the full pushed commit. It creates a clean detached
checkout and a fresh output root; registration may take several minutes while
it authenticates source and saved evidence. Declare at least13 personally
available GiB (12GiB output allowance plus1GiB reserve).

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed reconstruction-repeat commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
BENCH="$BASE/runs/r24_bench_1094dea4_1080ti_r1/benchmark_spec.json"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$BENCH")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$BENCH")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_reconstruction_repeat_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="$BASE/runs/reconstruction_repeat_${COMMIT:0:8}_4x1080ti_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for new outputs (at least 13): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
LOG="$(mktemp "$BASE/runs/reconstruction_repeat_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_reconstruction_repeat_hydra.py" create \
    --root "$ROOT" --benchmark-spec "$BENCH" --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$LOG" 2>&1; then
    tail -n 100 "$LOG"
    printf 'Registration stopped; preserve root and log.\n' >&2
    exit 1
fi
bash "$PROJECT/scripts/queue_luka_reconstruction_repeat_hydra.sh" "$ROOT/campaign_spec.json"
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

The helper dry-reviews setup, release and the full scientific envelope before
submitting. No second prompt or interactive polling process is needed after
setup. If registration/submission fails, preserve the root, logs and journals;
inspect the failure before retrying. Exact accepted jobs are journaled under
`submission_bootstrap/` and `submission_science_pascal/`. No broad cancellation
or automatic retry is performed.

Job names start `lukarr_`; log names include `setup_pascal-JOBID.out`,
`release_pascal-JOBID.out`, `R24_s4-JOBID.out`, etc. Setup checks all six recipes,
then measures every fit. Numerical reference snapshots, miniatures and resource
profiles are saved in `setup_pascal/`. Poor finite accuracy does not prevent
scientific release; invalid execution or an exceeded resource bound does.

## Completed results and epoch-by-epoch recovery

Use the setup job ID to locate this exact campaign, including after it finishes:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'reconstruction-repeat setup job ID: ' JOB </dev/tty
[[ "$JOB" =~ ^[1-9][0-9]*$ ]]
shopt -s nullglob
LOGS=("$HOME"/hlt_u100_pairs/runs/reconstruction_repeat_*/setup_*-"$JOB".out)
[[ ${#LOGS[@]} -eq 1 ]]
SPEC="$(dirname -- "${LOGS[0]}")/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_reconstruction_repeat_hydra.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_reconstruction_repeat_hydra.py" progress --spec "$SPEC"
BASH
```

`results` shows authenticated completed runs and per-recipe mean/minimum recovery,
matched-control deltas, completion counts and late-window stability. `progress`
shows every saved scientific epoch, selected-so-far recovery, best-control and
same-epoch deltas, and encoder/decoder coefficients. Setup/startup miniatures
are excluded. Each `*_s4` row compares with `C_s4`, and similarly for5/6.

The fixed late window is epochs21-30;52% counts/streaks describe individual
checkpoints. Missing controls/unfinished seeds are explicit, and a partial
window is not presented as a complete-window average. All results are
exploratory validation using historical recovery anchors, with no independent
test. Three seeds test repeatability; they do not guarantee52% recovery.

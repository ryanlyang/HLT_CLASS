# Queue 42 reconstruction stability fits on Hydra

This campaign runs **14 configurations at fresh seed indices7,8,9:42 fits**.
The five control configurations are C128,C_L50,C_L25,C256,C512. Reconstruction
arms are R24,G30,R_L50,R_L25,PROJ,DEC2,R256,R512,SOFT. Every fit uses4 GTX1080 Ti,
12 CPUs,64000MiB, FP32 and30 epochs. CE + U100-w5 stays active. Global128
microbatches preserve SyncBatchNorm populations;256/512 arms accumulate2/4.
See the [scientific plan](plans/LUKA_RECONSTRUCTION_STABILITY_IDEAS.md) and
[validation record](LUKA_RECONSTRUCTION_STABILITY_HYDRA_VALIDATION.md).

Bootstrap submits two jobs: `setup_pascal` and its dependent CPU
`release_pascal`. After successful qualification, release automatically queues
all42 independent fits plus an afterany report. You can log out once both
bootstrap IDs are saved. Setup is capped at6h; fit walltimes are measured and
capped at48h. There are45 jobs in total. No A6000 jobs or new data exports are needed.

## Commit and push locally

Review existing staged changes first. Stage only this new campaign; shared
README/handoff/source-map hunks can be reviewed separately because they contain
unrelated work. In PowerShell, run this as one block so a failed check stops
before commit/push:

```powershell
& {
    git add -- src/hlt_classification/luka_reconstruction_stability_hydra scripts/luka_reconstruction_stability_hydra.py scripts/queue_luka_reconstruction_stability_hydra.sh sbatch/run_luka_reconstruction_stability_hydra.sh tests/test_luka_reconstruction_stability_hydra.py docs/plans/LUKA_RECONSTRUCTION_STABILITY_IDEAS.md docs/contracts/LUKA_RECONSTRUCTION_STABILITY_HYDRA.md docs/LUKA_RECONSTRUCTION_STABILITY_HYDRA_RUNBOOK.md docs/LUKA_RECONSTRUCTION_STABILITY_HYDRA_VALIDATION.md
    if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
    git diff --cached --stat
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged checks failed' }
    git commit -m "Add 42 paired reconstruction stability fits with automatic Hydra release"
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
it authenticates source and saved evidence. Registration serializes the current native/full model to calculate a42-fit
output inventory including all four readouts, startup miniatures and setup.
At the existing298-slot support, the local estimate is29GiB plus1GiB reserve:
declare at least30GiB that are personally available. The installed serializer
is authoritative and may request more. Existing inputs are retained in place.
Setup also measures actual artifact sizes and refuses automatic release if
its conservative projection exceeds the registered allowance.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed reconstruction-stability commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
BENCH="$BASE/runs/r24_bench_1094dea4_1080ti_r1/benchmark_spec.json"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$BENCH")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$BENCH")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_reconstruction_stability_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="$BASE/runs/reconstruction_stability_${COMMIT:0:8}_4x1080ti_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for new outputs (at least 30 for 298 slots): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
LOG="$(mktemp "$BASE/runs/reconstruction_stability_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_reconstruction_stability_hydra.py" create \
    --root "$ROOT" --benchmark-spec "$BENCH" --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$LOG" 2>&1; then
    tail -n 100 "$LOG"
    printf 'Registration stopped; preserve root and log.\n' >&2
    exit 1
fi
bash "$PROJECT/scripts/queue_luka_reconstruction_stability_hydra.sh" "$ROOT/campaign_spec.json"
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

The helper dry-reviews setup, release and the full scientific envelope before
submitting. No second prompt or interactive polling process is needed after
setup. If registration/submission fails, preserve the root, logs and journals;
inspect the failure before retrying. Exact accepted jobs are journaled under
`submission_bootstrap/` and `submission_science_pascal/`. No broad cancellation
or automatic retry is performed.

Job names start `lukars_`; log names include `setup_pascal-JOBID.out`,
`release_pascal-JOBID.out`, `R24_s7-JOBID.out`, etc. Setup checks all14 configurations and all42 fresh-seed miniatures, measures
both decoder catch-up and ordinary phases, soft temperatures, full particle
support, partial/accumulated batches, all four validation readouts and BN
calibration. Complete common-start serial/distributed reference tensors are
broadcast in memory; their hashes and numerical comparisons are saved along
with miniatures and resource profiles in `setup_pascal/`. Poor finite accuracy does not prevent
scientific release; invalid execution or an exceeded resource bound does.

## Completed results and epoch-by-epoch recovery

The current `results`/`progress` paths authenticate existing artifacts without
checking the reserve for future training outputs. This read-only fix does not
retroactively modify an older pinned CLI. A completed campaign also has a
saved summary at `report_pascal/results.json`, bound by `report_pascal/receipt.json`.

Use the setup job ID to locate this exact campaign, including after it finishes:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'reconstruction-stability setup job ID: ' JOB </dev/tty
[[ "$JOB" =~ ^[1-9][0-9]*$ ]]
shopt -s nullglob
LOGS=("$HOME"/hlt_u100_pairs/runs/reconstruction_stability_*/setup_*-"$JOB".out)
[[ ${#LOGS[@]} -eq 1 ]]
SPEC="$(dirname -- "${LOGS[0]}")/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_reconstruction_stability_hydra.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_reconstruction_stability_hydra.py" progress --spec "$SPEC"
BASH
```

`results` shows authenticated completed runs and per-recipe mean/minimum/sample-SD recovery,
matched-control deltas, completion counts and late-window stability. `progress`
shows every saved scientific epoch, selected-so-far recovery, best-control and
same-epoch deltas, and encoder/decoder coefficients. Setup/startup miniatures
are excluded. Each row uses its declared same-seed control and the same readout. The default
is RAW. Add `--readout EMA05`, `--readout EMA10`, `--readout RAW_BN` or
`--readout all` to either command. JSON includes all readouts even when the
printed table selects one. RAW_BN isolates TRAIN BN recalibration; compare
EMA to RAW_BN when attributing an averaging benefit. These are correlated
readouts of each fit, not extra independent seeds.

The fixed late window is epochs21-30;50%/52% counts/streaks describe individual
checkpoints. Missing controls/unfinished seeds are explicit, and a partial
window is not presented as a complete-window average. All results are
exploratory validation using historical recovery anchors, with no independent
test. Three seeds test repeatability; they do not guarantee52% recovery.

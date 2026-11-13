# Queue the fourteen-fit R24 diagnostic campaign on Hydra

Eight fits use 4 GTX1080 Ti GPUs each; six use 1 A6000 each. See the
[exact paired run matrix](plans/LUKA_R24_DIAGNOSTIC_HYDRA_PLAN.md).
There are no four-A6000 jobs. Every fit retains CE + U100 embedding weight5.

Bootstrap submits `setup_pascal`, `release_pascal`, `setup_a6000` and
`release_a6000`. Each CPU release waits for its own successful setup, then
queues its independent fits and an afterany report. The A6000 lane cannot
hold up Pascal science. You can log out once all four bootstrap IDs are saved.
Setup is capped at 6h per lane; fit limits are measured and capped at 48h.
No new data export is required; keep the original and benchmark directories.

## Commit and push locally

Stage only the new campaign files; the shared README/handoff/source map may
already contain unrelated work. Review those hunks separately rather than
staging the whole repository. In PowerShell:

```powershell
git add -- src/hlt_classification/luka_r24_diagnostic_hydra scripts/luka_r24_diagnostic_hydra.py scripts/queue_luka_r24_diagnostic_hydra.sh sbatch/run_luka_r24_diagnostic_hydra.sh tests/test_luka_r24_diagnostic_hydra.py docs/plans/LUKA_R24_DIAGNOSTIC_HYDRA_PLAN.md docs/contracts/LUKA_R24_DIAGNOSTIC_HYDRA.md docs/LUKA_R24_DIAGNOSTIC_HYDRA_RUNBOOK.md docs/LUKA_R24_DIAGNOSTIC_HYDRA_VALIDATION.md
git diff --cached --stat
git diff --cached --check
git commit -m "Add fourteen R24 diagnostic fits with independent Hydra release lanes"
git push origin main
git rev-parse HEAD
```

Review the staged diff before committing if other changes were already staged.
The production detached checkout must be completely clean and the exact full
commit must be reachable from origin/main. Keep both original worktrees and
the benchmark/results/data directories; no new data export is required.

## Register and queue on Brown

Paste this complete block into your Brown terminal. It uses the completed
benchmark173643 and the original source/environment recorded there.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed R24-diagnostic commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
BENCH="$BASE/runs/r24_bench_1094dea4_1080ti_r1/benchmark_spec.json"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$BENCH")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$BENCH")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_r24_diagnostic_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="$BASE/runs/r24_diagnostic_${COMMIT:0:8}_mixed_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for new outputs (at least 10): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
LOG="$(mktemp "$BASE/runs/r24_diagnostic_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_r24_diagnostic_hydra.py" create \
    --root "$ROOT" --benchmark-spec "$BENCH" --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$LOG" 2>&1; then
    tail -n 100 "$LOG"
    printf 'Registration stopped; preserve root and log.\n' >&2
    exit 1
fi
bash "$PROJECT/scripts/queue_luka_r24_diagnostic_hydra.sh" "$ROOT/campaign_spec.json"
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

Registration authenticates old science and benchmark evidence and can take
several minutes. The queue helper dry-reviews the complete bounded DAG and
submits it under the campaign authorization already given. It does not ask you
to return after setup. If a command fails, preserve its output and journals;
do not blindly re-run into an existing root or delete a claim.

Logs use prefix `lukard_`. Setup logs are `setup_pascal-JOBID.out` and
`setup_a6000-JOBID.out`; fit logs are `P0_R-JOBID.out`, etc. Miniature epochs
are technical; the readers below show only scientific epochs.

## Completed results and epoch-by-epoch recovery

Run this with either **new setup job ID**. It locates the exact campaign through
that log, even after the setup job has finished.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'R24-diagnostic setup job ID: ' JOB </dev/tty
[[ "$JOB" =~ ^[1-9][0-9]*$ ]]
shopt -s nullglob
LOGS=("$HOME"/hlt_u100_pairs/runs/r24_diagnostic_*/setup_*-"$JOB".out)
[[ ${#LOGS[@]} -eq 1 ]]
SPEC="$(dirname -- "${LOGS[0]}")/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_r24_diagnostic_hydra.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_r24_diagnostic_hydra.py" progress --spec "$SPEC"
BASH
```

Recovery uses historical M0HLT/OFFLINE anchors. Every `_R` row compares with
its matching `_C` row, with best-so-far and same-epoch differences in progress.
The P0/PD/PO/PI pairs run on four Pascal cards; OB is the original BF16 worker,
NB is the new BF16 worker, NF is the new FP32 worker on one A6000. Missing
controls print `--`. Results are exploratory validation, not independent tests.

The runbook requires 10 personally available GiB (9 GiB allowance plus 1 GiB
reserve). Numerical reference snapshots are retained alongside setup evidence.
If either setup fails, its lane stays unreleased and preserves diagnostics;
the other lane can continue. Do not delete claims or retry an ambiguous submit.

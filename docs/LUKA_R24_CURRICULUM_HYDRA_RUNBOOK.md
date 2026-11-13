# Queue the twenty-run R24 curriculum on Hydra

Each fit uses **4 GTX1080Ti GPUs,12 CPUs and64000MiB** on one node. This is a
new campaign with D01–D20 IDs; old R24/R8 jobs and artifacts stay separate.
Every fit retains native CE1+U100 embedding weight5. See the
[exact run table](plans/LUKA_R24_CURRICULUM_HYDRA_PLAN.md).

One command sequence submits GPU setup and a durable CPU release job. After
setup passes, release automatically submits all20 independent fits and an
afterany report. You can log out once both bootstrap IDs are recorded. Setup
has a6h cap, individual fits a48h cap; actual fit requests are measured. The
benchmark's ~12h R24 projection is an estimate, not a guarantee or queue time.

## Commit and push locally

Stage only the new campaign files; the shared README/handoff/source map may
already contain unrelated work. Review those hunks separately rather than
staging the whole repository. In PowerShell:

```powershell
git add -- src/hlt_classification/luka_r24_curriculum_hydra scripts/luka_r24_curriculum_hydra.py scripts/queue_luka_r24_curriculum_hydra.sh sbatch/run_luka_r24_curriculum_hydra.sh tests/test_luka_r24_curriculum_hydra.py docs/plans/LUKA_R24_CURRICULUM_HYDRA_PLAN.md docs/contracts/LUKA_R24_CURRICULUM_HYDRA.md docs/LUKA_R24_CURRICULUM_HYDRA_RUNBOOK.md docs/LUKA_R24_CURRICULUM_HYDRA_VALIDATION.md
git diff --cached --stat
git diff --cached --check
git commit -m "Add twenty R24 curricula with four-GPU Hydra automatic release"
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
read -r -p 'New pushed R24-curriculum commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
BENCH="$BASE/runs/r24_bench_1094dea4_1080ti_r1/benchmark_spec.json"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$BENCH")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$BENCH")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_r24_curriculum_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="$BASE/runs/r24_curriculum_${COMMIT:0:8}_4x1080ti_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for new outputs (at least 9): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
LOG="$(mktemp "$BASE/runs/r24_curriculum_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_r24_curriculum_hydra.py" create \
    --root "$ROOT" --benchmark-spec "$BENCH" --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$LOG" 2>&1; then
    tail -n 100 "$LOG"
    printf 'Registration stopped; preserve root and log.\n' >&2
    exit 1
fi
bash "$PROJECT/scripts/queue_luka_r24_curriculum_hydra.sh" "$ROOT/campaign_spec.json"
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

Registration authenticates old science and benchmark evidence and can take
several minutes. The queue helper dry-reviews the complete bounded DAG and
submits it under the campaign authorization already given. It does not ask you
to return after setup. If a command fails, preserve its output and journals;
do not blindly re-run into an existing root or delete a claim.

Logs use prefix `lukarc_`. Setup log is `setup-JOBID.out`; fit logs are
`D02-JOBID.out`, etc. Both setup and fit miniatures log their technical epochs
before scientific epochs; the results reader separates them.

## Completed results and epoch-by-epoch recovery

Run this with the **new setup job ID**. It locates the exact campaign through
that log, even after the setup job has finished.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'R24-curriculum setup job ID: ' JOB </dev/tty
[[ "$JOB" =~ ^[1-9][0-9]*$ ]]
shopt -s nullglob
LOGS=("$HOME"/hlt_u100_pairs/runs/r24_curriculum_*/setup-"$JOB".out)
[[ ${#LOGS[@]} -eq 1 ]]
SPEC="$(dirname -- "${LOGS[0]}")/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_r24_curriculum_hydra.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_r24_curriculum_hydra.py" progress --spec "$SPEC"
BASH
```

Recovery uses historical M0HLT/OFFLINE anchors. Delta compares D03 for seed1
and D01 for seed0, with both best-so-far and same-epoch comparisons in progress.
A missing fresh control prints `--`; it is never silently replaced by A6000 C0.
These are selected development-validation results, not independent test gains.

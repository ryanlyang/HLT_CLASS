# Benchmark one versus two GTX 1080 Ti GPUs on Hydra

This submits **one job: two GTX 1080 Ti GPUs on one node, six CPUs, 32 GB RAM,
at most two hours**. It measures one-card then two-card FP32 execution of the
real subclass/U100 losses. The global batch stays 128. The two-card phase uses
DDP and synchronized BatchNorm. It does not submit the ten scientific fits.
See the [benchmark plan](plans/LUKA_SUBCLASS_GPU_BENCHMARK_PLAN.md).

No SPORC export is needed. Reuse the completed teacher-sweep T00 artifacts,
original U100 bank, D000 inputs and environment. Keep all original worktrees
and the unsuccessful A6000 registration. Allow at least 3 GiB of personally
available space for this separate benchmark.

## Commit and push locally

In the repository PowerShell terminal, review and stage this campaign's files.
The list includes the pending L40/L40S profile support on which this snapshot
is based. Shared documentation is staged interactively to preserve other work.

```powershell
$paths = @(
    'src/hlt_classification/luka_subclass_hydra',
    'scripts/luka_subclass_hydra.py',
    'scripts/queue_luka_subclass_hydra.sh',
    'sbatch/run_luka_subclass_hydra.sh',
    'scripts/luka_subclass_gpu_benchmark.py',
    'scripts/queue_luka_subclass_gpu_benchmark.sh',
    'sbatch/run_luka_subclass_gpu_benchmark.sh',
    'tests/test_luka_subclass_hydra.py',
    'tests/test_luka_subclass_gpu_benchmark.py',
    'docs/plans/LUKA_SUBCLASS_HYDRA_PLAN.md',
    'docs/plans/LUKA_SUBCLASS_GPU_BENCHMARK_PLAN.md',
    'docs/contracts/LUKA_SUBCLASS_HYDRA.md',
    'docs/LUKA_SUBCLASS_HYDRA_RUNBOOK.md',
    'docs/LUKA_SUBCLASS_GPU_BENCHMARK_RUNBOOK.md',
    'docs/LUKA_SUBCLASS_HYDRA_VALIDATION.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin ($paths + @('README.md','docs/HANDOFF.md','docs/LEGACY_SOURCE_MAP.md')) -and -not $_.StartsWith('src/hlt_classification/luka_subclass_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Benchmark one versus two GTX 1080 Ti GPUs for subclass training'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Register and queue on Brown

This checks the pushed commit, makes a fresh detached worktree, authenticates
the parent inputs, performs Slurm dry review and asks for `YES` to submit the
one bounded job. Registration may take several minutes; its log is printed.
The source checker requires the complete scoped commit above. Do not patch
the existing c8e15195 worktree or change its registered GPU request.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
PARENT="${BASE}/runs/teacher_sweep_6fe7e349_a6000_r1/campaign_spec.json"
test -f "$PARENT"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$PARENT")"
read -r -p 'New pushed 40-character benchmark commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_subclass_bench_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="${BASE}/runs/subclass_bench_${COMMIT:0:8}_1080ti_r1"
test ! -e "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
LOG="$(mktemp "${BASE}/runs/subclass_bench_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s \
    "${PROJECT}/scripts/luka_subclass_gpu_benchmark.py" create \
    --root "$ROOT" --teacher-sweep-spec "$PARENT" --expected-commit "$COMMIT" \
    >"$LOG" 2>&1; then
    tail -n 100 -- "$LOG"
    printf 'Registration stopped; preserve its log and output root.\n' >&2
    exit 1
fi
printf 'Registered benchmark: %s/benchmark_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_subclass_gpu_benchmark.sh" \
    "${ROOT}/benchmark_spec.json" </dev/tty
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

Expect one `lukascb_probe` job. After its accepted ID is printed, logout is
safe. Slurm selects an eligible node; there is no guarantee of immediate start.
The first phase uses only one of the two reserved cards intentionally, so
both measurements have the same allocation and node. The second uses both.

If `YES` was mistyped before submission, rerun only the queue helper with the
existing spec. A partial submission or interrupted worker requires inspecting
the exact journal/log; do not erase claims or submit a duplicate blindly.

## Inspect progress and the summary

Find exactly one benchmark (or replace `SPECS` with its full path). `tail -f`
can follow `probe-JOBID.out`, `world1.log` or `world2.log`. Ctrl-C stops the log
viewer without cancelling the Slurm job.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/subclass_bench_*_1080ti_r1/benchmark_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact benchmark; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
ROOT="$(dirname -- "$SPEC")"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOGS=("$ROOT"/probe-*.out "$ROOT"/world1.log "$ROOT"/world2.log)
if [[ ${#LOGS[@]} -gt 0 ]]; then tail -n 12 -- "${LOGS[@]}"; fi
if [[ -f "$ROOT/report.json" ]]; then
    "${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_subclass_gpu_benchmark.py" \
        results --spec "$SPEC"
else
    printf 'Benchmark report not published yet; inspect the logs above.\n'
fi
BASH
```

Review parity results, per-rank memory, longest-batch support, per-step timing
and both phase exit codes. The 30-epoch number is a training-only projection,
not a safe walltime request. `2/speedup` is idealized independent-run throughput;
two concurrent independent fits were not measured. Parity checks disable
dropout; timings enable it. The benchmark does not report recovery or choose
scientific checkpoints.

Real two-GPU qualification is pending until this job runs. A promising report
would justify a production FP32 execution profile and a fresh matched U100-w5
control; the original BF16 campaign must not be silently switched to Pascal.

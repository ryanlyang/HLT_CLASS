# R24 benchmark: two, three and four GTX1080Ti GPUs

This queues one two-hour benchmark job on Brown Hydra: four GTX1080Ti GPUs,
twelve CPUs and64000MiB. It tests2/3/4 GPUs sequentially on the same node and
compares with the completed A6000 R24 run. No new science campaign or automatic
release follows. Four GPUs remain reserved throughout. Availability is checked
at submission; no particular node is required.

First commit and push the benchmark files listed in
[validation](LUKA_R24_GPU_BENCHMARK_VALIDATION.md), preserving unrelated work.
Use that new full40-character commit below. Keep the original58d0c0ed worktree,
environment, setup and R24 output intact. No new data export is needed.

Run in your **Brown terminal**:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed benchmark commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
ORIGINAL="$BASE/runs/next_R8_58d0c0ed_a6000_r1/campaign_spec.json"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$ORIGINAL")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_r24_bench_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="$BASE/runs/r24_bench_${COMMIT:0:8}_1080ti_r1"
bash "$PROJECT/scripts/queue_luka_r24_gpu_benchmark.sh" "$ORIGINAL" "$ROOT" "$COMMIT"
BASH
```

The dedicated script asks for personally available output space (minimum3GiB),
authenticates existing evidence, dry-reviews the one job, then submits it.
Registration can take several minutes while it verifies the completed run.
You may log out after receiving the exact job ID. Do not repeat submission into
an existing/partial root: preserve its journal and diagnose any failure.

To view results, substitute the benchmark job ID shown at submission:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Benchmark job ID: ' JOB </dev/tty
[[ "$JOB" =~ ^[0-9]+$ ]]
shopt -s nullglob
LOGS=("$HOME"/hlt_u100_pairs/runs/r24_bench_*/probe-"$JOB".out)
[[ ${#LOGS[@]} -eq 1 ]]
SPEC="$(dirname -- "${LOGS[0]}")/benchmark_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
"${ENV_DIR}/bin/python" -u -B -s "$PROJECT/scripts/luka_r24_gpu_benchmark.py" results --spec "$SPEC"
BASH
```

While running, the job root contains `prepare.log`, `reference.log` and
`world2.log`, `world3.log`, `world4.log` as each phase starts. The main Slurm log
prints phase boundaries. Read those logs for live progress; results requires
the completed report/receipt. A completed Slurm job can contain failed hardware
profiles, which the report retains explicitly.

Compare **total estimated hours** for turnaround, **active GPU-hours** for
efficiency, and require passing parity and all memory stress cases before
considering a production implementation. These FP32 Pascal measurements are
not precision-matched to BF16-native A6000. They estimate timing, not AUC or
recovery; production still requires its own qualified profile.

Timing samples start from fresh model initialization and bounded warmup, not
the completed R24 decoder state. Assignment costs can change during long
training, so these are estimates rather than guaranteed completion times.

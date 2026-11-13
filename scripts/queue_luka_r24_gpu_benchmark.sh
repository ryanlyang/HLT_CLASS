#!/usr/bin/env bash
# Run from the new pinned benchmark worktree on Brown after committing/pushing.
set -euo pipefail
trap 'printf "Benchmark stopped; preserve the root, logs and journals.\n" >&2' ERR
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ $# -eq 3 ]] || { echo 'Usage: bash scripts/queue_luka_r24_gpu_benchmark.sh ORIGINAL_SPEC NEW_ROOT FULL_COMMIT' >&2; exit 2; }
ORIGINAL="$1"
ROOT="$2"
COMMIT="$3"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
test ! -e "$ROOT"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$ORIGINAL")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
read -r -p 'Personally available GiB for benchmark output (at least 3): ' QUOTA </dev/tty
[[ "$QUOTA" =~ ^[1-9][0-9]*$ ]]
CLI="${PROJECT}/scripts/luka_r24_gpu_benchmark.py"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" create --root "$ROOT" --original-spec "$ORIGINAL" --expected-commit "$COMMIT" --quota-gib "$QUOTA"
SPEC="${ROOT}/benchmark_spec.json"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" review --spec "$SPEC"
HASH="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${ROOT}/submission/command_plan.json")"
printf 'Review hash: %s\nOne 4x GTX1080Ti job; 2/3/4-GPU bounded measurements, no scientific fits.\n' "$HASH"
# The user invoking this dedicated queue script authorizes this single job;
# the exact dry-run hash binds execution to what was just reviewed.
"${ENV_DIR}/bin/python" -u -B -s "$CLI" submit --spec "$SPEC" --plan-hash "$HASH" --authorization AUTHORIZE_R24_234_GPU_BENCHMARK_V1
squeue --me -o '%.18i %.30j %.12T %.12l %R'

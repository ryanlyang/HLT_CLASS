# Queue HLT capacity and complementarity on Hydra

This adds **12 fresh student fits, no teacher fits**, plus CPU analysis, GPU
setup, automatic CPU release and CPU report. Native128 and wide256 each run
CE-only, U100-w5, and U100-w5+IT-w5 at two matched seeds. The CPU analysis tests
fixed ensembles of completed S2IT/S4/S6 predictions.

Keep the completed selective campaign and all of its original parent artifacts
and worktrees. Normal execution needs no SPORC export or new data transfer.
The [plan](plans/LUKA_CAPACITY_HYDRA_PLAN.md),
[contract](contracts/LUKA_CAPACITY_HYDRA.md), and
[validation record](LUKA_CAPACITY_HYDRA_VALIDATION.md) define the study.

## 1. Commit and push from local PowerShell

Run from the repository. Review this scoped diff; it excludes unrelated runtime
changes currently in the workspace. Do not use `git add .`.

```powershell
$paths = @(
    'src/hlt_classification/luka_capacity_hydra',
    'scripts/luka_capacity_hydra.py',
    'scripts/queue_luka_capacity_hydra.sh',
    'sbatch/run_luka_capacity_hydra.sh',
    'tests/test_luka_capacity_hydra.py',
    'docs/plans/LUKA_CAPACITY_HYDRA_PLAN.md',
    'docs/contracts/LUKA_CAPACITY_HYDRA.md',
    'docs/LUKA_CAPACITY_HYDRA_RUNBOOK.md',
    'docs/LUKA_CAPACITY_HYDRA_VALIDATION.md',
    'README.md', 'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin $paths -and -not $_.StartsWith('src/hlt_classification/luka_capacity_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Add matched HLT capacity fits and saved prediction complementarity'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Register and queue from Brown

Paste the complete block. Enter the newly pushed40-character commit, at least9
GiB of personally available quota, and `YES` at the final bounded dry review.
Registration authenticates the original selected artifacts in their pinned
worktree and can be quiet for several minutes. Its printed log can be tailed
from another terminal.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
PARENT="${BASE}/runs/selective_teacher_3fb9e7f1_a6000_r1/campaign_spec.json"
test -f "$PARENT"
read -r -p 'New pushed capacity commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_capacity_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="${BASE}/runs/capacity_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for outputs (at least 9): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_capacity_hydra.py"
REG_LOG="$(mktemp "${BASE}/runs/capacity_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$CLI" create \
    --root "$ROOT" --selective-spec "$PARENT" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve the root and log.\n' >&2
    exit 1
fi
printf 'Registered spec: %s/campaign_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_capacity_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

Expect `lukacap_analysis`, `lukacap_setup`, `lukacap_release`. Once all three
IDs are recorded, logging out is safe. Successful setup automatically releases
all12 independent A6000 fits and the CPU report. The analysis can finish before
or after setup and does not block training. No later manual approval is needed.

Queue order per seed is `W_UI,W_CE,N_UI,N_CE,W_U,N_U`, seed0 then seed1.
Slurm chooses actual start order. There is no fit-to-fit dependency or array
concurrency cap. Report waits afterany so a failed fit can be listed alongside
completed results. Per-fit walltime is measured, bounded at48h; one typed
RTX A6000, six CPUs,32000MiB. Native/deployed compute is measured separately.

The setup performs real installed-Weaver parity, same-width reload, actual
target miniatures and resource qualification before any science fit is queued.
Finite poor accuracy never blocks the release. If setup fails technically,
keep its outputs/log and inspect; do not delete claims/journals to force retry.
If `YES` was mistyped before submission, rerun only the queue helper using the
same spec. `inspect-submission` reports accepted IDs after interrupted sessions.

## 3. Read results or every epoch

Set MODE to `complementarity`, `progress`, `results`, or `inspect-submission`.
The command below requires exactly one campaign spec; with multiple runs select
the intended exact spec explicitly instead of guessing the latest directory.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/capacity_*_a6000_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=complementarity
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_capacity_hydra.py" "$MODE" --spec "$SPEC"
BASH
```

`complementarity` reads the authenticated CPU analysis if complete; otherwise
it computes a read-only CPU analysis from saved probabilities. No inference
or GPU is used. The automatic job stores its full per-class/QCD/confusion and
paired-interval data in `analysis/results.json`.

`N` means native128, `W` wide256; `CE` classification only, `U` U100-w5,
`UI` U100-w5+IT-w5. `_s0/_s1` are matched seeds. All12 new rows are deployable
HLT-only models. The report uses same-width/same-seed controls; historical
rows are context. A wide CE gain alone is a capacity gain. A larger UI-minus-U
gap in the wide student suggests capacity helps exploit the IT target.

Fixed ensembles test useful diversity but have multiple-model deployment
cost. Bootstrap intervals are conditional jet-IID sensitivity, not independent
test significance. Original50k overlaps validation, and all comparisons remain
exploratory. A failed size or schedule cannot establish an information ceiling.

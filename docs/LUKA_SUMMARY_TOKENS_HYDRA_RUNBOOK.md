# Queue shared ParT summary tokens on Hydra

This adds **five configurations at two matched seeds: ten fresh HLT fits**,
plus A6000 setup, durable CPU release and CPU report. No new teachers, exports
or data transfers are needed. Keep the completed selective IT teacher, its
TRAIN bank, original U100 inputs/targets, parent artifacts and pinned worktrees.

| Cell | Summaries | Objective | U100 / IT predictor inputs |
| --- | ---: | --- | --- |
| A | 1 | CE | none |
| B | 1 | CE + U100-w5 + IT-w5 | general / general |
| C | 3 | CE | none |
| D | 3 | CE + U100-w5 + IT-w5 | general / general |
| E | 3 | CE + U100-w5 + IT-w5 | U100 summary / IT summary |

C/D/E share the original pooling weights across independent queries and add
3072 deployed parameters. All models keep the original particle encoder.
See the [plan](plans/LUKA_SUMMARY_TOKENS_HYDRA_PLAN.md),
[contract](contracts/LUKA_SUMMARY_TOKENS_HYDRA.md), and
[local validation](LUKA_SUMMARY_TOKENS_HYDRA_VALIDATION.md).

## 1. Commit and push from local PowerShell

Run from the repository. Review this scoped diff; it excludes unrelated runtime
changes currently in the workspace. Do not use `git add .`.

```powershell
$paths = @(
    'src/hlt_classification/luka_summary_tokens_hydra',
    'scripts/luka_summary_tokens_hydra.py',
    'scripts/queue_luka_summary_tokens_hydra.sh',
    'sbatch/run_luka_summary_tokens_hydra.sh',
    'tests/test_luka_summary_tokens_hydra.py',
    'docs/plans/LUKA_SUMMARY_TOKENS_HYDRA_PLAN.md',
    'docs/contracts/LUKA_SUMMARY_TOKENS_HYDRA.md',
    'docs/LUKA_SUMMARY_TOKENS_HYDRA_RUNBOOK.md',
    'docs/LUKA_SUMMARY_TOKENS_HYDRA_VALIDATION.md',
    'README.md', 'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin $paths -and -not $_.StartsWith('src/hlt_classification/luka_summary_tokens_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Add shared ParT summary tokens with matched privileged supervision'
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
read -r -p 'New pushed summary-token commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_summary_tokens_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="${BASE}/runs/summary_tokens_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for outputs (at least 9): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_summary_tokens_hydra.py"
REG_LOG="$(mktemp "${BASE}/runs/summary_tokens_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$CLI" create \
    --root "$ROOT" --selective-spec "$PARENT" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve the root and log.\n' >&2
    exit 1
fi
printf 'Registered spec: %s/campaign_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_summary_tokens_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

Expect `lukast_setup` and `lukast_release`. After both IDs are recorded,
logging out is safe. Successful setup automatically queues all ten independent
fits and the CPU report. No further terminal session or approval is needed.

Queue order is E,B,D,C,A for seed0, then E,B,D,C,A for seed1. Slurm controls
actual start order. There are no fit-to-fit dependencies or array concurrency
caps; the report waits afterany. Fits request one RTX A6000, six CPUs and
32000 MiB, with warmed measured walltimes capped at48h. Setup requests12h.

Setup authenticates installed Weaver and historical native replay, checks
initialization and independent shared pooling, runs actual-target production
miniatures and validates deployment reload for all ten cells. It measures
full/partial and longest-support backward and full validation. Technical
weights are discarded. Poor finite accuracy never prevents science release.

If setup fails, preserve its diagnostics and exact submission journals. Do not
remove claims to retry. If YES was mistyped before live submission, rerun only
the queue helper with the same spec. `inspect-submission` reads accepted IDs.

## 3. Read results or every epoch

Set MODE to `progress`, `results`, or `inspect-submission`.
The command below requires exactly one campaign spec; with multiple runs select
the intended exact spec explicitly instead of guessing the latest directory.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/summary_tokens_*_a6000_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=results
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_summary_tokens_hydra.py" "$MODE" --spec "$SPEC"
BASH
```

`results` authenticates completed fits and prints AUC, recovery, matched CE
and B deltas, all registered contrasts and deployed parameter/latency costs.
`progress` reads immutable epoch logs, including losses, recovery and current
best-epoch contrasts. Neither command performs inference or changes jobs/files.

The primary comparison is E-D: same architecture, losses and initial weights,
but teacher-specific prediction routes. C-A checks the architecture without
privilege; E-B checks the combined change. Missing same-seed controls remain
missing. Two-seed summaries are descriptive. Original50k overlaps validation;
there is no independent-test claim. Reported latency includes transfer and
shared-node conditions; it is not production certification.

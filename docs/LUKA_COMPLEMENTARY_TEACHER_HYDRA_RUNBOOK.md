# Queue complementary OFFLINE teachers on Hydra

Execution v2 repairs the runtime failure in setup160118 from commitd88eb82c.
TE's eight-jet explanation passes were too slow: measured2.143s/update and
projected5058min exceeded the2880min ceiling. The new execution uses64-jet
explanation microbatches, with the same loss, full128 training batch and30
epochs. Commit/push the repair and register the fresh v2 root below. Preserve
the old root, logs and journals; no export or input transfer needs repeating.
The real setup must still qualify its new speed and memory before release.

This queues **three final teachers, four students and three OOF preparation
fits**: ten training jobs, plus setup, durable CPU release and CPU report.
Every student uses CE and total embedding weight5; there is no U100 loss.
Deployment remains native HLT ParT. See the
[plan](plans/LUKA_COMPLEMENTARY_TEACHER_HYDRA_PLAN.md),
[contract](contracts/LUKA_COMPLEMENTARY_TEACHER_HYDRA.md) and
[validation record](LUKA_COMPLEMENTARY_TEACHER_HYDRA_VALIDATION.md).

Reuse the original teacher-sweep spec and its native OFFLINE transfer already
on Brown. No new SPORC export or transfer is needed. Preserve parent worktrees,
input bundles, installed environment and historical reference artifacts.

## Commit and push from local PowerShell

Review these paths. Existing unrelated changes stay separate. The shared
summary-token campaign can be committed separately before or after this one;
the source guard accepts other additive campaigns while preserving donor code.
Do not stage the whole workspace.

```powershell
$paths = @(
    'src/hlt_classification/luka_complementary_teacher_hydra',
    'scripts/luka_complementary_teacher_hydra.py',
    'scripts/queue_luka_complementary_teacher_hydra.sh',
    'sbatch/run_luka_complementary_teacher_hydra.sh',
    'tests/test_luka_complementary_teacher_hydra.py',
    'docs/plans/LUKA_COMPLEMENTARY_TEACHER_HYDRA_PLAN.md',
    'docs/contracts/LUKA_COMPLEMENTARY_TEACHER_HYDRA.md',
    'docs/LUKA_COMPLEMENTARY_TEACHER_HYDRA_RUNBOOK.md',
    'docs/LUKA_COMPLEMENTARY_TEACHER_HYDRA_VALIDATION.md',
    'README.md', 'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin $paths -and -not $_.StartsWith('src/hlt_classification/luka_complementary_teacher_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Add complementary OFFLINE teachers and four matched HLT students'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Register and queue from Brown

Paste the complete block. Supply the new40-character commit, personally
available quota (at least9GiB), and `YES` at the concrete dry review. Registration
authenticates the parent and can be quiet for several minutes; its log is printed.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
PARENT="${BASE}/runs/teacher_sweep_6fe7e349_a6000_r1/campaign_spec.json"
test -f "$PARENT"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$PARENT")"
read -r -p 'New pushed complementary-teacher commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_complementary_teacher_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="${BASE}/runs/complementary_teacher_${COMMIT:0:8}_a6000_v2_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for outputs (at least 9): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_complementary_teacher_hydra.py"
REG_LOG="$(mktemp "${BASE}/runs/complementary_teacher_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$CLI" create \
    --root "$ROOT" --teacher-sweep-spec "$PARENT" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve the root and log.\n' >&2
    exit 1
fi
printf 'Registered spec: %s/campaign_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_complementary_teacher_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

Expect `lukact_setup` and `lukact_release`. Once both numeric IDs are recorded,
logout is safe. Successful setup automatically queues T0, three fold helpers,
TE/TF and the four students with the required dependencies, then an afterany
report. No login dispatcher or second approval is needed. Scheduling order
does not guarantee simultaneous starts.

T0 and F0/F1/F2 are independent. TE waits for T0; TF waits for all fold helpers.
S0 waits for T0; SE waits for T0/TE; SF waits for T0/TF; SEF waits for all three
final teachers. Fold helpers are preparation, not extra student configurations.
Only true input dependencies are introduced.

Setup runs real data/model miniatures and measures every objective path, including
TE second derivatives, on an A6000. Maximum walltime is48h per fit. If setup
fails, preserve its diagnostics and exact journals; do not delete claims or
blindly retry. A mistyped YES before live submission can be retried with the
same queue helper. `inspect-submission` prints accepted IDs without resubmitting.

## Completed results and every epoch

Set MODE to `results`, `progress`, or `inspect-submission`. With multiple
campaigns, explicitly select the desired spec instead of choosing by timestamp.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/complementary_teacher_*_a6000_v2_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=results
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'Campaign: %s\nAuthenticating saved artifacts; this may take a few minutes.\n' "$SPEC"
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_complementary_teacher_hydra.py" "$MODE" --spec "$SPEC"
BASH
```

Results show S0/SE/SF/SEF against the new S0 control, historical recovery and
explicit privileged teacher rows. F0/F1/F2 show preparation status and counts,
not a misleading development-validation comparison. Original50k overlaps
validation; all results are single-seed exploratory evidence.

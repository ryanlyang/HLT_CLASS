# Queue eight U100 embedding-objective fits on Hydra

Eight new student configurations, with the existing matched Hydra C0 U100-w5
control reused. The [plan](plans/LUKA_EMBEDDING_OBJECTIVE_HYDRA_PLAN.md),
[contract](contracts/LUKA_EMBEDDING_OBJECTIVE_HYDRA.md) and
[validation evidence](LUKA_EMBEDDING_OBJECTIVE_HYDRA_VALIDATION.md) specify
coordinate weighting, class directions, per-jet sensitivity, discriminant
prediction, covariance balancing and CE-protected gradients. All predict only
U100 and deploy the unchanged HLT ParT.

No new SPORC transfer is needed. Reuse the existing Group A spec, bundle,
frozen-adapter validation U100 delta, U100 encoder delta, environment and C0
reference. Registration derives their locations from authenticated prior specs.
Keep those artifacts and their original source worktrees. This creates a fresh
campaign and does not alter the running A/B/C/R jobs.

## Commit and push locally

Run in the repository's PowerShell terminal. Review the listed changes before
committing; unrelated local work must stay out of this commit.

```powershell
$paths = @(
    'src/hlt_classification/luka_embedding_objective_hydra',
    'scripts/luka_embedding_objective_hydra.py',
    'scripts/queue_luka_embedding_objective_hydra.sh',
    'sbatch/run_luka_embedding_objective_hydra.sh',
    'tests/test_luka_embedding_objective_hydra.py',
    'docs/plans/LUKA_EMBEDDING_OBJECTIVE_HYDRA_PLAN.md',
    'docs/contracts/LUKA_EMBEDDING_OBJECTIVE_HYDRA.md',
    'docs/LUKA_EMBEDDING_OBJECTIVE_HYDRA_RUNBOOK.md',
    'docs/LUKA_EMBEDDING_OBJECTIVE_HYDRA_VALIDATION.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin ($paths + @('README.md','docs/HANDOFF.md','docs/LEGACY_SOURCE_MAP.md')) -and -not $_.StartsWith('src/hlt_classification/luka_embedding_objective_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
# These shared docs may contain unrelated edits: select only this campaign's hunks.
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Add eight U100 classification-aware embedding objectives for Hydra'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Register and queue on Brown

Paste the complete block, preserving its newlines. Enter the new pushed
40-character commit, available personal quota (at least 5 GiB after existing
inputs/environment), and `YES` at the final prompt. The default Group A path is
the user's known execution; change it only if the actual spec is elsewhere.
Registration authenticates prior source and artifacts, so it can take a few
minutes. Its printed log can be tailed from another terminal.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
GROUP_A_SPEC="${BASE}/runs/next_A_342e5d3a_a6000_v2_r1/campaign_spec.json"
test -f "$GROUP_A_SPEC"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$GROUP_A_SPEC")"
read -r -p 'New pushed 40-character embedding-objective commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_embedding_objective_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$GROUP_A_SPEC")"
ROOT="${BASE}/runs/embedding_objective_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for new outputs (at least 5): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REG_LOG="$(mktemp "${BASE}/runs/embedding_objective_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s \
    "${PROJECT}/scripts/luka_embedding_objective_hydra.py" create \
    --root "$ROOT" --group-a-spec "$GROUP_A_SPEC" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve its log and any output root.\n' >&2
    exit 1
fi
printf 'Registered spec: %s/campaign_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_embedding_objective_hydra.sh" \
    "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

Expect `lukaeo_setup` and `lukaeo_release` IDs. After both IDs are recorded,
logout is safe. Successful setup builds TRAIN-only geometry/calibration and
validation diagnostic targets, verifies native parity and runs every actual-target
miniature. The CPU release then submits R8,R2,R4,R1,R7,R3,R5,R6 with measured
walltimes, followed by an afterany CPU report. Fits are independent and request
one A6000 each. No additional confirmation or login-side polling is needed.
Local checks cannot promise start times, throughput or improved recovery.

If authorization text was mistyped before submission, keep the registered
spec and rerun only the queue helper with that exact spec. If submission reports
partial/ambiguous state, preserve its claims and journals and inspect exact IDs;
do not create a duplicate campaign or remove its evidence. Never rerun `create`
against an existing output directory.

## Epoch-by-epoch and completed results

This finds exactly one campaign. If several exist, replace the `SPECS` line
with the intended full spec path. Set `MODE=results` for authenticated completed
fits, or `MODE=inspect-submission` for exact setup/release/science IDs.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/embedding_objective_*_a6000_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact embedding-objective spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=progress
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'Campaign: %s\n' "$SPEC"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_embedding_objective_hydra.py" \
    "$MODE" --spec "$SPEC"
printf '\nRead-only; no inference, training, file writes or job changes.\n'
BASH
```

Progress shows every published epoch's native AUC/recovery, delta from the
selected C0, delta from its same epoch, ordinary embedding error and error in
the frozen teacher's class-discriminating subspace. Complete epoch JSONs also
contain complement errors, real/predicted teacher-head accuracy and CE, encoder
gradient diagnostics and projection counts. These diagnostics never select
checkpoints. `results` prints a compact selected-checkpoint table for finished
fits, and identifies missing/running jobs. Neither command performs inference.

Recovery is `100*(AUC-M0HLT)/(OFFLINE-M0HLT)` using historical SPORC anchors.
Use `delta C0` for the same-seed Hydra U100-w5 comparison. `original50k` overlaps
validation. These are eight single-seed screening configurations, not replicated
findings. Existing jobs and source worktrees remain untouched.

If setup/release fails, inspect the exact `setup-JOBID.out` or `release-JOBID.out`
under this root, plus `setup/resource_diagnostic.json` and
`submission_bootstrap/submission_ledger.json`. Preserve all claims and journals;
never remove them to force a retry. No scientific accuracy threshold blocks release.

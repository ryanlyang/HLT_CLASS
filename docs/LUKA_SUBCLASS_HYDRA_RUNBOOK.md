# Queue ten U100-w5 plus subclass fits on Hydra

Ten new students at subclass weights **0.03, 0.1, 0.3, 0.5, 1, 2, 5, 10, 40, 100**.
Each uses native HLT CE + U100 embedding weight5, 30 epochs, and a three-epoch
subclass ramp. Reuse the matched C0 control; no new teacher fit.
See the [plan](plans/LUKA_SUBCLASS_HYDRA_PLAN.md),
[contract](contracts/LUKA_SUBCLASS_HYDRA.md) and
[validation evidence](LUKA_SUBCLASS_HYDRA_VALIDATION.md).

No SPORC export or transfer is needed. Keep the existing teacher-sweep spec,
its T00 selected OFFLINE checkpoint and TRAIN embedding bank, C0 reference,
input bundle, environment and original source worktrees. Registration authenticates
these through their original code. Setup rebuilds one subclass bank on the
299234 TRAIN rows; it never uses FullStat3M targets. HLT-only inference remains
unchanged.

## Commit and push locally

Run in the repository's PowerShell terminal. Review the listed changes before
committing; unrelated local work must stay out of this commit.

```powershell
$paths = @(
    'src/hlt_classification/luka_subclass_hydra',
    'scripts/luka_subclass_hydra.py',
    'scripts/queue_luka_subclass_hydra.sh',
    'sbatch/run_luka_subclass_hydra.sh',
    'tests/test_luka_subclass_hydra.py',
    'docs/plans/LUKA_SUBCLASS_HYDRA_PLAN.md',
    'docs/contracts/LUKA_SUBCLASS_HYDRA.md',
    'docs/LUKA_SUBCLASS_HYDRA_RUNBOOK.md',
    'docs/LUKA_SUBCLASS_HYDRA_VALIDATION.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin ($paths + @('README.md','docs/HANDOFF.md','docs/LEGACY_SOURCE_MAP.md')) -and -not $_.StartsWith('src/hlt_classification/luka_subclass_hydra/')
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
git commit -m 'Add ten U100-w5 subclass weights with automatic Hydra release'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Register and queue on Brown

Paste the complete block, preserving its newlines. Enter the new pushed
40-character commit, available personal quota (at least 5 GiB after existing
inputs/environment), and `YES` at the final prompt. The default teacher-sweep path is
the user's known execution; change it only if the actual spec is elsewhere.
Registration authenticates prior source and artifacts, so it can take a few
minutes. Its printed log can be tailed from another terminal.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
TEACHER_SWEEP_SPEC="${BASE}/runs/teacher_sweep_6fe7e349_a6000_r1/campaign_spec.json"
test -f "$TEACHER_SWEEP_SPEC"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$TEACHER_SWEEP_SPEC")"
read -r -p 'New pushed 40-character subclass commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_subclass_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$TEACHER_SWEEP_SPEC")"
ROOT="${BASE}/runs/subclass_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for new outputs (at least 5): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REG_LOG="$(mktemp "${BASE}/runs/subclass_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s \
    "${PROJECT}/scripts/luka_subclass_hydra.py" create \
    --root "$ROOT" --teacher-sweep-spec "$TEACHER_SWEEP_SPEC" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve its log and any output root.\n' >&2
    exit 1
fi
printf 'Registered spec: %s/campaign_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_subclass_hydra.sh" \
    "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

Expect `lukasc_setup` and `lukasc_release` IDs. After both IDs are recorded,
logout is safe. Successful setup authenticates T00, builds the frozen TRAIN
subclass bank, verifies native/U100 parity and runs every actual-target
miniature. The CPU release then submits S0p5,S5,S40,S1,S10,S100,S0p3,S2,S0p1,S0p03
with measured walltimes, followed by an afterany CPU report. Each fit requests
one A6000 and has no dependency on another fit. There are 13 jobs in total:
setup, release, ten fits and report. No additional confirmation or login-side
process is needed after the initial authorization. Setup can still reject
invalid inputs, resource limits or numerical parity; it never rejects a finite
scientific result because accuracy is poor.

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
SPECS=("$HOME"/hlt_u100_pairs/runs/subclass_*_a6000_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact subclass spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=progress
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'Campaign: %s\n' "$SPEC"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_subclass_hydra.py" \
    "$MODE" --spec "$SPEC"
printf '\nRead-only; no inference, training, file writes or job changes.\n'
BASH
```

Progress shows every published epoch's native AUC/recovery, delta from
selected C0 and its same epoch, current subclass coefficient, raw KL, weighted
KL and weighted U100 loss. Epoch JSONs also contain native accuracy/rejection,
TRAIN subclass agreement/entropy and sampled encoder gradient norms/cosines at
epochs1,3,10,20,30. The latter compare CE, weighted U100 and weighted subclass
losses. They do not alter gradients or checkpoint selection. `results` prints
selected-checkpoint tables and missing/running states without inference.

Recovery is `100*(AUC-M0HLT)/(OFFLINE-M0HLT)` using historical SPORC anchors.
Use `delta C0` for the same-seed Hydra U100-w5 comparison. `original50k` overlaps
validation. These are ten single-seed screening configurations, not replicated
findings. Existing jobs and source worktrees remain untouched.

If setup/release fails, inspect the exact `setup-JOBID.out` or `release-JOBID.out`
under this root, plus `setup/resource_diagnostic.json` and
`submission_bootstrap/submission_ledger.json`. Preserve all claims and journals;
never remove them to force a retry. No scientific accuracy threshold blocks release.

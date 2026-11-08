# Queue OFFLINE pretext and structure supervision on Hydra

This campaign has **8 teachers +22 HLT fits**. Every HLT fit retains U100-w5.
One GPU setup job prepares and qualifies the tasks. A Slurm CPU release job
automatically queues all30 fits and the report after setup succeeds. You can
log out once both bootstrap IDs are recorded; no later approval is needed.

See the [plan](plans/LUKA_OFFLINE_PRETEXT_STRUCTURE_HYDRA_PLAN.md),
[contract](contracts/LUKA_PRETEXT_STRUCTURE_HYDRA.md) and
[validation evidence](LUKA_PRETEXT_STRUCTURE_HYDRA_VALIDATION.md).

Keep the original teacher-sweep/Group A source worktrees, bundles, raw TRAIN
export and completed T00/S00 artifacts. The only new source transfer is a
minimal physical validation delta. It contains no final-test data.

## 1. Commit and push locally

Review the scoped changes. Do not stage unrelated runtime changes.

```powershell
$paths = @(
  'src/hlt_classification/luka_pretext_structure_hydra',
  'scripts/luka_pretext_structure_hydra.py',
  'scripts/queue_luka_pretext_structure_hydra.sh',
  'sbatch/run_luka_pretext_structure_hydra.sh',
  'tests/test_luka_pretext_structure_hydra.py',
  'docs/plans/LUKA_OFFLINE_PRETEXT_STRUCTURE_HYDRA_PLAN.md',
  'docs/contracts/LUKA_PRETEXT_STRUCTURE_HYDRA.md',
  'docs/LUKA_PRETEXT_STRUCTURE_HYDRA_RUNBOOK.md',
  'docs/LUKA_PRETEXT_STRUCTURE_HYDRA_VALIDATION.md',
  'README.md', 'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
  $_ -notin $paths -and -not $_.StartsWith('src/hlt_classification/luka_pretext_structure_hydra/')
}
if ($unexpected) { throw ('Review existing staged work: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace' }
git commit -m 'Add OFFLINE pretext teachers and HLT structure campaign with automatic Hydra release'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Export validation physical particles on SPORC

Paste on SPORC. This reads source data and publishes a separate delta; it does
not train or submit jobs. It can run quietly while authenticating old inputs.

```bash
bash <<'BASH'
set -euo pipefail
MAIN=/home/ryreu/atlas/HLT_Classification
read -r -p 'New pushed pretext commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="/home/ryreu/atlas/HLT_Classification_pretext_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
  test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
  test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
  git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR=/home/ryreu/miniconda3/envs/atlas_kd_sporc
OUTPUT="/home/ryreu/atlas/luka_pretext_validation_${COMMIT:0:8}"
test ! -e "$OUTPUT"
test ! -e "${OUTPUT}.log"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup "${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_pretext_structure_hydra.py" export \
  --prepared "${MAIN}/checkpoints/luka_alljets_9e591c7a_r1/prepared" \
  --origin-spec "${MAIN}/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json" \
  --output "$OUTPUT" --expected-commit "$COMMIT" >"${OUTPUT}.log" 2>&1 </dev/null &
printf 'Exporter PID: %s\nLog: %s\n' "$!" "${OUTPUT}.log"
BASH
```

Read the log with `tail -f /home/ryreu/atlas/luka_pretext_validation_COMMIT8.log`,
replacing COMMIT8. Ctrl-C stops the log viewer. Completion prints `directory`,
`manifest_hash`, and `payload_bytes`. Copy that64-character manifest hash.
An existing directory without a completion manifest is an incomplete export;
preserve it and inspect the log rather than overwriting it.

## 3. Transfer, register and queue on Brown

Set `SPORC_HOST` to the SSH alias/hostname you already use from Brown to reach
SPORC. This is the same transfer direction as the previous teacher sweep.
The destination uses the commit as identity and is separate from earlier data.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
PARENT="${BASE}/runs/teacher_sweep_6fe7e349_a6000_r1/campaign_spec.json"
test -f "$PARENT"
read -r -p 'New pushed pretext commit (40 characters): ' COMMIT </dev/tty
read -r -p 'Completed validation manifest hash (64 characters): ' VALIDATION_HASH </dev/tty
read -r -p 'Existing SPORC SSH hostname/alias: ' SPORC_HOST </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$VALIDATION_HASH" =~ ^[0-9a-f]{64}$ ]]
[[ "$SPORC_HOST" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_pretext_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
  test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
  test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
  git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
VALIDATION="${BASE}/pretext_validation_${COMMIT:0:8}"
test ! -e "$VALIDATION"
mkdir "$VALIDATION"
rsync -av --progress "ryreu@${SPORC_HOST}:/home/ryreu/atlas/luka_pretext_validation_${COMMIT:0:8}/" "$VALIDATION/"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="${BASE}/runs/pretext_structure_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB after transfer (at least 34): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_pretext_structure_hydra.py"
REG_LOG="$(mktemp "${BASE}/runs/pretext_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$CLI" create \
  --root "$ROOT" --teacher-sweep-spec "$PARENT" \
  --validation-root "$VALIDATION" --validation-hash "$VALIDATION_HASH" \
  --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
  tail -n 100 -- "$REG_LOG"
  printf 'Registration stopped; preserve the root and log.\n' >&2
  exit 1
fi
bash "${PROJECT}/scripts/queue_luka_pretext_structure_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

Review the printed bounded plan and type only `YES`. Expect `lukaps_setup`
and `lukaps_release`, with the release pending on setup. Both IDs in the live
ledger mean the workflow survives logout. Setup checks real data, installed
Weaver, miniatures and measured resources before releasing science.

Teachers and direct-structure/control fits are independent after setup.
Exx waits on Txx; each M row waits only on its listed teachers. Queue order is
interleaved as registered in the plan; Slurm controls actual start times.
The report waits afterany and lists failed/incomplete rows as well as results.

If setup fails, preserve its root/log and inspect the diagnostics. There is no
automatic retry, truncation or unreviewed increase in the48h fit ceiling. A
mistyped `YES` before submission can be corrected by rerunning only the queue
helper with the same spec. For an interrupted submission, inspect the ledger
before retrying; never delete claims or submission journals.

## 4. Read results and epoch progress

Use `results`, `progress`, `teacher-results` or `inspect-submission` for MODE.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/pretext_structure_*_a6000_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Choose one exact spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=results
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_pretext_structure_hydra.py" "$MODE" --spec "$SPEC"
BASH
```

These commands read saved artifacts and accounting; they do not run inference
or alter jobs/files. Teachers report pretext loss, not recovery. Student results
include recovery versus historical M0HLT/OFFLINE and deltas to fresh C_U/C_CE.
Missing controls stay missing. All results remain exploratory single-seed
validation; original50k is an overlapping secondary view.

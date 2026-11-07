# Queue the twelve-teacher Hydra sweep

This adds **12 OFFLINE teachers + 12 HLT students**, plus GPU setup, durable CPU
release and CPU report. Every student retains CE + U100 embedding weight5 and
adds its selected new teacher's embedding at weight5. Existing campaigns and
the SPORC SupCon5 pair remain separate. See the
[plan](plans/LUKA_TEACHER_SWEEP_HYDRA_PLAN.md),
[contract](contracts/LUKA_TEACHER_SWEEP_HYDRA.md), and
[validation evidence](LUKA_TEACHER_SWEEP_HYDRA_VALIDATION.md).

The original Hydra bundle/Group A spec/environment/C0 are reused. One new delta
is required: native OFFLINE TRAIN and validation input caches. The older Group A
raw transfer has no OFFLINE validation. Do not transfer a whole dataset or
delete original artifacts/worktrees.

## 1. Commit and push in Windows PowerShell

Run from the repository. This stages only the new campaign and its documentation.
The source guard permits only the new runtime since donor
`0ae4f28fbc25df4aef0ddc617a16555f98b02591`.

```powershell
$paths = @(
    'src/hlt_classification/luka_teacher_sweep_hydra',
    'scripts/luka_teacher_sweep_hydra.py',
    'scripts/queue_luka_teacher_sweep_hydra.sh',
    'sbatch/run_luka_teacher_sweep_hydra.sh',
    'tests/test_luka_teacher_sweep_hydra.py',
    'docs/plans/LUKA_TEACHER_SWEEP_HYDRA_PLAN.md',
    'docs/contracts/LUKA_TEACHER_SWEEP_HYDRA.md',
    'docs/LUKA_TEACHER_SWEEP_HYDRA_RUNBOOK.md',
    'docs/LUKA_TEACHER_SWEEP_HYDRA_VALIDATION.md',
    'README.md', 'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin $paths -and -not $_.StartsWith('src/hlt_classification/luka_teacher_sweep_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Add twelve OFFLINE teacher recipes and paired Hydra students'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Export only the new inputs on SPORC

Paste the entire block with its newlines. This is CPU-only preparation, with
one numerical-library thread and a bounded native cache. It uses a detached
process so closing the terminal does not interrupt the export. Supply the new
pushed 40-character commit. Existing incomplete exports must be preserved;
choose a fresh suffix if recovering an interrupted attempt.

```bash
bash <<'BASH'
set -euo pipefail
MAIN=/home/ryreu/atlas/HLT_Classification
read -r -p 'New pushed teacher-sweep commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="/home/ryreu/atlas/HLT_Classification_teacher_sweep_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR=/home/ryreu/miniconda3/envs/atlas_kd_sporc
ORIGIN="${MAIN}/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json"
OUTPUT="/home/ryreu/atlas/luka_teacher_offline_${COMMIT:0:8}"
test ! -e "$OUTPUT"
test ! -e "${OUTPUT}.log"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup "${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_teacher_sweep_hydra.py" export \
    --origin-spec "$ORIGIN" --output "$OUTPUT" --expected-commit "$COMMIT" \
    >"${OUTPUT}.log" 2>&1 </dev/null &
printf 'Exporter PID: %s\nLog: %s\n' "$!" "${OUTPUT}.log"
BASH
```

Use `tail -f` on the printed log. Ctrl-C exits the log viewer, not the detached
exporter. Completion prints `directory`, `manifest_hash` and `payload_bytes`;
only a completed `manifest.json` permits transfer/registration. Copy the hash
and exact export directory. If the log stops, inspect the printed PID and log
before starting another export. An existing directory alone is not completion.

## 3. Transfer, register and queue from Brown

Paste this on the Brown login terminal. The SPORC transfer may ask for your
password and Duo. Enter the exact export path and manifest hash from step2.
Registration can take several minutes while it authenticates prior sources;
the printed log can be tailed in a second terminal. Enter at least **9 GiB of
personally available output quota after** the new transfer and environment.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
GROUP_A_SPEC="${BASE}/runs/next_A_342e5d3a_a6000_v2_r1/campaign_spec.json"
test -f "$GROUP_A_SPEC"
read -r -p 'New pushed teacher-sweep commit (40 characters): ' COMMIT </dev/tty
read -r -p 'Completed native OFFLINE export directory on SPORC: ' REMOTE </dev/tty
read -r -p 'Native OFFLINE manifest hash (64 characters): ' OFFLINE_HASH </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$OFFLINE_HASH" =~ ^[0-9a-f]{64}$ ]]
[[ "$REMOTE" =~ ^/home/ryreu/atlas/luka_teacher_offline_[a-zA-Z0-9_]+$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_teacher_sweep_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$GROUP_A_SPEC")"
BUNDLE="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle_root"])' "$GROUP_A_SPEC")"
OFFLINE="${BASE}/offline_teacher_${COMMIT:0:8}"
mkdir -p "$OFFLINE"
rsync -av --partial --info=progress2 "ryreu@sporcsubmit.rc.rit.edu:${REMOTE}/" "${OFFLINE}/"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_teacher_sweep_hydra.py"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" verify-transfer \
    --offline-root "$OFFLINE" --offline-hash "$OFFLINE_HASH" --bundle-root "$BUNDLE"
ROOT="${BASE}/runs/teacher_sweep_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for outputs AFTER transfer (at least 9): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
REG_LOG="$(mktemp "${BASE}/runs/teacher_sweep_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$CLI" create \
    --root "$ROOT" --group-a-spec "$GROUP_A_SPEC" \
    --offline-root "$OFFLINE" --offline-hash "$OFFLINE_HASH" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve the log and output root.\n' >&2
    exit 1
fi
printf 'Registered spec: %s/campaign_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_teacher_sweep_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

The helper displays the exact review and asks for `YES`. This single approval
covers setup, durable release, all measured teacher/student fits and report.
Expect **`lukats_setup` and `lukats_release`** IDs. Once both are recorded you
can log off. Successful setup makes the release submit 24 fits automatically;
no later manual command or confirmation is needed. Each student waits only for
its own teacher. A weak finite teacher score does not block that student.

The initial review tests the largest authorized resource shapes with Slurm.
The release tests measured shapes again. Availability/start times are determined
by Hydra; the implementation does not claim currently free A6000s.

If `YES` was mistyped **before any submission**, rerun only the queue helper
with the same registered spec. For failed setup, missing release or partial
submission, preserve the root, claims and journals and inspect exact IDs/logs.
Do not blindly recreate a campaign or erase claims. No automatic retries.

## 4. Progress and completed results

Use `MODE=progress` for epochs/recoveries/weighted losses, `MODE=results` for
completed checkpoint scores, or `MODE=inspect-submission` for exact recorded
IDs. If there are several runs, select the intended exact spec explicitly.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/teacher_sweep_*_a6000_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=progress
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_teacher_sweep_hydra.py" "$MODE" --spec "$SPEC"
BASH
```

`Txx` are privileged OFFLINE oracles; `Sxx` are deployable HLT students. Compare
students with `C0_U100_HYDRA`, and with `S00` to isolate the benefit of a
different teacher objective from simply adding another teacher. Recovery uses
historical SPORC M0HLT/OFFLINE anchors and may exceed100% for oracle teachers.
The original50k slice overlaps validation; this is exploratory single-seed
screening, with no independent-test claim.

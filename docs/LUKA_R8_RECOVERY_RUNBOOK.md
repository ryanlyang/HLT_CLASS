# Recover C3/R14 without repeating the R8 setup

Original campaign: `next_R8_58d0c0ed_a6000_r1`. Jobs168726/168727 failed
before training and were purged from the controller. Successful setup168256
is reused. This queues only two replacements and a CPU snapshot report,
with a separate root and immutable replacement ledger. The original campaign
and remaining jobs stay intact. See [contract](contracts/LUKA_R8_RECOVERY.md).

## Local commit/push

Stage only these new recovery files; existing unrelated edits remain unstaged.

```powershell
Set-Location C:\Users\22rya\ComputerScience\CERN\HLT_Classification
if (@(git diff --cached --name-only).Count -ne 0) { throw 'Inspect already staged changes first' }
$paths = @(
  'src/hlt_classification/luka_r8_recovery.py',
  'scripts/luka_r8_recovery.py', 'scripts/queue_luka_r8_recovery.sh',
  'sbatch/run_luka_r8_recovery.sh', 'tests/test_luka_r8_recovery.py',
  'docs/contracts/LUKA_R8_RECOVERY.md', 'docs/LUKA_R8_RECOVERY_RUNBOOK.md',
  'docs/LUKA_R8_RECOVERY_VALIDATION.md',
  'docs/plans/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP_PLAN.md'
)
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git diff --cached --stat
git commit -m 'Recover two pre-training R8 Slurm failures with pinned donor runtime'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Brown recovery registration and submission

Paste the printed commit when prompted. Registration authenticates the existing
setup and files; it can take several minutes. No new export or setup GPU job
is needed. Type YES once at the exact-plan confirmation, then logout is safe
after all three job IDs are recorded.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
OLD="${BASE}/runs/next_R8_58d0c0ed_a6000_r1/campaign_spec.json"
MAIN="${BASE}/code_next_R8_58d0c0ed"
read -r -p 'New pushed 40-character recovery commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_R8_recovery_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$OLD")"
ROOT="${BASE}/runs/R8_recovery_${COMMIT:0:8}_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for recovery (at least 3): ' QUOTA </dev/tty
[[ "$QUOTA" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
LOG="$(mktemp "${BASE}/runs/R8_recovery_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Authenticating existing setup; registration log: %s\n' "$LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_r8_recovery.py" create \
    --root "$ROOT" --original-spec "$OLD" --expected-commit "$COMMIT" \
    --quota-gib "$QUOTA" >"$LOG" 2>&1; then
    tail -n 100 -- "$LOG"
    exit 1
fi
SPEC="${ROOT}/recovery_spec.json"
printf 'Recovery spec: %s\n' "$SPEC"
bash "${PROJECT}/scripts/queue_luka_r8_recovery.sh" "$SPEC" </dev/tty
squeue --me -o '%.18i %.30j %.12T %.12l %R'
BASH
```

## Combined results and recovered epoch progress

Use the printed recovery spec. The original campaign's command still reports
its immutable failed attempts; this recovery command merges the replacements.
The automatic report runs after the two replacements and may precede other
original fits. Run `results` again later for a current combined snapshot.

```bash
SPEC="/absolute/path/to/R8_recovery_COMMIT_r1/recovery_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset PYTHONPATH
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_r8_recovery.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_r8_recovery.py" progress --spec "$SPEC"
```

Preserve any partial submission journals and new claims. Do not remove claims,
edit old ledgers, or manually sbatch replacement fits outside this reviewed plan.

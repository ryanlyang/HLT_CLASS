# Queue the two-GTX1080Ti subclass campaign

This execution trains **eleven models**: a fresh FP32 U100-w5 C0 plus the ten
subclass weights0.03,0.1,0.3,0.5,1,2,5,10,40,100. Each GPU job uses two GTX1080Ti
cards on one node, global batch128, FP32/SyncBatchNorm and30 epochs. Deployed
models remain ordinary HLT-only ParT. There is no new data export or teacher.

See the [plan](plans/LUKA_SUBCLASS_DDP_HYDRA_PLAN.md),
[contract](contracts/LUKA_SUBCLASS_DDP_HYDRA.md) and
[validation record](LUKA_SUBCLASS_DDP_HYDRA_VALIDATION.md). Keep the existing
teacher-sweep spec, T00 checkpoint/bank, original U100 bundle, environment and
source worktrees. Keep the completed benchmark and rejected A6000 registration
as evidence; do not modify their specs.

## Commit and push on Windows

These paths isolate this implementation from other local work. Review the
staged diff. The new source guard is additive relative to benchmark commit
`1d83d69d4339599d0cf8c59a8b5cafe52704cd81`; donor runtime files must remain
unchanged. Shared documentation may contain other edits, so stage its relevant
hunks separately.

```powershell
$paths = @(
    'src/hlt_classification/luka_subclass_ddp_hydra',
    'scripts/luka_subclass_ddp_hydra.py',
    'scripts/queue_luka_subclass_ddp_hydra.sh',
    'sbatch/run_luka_subclass_ddp_hydra.sh',
    'tests/test_luka_subclass_ddp_hydra.py',
    'docs/plans/LUKA_SUBCLASS_DDP_HYDRA_PLAN.md',
    'docs/contracts/LUKA_SUBCLASS_DDP_HYDRA.md',
    'docs/LUKA_SUBCLASS_DDP_HYDRA_RUNBOOK.md',
    'docs/LUKA_SUBCLASS_DDP_HYDRA_VALIDATION.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin ($paths + @('README.md','docs/HANDOFF.md','docs/LEGACY_SOURCE_MAP.md')) -and
    -not $_.StartsWith('src/hlt_classification/luka_subclass_ddp_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Run subclass sweep and matched U100 control on two GTX1080Ti GPUs'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Register and queue on Brown

Enter the new pushed40-character commit, personal available quota of at least
6GiB, and `YES` at the final exact-plan review. Registration can take several
minutes while authenticating prior artifacts. Another terminal can tail the
printed log. The old benchmark's3.2–3.3h projections exclude validation and
startup; production limits come from this new setup, not that estimate.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
PARENT="${BASE}/runs/teacher_sweep_6fe7e349_a6000_r1/campaign_spec.json"
test -f "$PARENT"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$PARENT")"
read -r -p 'New pushed 40-character two-GPU subclass commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_subclass_ddp_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="${BASE}/runs/subclass_ddp_${COMMIT:0:8}_1080ti_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for new outputs (at least 6): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REG_LOG="$(mktemp "${BASE}/runs/subclass_ddp_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s \
    "${PROJECT}/scripts/luka_subclass_ddp_hydra.py" create \
    --root "$ROOT" --teacher-sweep-spec "$PARENT" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" \
    >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve its log and output root.\n' >&2
    exit 1
fi
printf 'Registered spec: %s/campaign_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_subclass_ddp_hydra.sh" \
    "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.30j %.12T %.12l %R"
BASH
```

Expect **lukascd_setup** and **lukascd_release** IDs. After both are recorded,
logout is safe. Setup requests two GTX1080Ti, six CPUs,48000MiB and4h. It
prepares targets in a separate process, then qualifies the full production
worker with all eleven real-data miniatures, numerical checks, longest-support
batches and full validation timing. Its success releases C0 first, followed
by S0p5,S5,S40,S1,S10,S100,S0p3,S2,S0p1,S0p03. Fits are independent; their
submission order encourages early control results but cannot force scheduler
start order. A CPU report waits afterany. Total: **14 jobs**.

No extra confirmation or login-side process is needed after initial review.
If setup fails, science is not submitted. If submission is partial/ambiguous,
preserve claims and journals and inspect exact IDs before retrying. If only
the `YES` text was mistyped before submission, rerun the queue helper with the
same spec. Never rerun `create` against an existing output root.

## Completed results and epoch-by-epoch progress

This requires exactly one matching execution; if several exist, replace
`SPECS` with the intended explicit spec path. These commands authenticate saved
artifacts and do not run inference or change jobs. Change `MODE` to
`inspect-submission` to print accepted job IDs.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/subclass_ddp_*_1080ti_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for MODE in results progress; do
    "${ENV_DIR}/bin/python" -u -B -s \
        "${PROJECT}/scripts/luka_subclass_ddp_hydra.py" "$MODE" --spec "$SPEC"
done
BASH
```

`delta fresh C0` compares with this campaign's newly trained FP32 control.
The older `C0_BF16_HISTORICAL` remains visible as context. Progress distinguishes
best-control-so-far from same-epoch control. Recovery uses the historical
M0HLT/OFFLINE anchors; original50k overlaps validation, and this remains a
single-seed exploratory study.

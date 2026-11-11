# Queue the embedding bridge, response and regional campaign

This campaign has **30 scientific fits, including a fresh U100-w5 control**,
plus setup, automatic release and report:33 jobs total. Each GPU job requests
two GTX1080Ti cards on one node,6CPU,48000MiB and FP32/SyncBatchNorm.
The [plan](plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md) fixes
the scientific comparisons; see the [contract](contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md)
and [validation record](LUKA_EMBEDDING_BRIDGE_HYDRA_VALIDATION.md).

Keep the original Hydra bundle, teacher-sweep spec and its authenticated inputs,
environment, original assignments and source worktrees. This needs a new
bounded physical export with **both TRAIN and validation** and the original
U100 weights. The earlier Group A raw export is insufficient.

## 1. Commit and push from Windows

Review the scoped diff. The source guard preserves donor runtime at
45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08. Stage relevant shared-document hunks
separately so unrelated work is preserved.

```powershell
$paths = @(
    'src/hlt_classification/luka_embedding_bridge_hydra',
    'scripts/luka_embedding_bridge_hydra.py',
    'scripts/queue_luka_embedding_bridge_hydra.sh',
    'sbatch/run_luka_embedding_bridge_hydra.sh',
    'tests/test_luka_embedding_bridge_hydra.py',
    'docs/plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md',
    'docs/contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md',
    'docs/LUKA_EMBEDDING_BRIDGE_HYDRA_RUNBOOK.md',
    'docs/LUKA_EMBEDDING_BRIDGE_HYDRA_VALIDATION.md'
)
$shared = @('README.md','docs/HANDOFF.md','docs/LEGACY_SOURCE_MAP.md','docs/plans/README.md')
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin ($paths + $shared) -and
    -not $_.StartsWith('src/hlt_classification/luka_embedding_bridge_hydra/')
}
if ($unexpected) { throw ('Review existing staged work: ' + ($unexpected -join ', ')) }
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git add -p -- $shared
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Add embedding bridges, shared responses and regional targets on Hydra'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Export on SPORC

This CPU preparation reads authenticated selected rows, original assignments
and the original U100 checkpoint. It does not train or access final-test rows.
The source files must still exist; missing/corrupt parents stop the export.
Use a fresh suffix if an earlier export was interrupted; preserve partial roots.

```bash
bash <<'BASH'
set -euo pipefail
MAIN=/home/ryreu/atlas/HLT_Classification
read -r -p 'New pushed 40-character campaign commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="/home/ryreu/atlas/HLT_Classification_embedding_bridge_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR=/home/ryreu/miniconda3/envs/atlas_kd_sporc
PREPARED="${MAIN}/checkpoints/luka_alljets_9e591c7a_r1/prepared"
ORIGIN="${MAIN}/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json"
OUTPUT="/home/ryreu/atlas/luka_embedding_bridge_raw_${COMMIT:0:8}"
test ! -e "$OUTPUT"
test ! -e "${OUTPUT}.log"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup "${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_embedding_bridge_hydra.py" export \
    --prepared "$PREPARED" --origin-spec "$ORIGIN" --output "$OUTPUT" --expected-commit "$COMMIT" \
    >"${OUTPUT}.log" 2>&1 </dev/null &
printf 'Exporter PID: %s\nLog: %s\n' "$!" "${OUTPUT}.log"
BASH
```

Follow the printed log with `tail -f`. Ctrl-C exits that viewer, not the
detached export. Completion prints `directory`, `manifest_hash`, `payload_bytes`;
copy the first two. A directory without its completion manifest is not reusable.

## 3. Transfer, inventory, register and queue on Brown

The verifier prints an input-derived output quota bound, including physical
views, eight native caches, all targets, checkpoints and temporary headroom.
Enter personally available output GiB **after** transfer. Registration rejects
an insufficient reservation before creating the campaign root. The byte count
is intentionally more complete than the approximately6GB compact target banks.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
PARENT="${BASE}/runs/teacher_sweep_6fe7e349_a6000_r1/campaign_spec.json"
test -f "$PARENT"
SELECTIVE="${BASE}/runs/selective_teacher_3fb9e7f1_a6000_r1/campaign_spec.json"
HISTORICAL=()
if [[ -f "$SELECTIVE" ]]; then HISTORICAL=(--selective-spec "$SELECTIVE"); fi
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$PARENT")"
read -r -p 'New pushed 40-character campaign commit: ' COMMIT </dev/tty
read -r -p 'Completed physical export directory on SPORC: ' REMOTE </dev/tty
read -r -p 'Physical export manifest hash (64 characters): ' RAW_HASH </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RAW_HASH" =~ ^[0-9a-f]{64}$ ]]
[[ "$REMOTE" =~ ^/home/ryreu/atlas/luka_embedding_bridge_raw_[a-zA-Z0-9_]+$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_embedding_bridge_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
BUNDLE="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle_root"])' "$PARENT")"
RAW="${BASE}/embedding_bridge_raw_${COMMIT:0:8}"
test ! -e "$RAW"
mkdir "$RAW"
rsync -av --partial --info=progress2 "ryreu@sporcsubmit.rc.rit.edu:${REMOTE}/" "${RAW}/"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_embedding_bridge_hydra.py"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" verify-transfer \
    --bundle-root "$BUNDLE" --transfer-root "$RAW" --transfer-hash "$RAW_HASH"
ROOT="${BASE}/runs/embedding_bridge_${COMMIT:0:8}_1080ti_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available output GiB (at least the printed minimum_quota_gib): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
REG_LOG="$(mktemp "${BASE}/runs/embedding_bridge_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$CLI" create \
    --root "$ROOT" --teacher-sweep-spec "$PARENT" --transfer-root "$RAW" --transfer-hash "$RAW_HASH" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" "${HISTORICAL[@]}" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve the root and log.\n' >&2
    exit 1
fi
bash "${PROJECT}/scripts/queue_luka_embedding_bridge_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

If transfer is interrupted, resume only that exact rsync destination/source and
rerun verification; do not restart the whole block or overwrite another export.
After successful registration, repeat only the queue helper if `YES` was mistyped
before submission. Registration and checks can be quiet while hashing; tail
the printed registration log in another terminal.

The block includes S2IT/S6 as labelled historical comparisons if that exact
selective-campaign spec is present. Registration authenticates its original
worktree and completed outputs; it never copies numbers from chat. If your
completed selective campaign has a different root, set `SELECTIVE` to its exact
spec. Without it, these optional reference rows are omitted. They are not
training parents, and fresh C0 remains the primary comparison.

The helper dry-reviews the whole bounded graph and asks for `YES` once. Expect
`lukaeb_setup` and `lukaeb_release` numeric IDs. After both are recorded, logout
is safe. Setup success automatically releases the scientific graph; no login
polling process or later confirmation is needed. A weak finite teacher releases
its children; a technical failure blocks its own dependency branch.

Setup also checks a disposable CPU measurement of release authentication and
reporting. Its `cpu_resources.json` distinguishes measured kernels from the
full-campaign extrapolation. GPU resource acceptance is insufficient unless
this check and the final setup receipt also complete.

The root order is C0, LS_D050, TR, TP, LM_D066, LL_D075. Descendants are
interleaved, with R5/P5 and the short endpoint early. Only producer dependencies
are present; there are no family-wide holds, serial lanes or concurrency caps.
Slurm decides actual start times. Hardware availability is not assumed.

## 4. Read results and progress

Choose `summary` for one selected-so-far line per task; `progress` for every
published epoch with recovery and same-epoch/selected-so-far C0 contrasts;
`results` for authenticated completed checkpoint metrics, including original50k;
`inspect-submission` for exact job IDs. These commands perform no inference.
Completed results also show measured worker GPU-hours for each fit and its
scientific producer chain, with shared setup cost separately reported.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/embedding_bridge_*_1080ti_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=summary
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_embedding_bridge_hydra.py" "$MODE" --spec "$SPEC"
BASH
```

Use fresh C0 as the primary comparison. Historical BF16 C0 is labelled and
never substituted. Assistants/TR/TP are separate diagnostic rows, not HLT
classifiers. Their selection criteria differ, so “selected” does not always
mean maximum primary AUC. Recovery uses historical anchors without clipping.
This is single-seed development screening; FullStat confirmation comes later.

## Recovery after a technical failure

Inspect the exact setup/job log, resource JSON and submission journal. A failed
setup never releases science. Partial/ambiguous submission preserves claims
and refuses automatic retry; reconcile accepted IDs before any further submit.
Do not delete claims, mutate a campaign spec, reuse a partial root as complete,
or cancel by broad job-name prefix. Queue readiness means readiness for this
real qualification, not a claim that it has already passed on Brown.

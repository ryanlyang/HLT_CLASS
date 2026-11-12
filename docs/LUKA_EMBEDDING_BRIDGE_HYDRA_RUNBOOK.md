# Queue the embedding bridge, response and regional campaign

This campaign has **30 scientific fits, including a fresh U100-w5 control**,
plus setup, automatic release and report:33 jobs total. Each GPU job requests
two GTX1080Ti cards on one node,6CPU,48000MiB and FP32/SyncBatchNorm.
The [plan](plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md) fixes
the scientific comparisons; see the [contract](contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md)
and [validation record](LUKA_EMBEDDING_BRIDGE_HYDRA_VALIDATION.md).

Keep the original Hydra bundle, teacher-sweep spec and its authenticated inputs,
environment, original assignments and source worktrees. Execution v2 reuses the
completed physical export already on Brown and adds a minimal native delta.
It copies exact historical tensors instead of re-encoding them on Brown.
Do not rerun the failed v1 source or reuse its partial preparation as complete.

**For failed setup162131, use the completed-preparation recovery below.**
Its92397736 run finished preparation; preserve that root. Execution v4 checks
the frozen teacher in original consecutive128-row contexts under the explicit
historical/local numerical policy and reuses authenticated arrays. It requires no new SPORC
export or particle reconstruction. Remaining GPU qualification can still fail;
the completed arrays are preserved for diagnosis and subsequent reuse.

## 1. Commit and push from Windows

Review the scoped diff. The source guard preserves donor runtime at
2fb09f38a3a1da85cdceff9df311963fdb3aa872 for this repair (original scientific
donor45075fe6c2d8ccdcf7463cd3a99cbe2fe6976a08). Stage relevant shared-document hunks
separately so unrelated work is preserved.

```powershell
$paths = @(
    'src/hlt_classification/luka_embedding_bridge_hydra',
    'scripts/luka_embedding_bridge_hydra.py',
    'scripts/queue_luka_embedding_bridge_hydra.sh',
    'sbatch/run_luka_embedding_bridge_hydra.sh',
    'tests/test_luka_embedding_bridge_hydra.py',
    'tests/test_luka_embedding_bridge_native.py',
    'tests/test_luka_embedding_bridge_recovery.py',
    'tests/test_luka_embedding_bridge_numerical.py',
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
git commit -m 'Qualify U100 historical replay with bounded drift and CPU GPU parity'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Export only the missing historical tensors on SPORC

Skip sections2 and3 for recovery of162131; both transfers already exist on Brown.

For the failed161306 execution, the physical transfer already on Brown is
`embedding_bridge_raw_0a00441f`, manifest
`e42ced64292f9c6ce4f8831bb43aafa9f56aad60077258451c3ff6734b3120de`.
Keep it. The following export supplies only TRAIN D033/D066 and validation
D033/D066/U100, approximately3GiB. Existing native OFFLINE tensors and the
original HLT/U100 bundle remain in place on Brown.

This works with retained preparation metadata, assignments and raw data even
if the source feature caches were deleted during cleanup. Missing caches are
rebuilt one source file at a time and must reproduce their original NPZ
hashes and full-view fingerprints. Existing corrupt files fail. The exporter
never writes into the historical preparation, `atlas/Fresh_check` or
`atlas/datasets`. No final-test rows, training or inference are involved.

Use the **new repair commit**, contained in fetched `origin/main`; the old
0a00441f worktree does not contain this exporter. Do not substitute a newer
unrelated commit. Completion requires the printed final manifest hash.

```bash
bash <<'BASH'
set -euo pipefail
MAIN=/home/ryreu/atlas/HLT_Classification
read -r -p 'New pushed 40-character repair commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
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
OUTPUT="/home/ryreu/atlas/luka_embedding_bridge_native_${COMMIT:0:8}"
test ! -e "$OUTPUT"
test ! -e "${OUTPUT}.log"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nohup "${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_embedding_bridge_hydra.py" export-native \
    --prepared "$PREPARED" --origin-spec "$ORIGIN" --output "$OUTPUT" --expected-commit "$COMMIT" \
    >"${OUTPUT}.log" 2>&1 </dev/null &
printf 'Exporter PID: %s\nLog: %s\n' "$!" "${OUTPUT}.log"
BASH
```

Follow the printed log with `tail -f`. Ctrl-C exits that viewer, not the
export. Completion prints `directory`, `manifest_hash`, `payload_bytes`;
copy the first two. A directory without its completion manifest is incomplete.
The old physical export on SPORC is not needed for this repair.

## 3. Copy the native delta, register a fresh v4 execution and queue on Brown

This block reuses the exact physical transfer from the failed run. It does
not copy that4.3GiB payload again. The original bundle and native OFFLINE
teacher-sweep delta must remain available. Registration verifies all three
historical input sources before any job submission.

Reserve at least the printed output minimum (currently33GiB) **after** the
native delta is copied. The user retired the failed setup's22 unpublished
arrays (12.87GiB logical size), preserving the spec, logs and hash receipt;
the latest Brown reading is39GiB free and the failed root is328KiB. No more
cleanup is needed for this retry at that reading. Recheck after the native
copy; enter33 only if at least33GiB remain personally available for outputs.
This block stops if the reservation cannot be satisfied.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
FAILED="${BASE}/runs/embedding_bridge_0a00441f_1080ti_r1/campaign_spec.json"
test -f "$FAILED"
PARENT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["parent_spec"]["path"])' "$FAILED")"
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$PARENT")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
BUNDLE="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["bundle_root"])' "$PARENT")"
RAW="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["transfer_root"])' "$FAILED")"
RAW_HASH=e42ced64292f9c6ce4f8831bb43aafa9f56aad60077258451c3ff6734b3120de
SELECTIVE="${BASE}/runs/selective_teacher_3fb9e7f1_a6000_r1/campaign_spec.json"
HISTORICAL=()
if [[ -f "$SELECTIVE" ]]; then HISTORICAL=(--selective-spec "$SELECTIVE"); fi
read -r -p 'New pushed 40-character repair commit: ' COMMIT </dev/tty
read -r -p 'Completed native export directory on SPORC: ' REMOTE </dev/tty
read -r -p 'Native export manifest hash (64 characters): ' NATIVE_HASH </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$NATIVE_HASH" =~ ^[0-9a-f]{64}$ ]]
[[ "$REMOTE" =~ ^/home/ryreu/atlas/luka_embedding_bridge_native_[a-zA-Z0-9_]+$ ]]
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="${BASE}/code_embedding_bridge_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
NATIVE="${BASE}/embedding_bridge_native_${COMMIT:0:8}"
test ! -e "$NATIVE"
mkdir "$NATIVE"
rsync -av --partial --info=progress2 "ryreu@sporcsubmit.rc.rit.edu:${REMOTE}/" "${NATIVE}/"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_embedding_bridge_hydra.py"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" verify-native \
    --bundle-root "$BUNDLE" --native-root "$NATIVE" --native-hash "$NATIVE_HASH"
"${ENV_DIR}/bin/python" -u -B -s "$CLI" verify-transfer \
    --bundle-root "$BUNDLE" --transfer-root "$RAW" --transfer-hash "$RAW_HASH"
ROOT="${BASE}/runs/embedding_bridge_${COMMIT:0:8}_1080ti_v4_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available output GiB (at least printed minimum_quota_gib): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
REG_LOG="$(mktemp "${BASE}/runs/embedding_bridge_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$CLI" create \
    --root "$ROOT" --teacher-sweep-spec "$PARENT" --transfer-root "$RAW" --transfer-hash "$RAW_HASH" \
    --native-root "$NATIVE" --native-hash "$NATIVE_HASH" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" "${HISTORICAL[@]}" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve the root and log.\n' >&2
    exit 1
fi
bash "${PROJECT}/scripts/queue_luka_embedding_bridge_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

If native transfer is interrupted, resume only that exact rsync
source/destination and rerun verification; do not overwrite another export.
Once copied, resume from environment initialization and verification rather
than repeating `test ! -e "$NATIVE"`. After registration, repeat only the
queue helper if `YES` was mistyped before submission. All checks can take
minutes while hashing; follow the printed registration log in another terminal.

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
SPECS=("$HOME"/hlt_u100_pairs/runs/embedding_bridge_*_1080ti_v4_r1/campaign_spec.json)
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

The4fb32e79 recovery attempt failed during registration because it used an
active-module source helper to authenticate a different checkout. It did not
publish a new campaign spec, submit jobs or alter completed preparation.
The subsequent e0f9d874 setup163911 successfully reused the arrays but failed
its strict numerical replay. Diagnostic164226 now supports the explicitly
versioned v4 numerical policy. Commit/push this latest repair before running
the block below; supply that new commit, not4fb32e79 or e0f9d874. No new export
or separate numerical-probe job is needed.

For162131, first commit/push the scoped repair as above, then run this on
Brown. Paste that new40-character commit. Registration checks the old terminal
jobs and all completed preparation hashes; allow several minutes for NFS reads.
The new setup links49 array payloads (32 view arrays,12 physical arrays and
5 U100 arrays) and reruns GPU qualification. The storage reservation credits
those links, so it does not demand a second full33GiB allocation. A fresh
root and source snapshot distinguish this execution from the failed run.

```bash
bash <<'BASH'
set -euo pipefail
trap 'printf "Stopped; preserve the old and new roots and paste the error.\n" >&2' ERR
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
OLD="${BASE}/runs/embedding_bridge_92397736_1080ti_v2_r1/campaign_spec.json"
test -f "$OLD"
read -r -p 'New pushed repair commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
MAIN="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$OLD")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$OLD")"
git -C "$MAIN" fetch origin main
git -C "$MAIN" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="${BASE}/code_embedding_bridge_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="${BASE}/runs/embedding_bridge_${COMMIT:0:8}_1080ti_v4_r1"
test ! -e "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REG_LOG="$(mktemp "${BASE}/runs/embedding_bridge_${COMMIT:0:8}_recovery.XXXXXX.log")"
printf 'Recovery registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s - "$OLD" "$PROJECT" "$ROOT" "$COMMIT" >"$REG_LOG" 2>&1 <<'PY'
import json, subprocess, sys
old_path, project, root, commit = sys.argv[1:]
with open(old_path) as handle:
    old = json.load(handle)
args = [sys.executable, '-u', '-B', '-s', project+'/scripts/luka_embedding_bridge_hydra.py', 'create',
        '--root', root, '--teacher-sweep-spec', old['parent_spec']['path'],
        '--transfer-root', old['transfer_root'], '--transfer-hash', old['transfer_hash'],
        '--native-root', old['native_root'], '--native-hash', old['native_hash'],
        '--expected-commit', commit, '--quota-gib', str(old['available_quota_gib']), '--resume-spec', old_path]
if old.get('historical_selective'):
    args += ['--selective-spec', old['historical_selective']['spec']['path']]
subprocess.run(args, check=True)
PY
then
    tail -n 100 -- "$REG_LOG"
    exit 1
fi
bash "${PROJECT}/scripts/queue_luka_embedding_bridge_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

The expected setup message is `reused 49 authenticated arrays; no particle
reconstruction`, followed by the frozen-U100 check. Any remaining mismatch
produces `u100_replay.json` with historical raw/standardized errors and local
CPU/GPU comparisons. Execution v4 uses the explicitly versioned bounds in the
plan/contract; do not tune them per job or edit old manifests. Source92397736 and its completed
preparation remain recovery provenance and must be retained.

Inspect the exact setup/job log, resource JSON and submission journal. A failed
setup never releases science. Partial/ambiguous submission preserves claims
and refuses automatic retry; reconcile accepted IDs before any further submit.
Do not delete claims, mutate a campaign spec, reuse a partial root as complete,
or cancel by broad job-name prefix. Queue readiness means readiness for this
real qualification, not a claim that it has already passed on Brown.


## Numerical diagnostic after setup163911

**Completed as job164226. No need to submit it again for the v4 retry.**
GPU default/math matched exactly; CPU/GPU RMS was5.16e-7, versus historical
RMS6.25e-5. The command below is retained as the diagnostic record. The v4 setup
automatically repeats the bounded check as part of its new qualification.

The e0f9d874 recovery reused49 authenticated arrays in401.10s, but the original
batch witness still failed (max0.00145745, RMS5.68152e-5). None of the128 TRAIN
scales is near the0.001 floor. Neither batch regrouping nor a floor-valued
normalization scale explains the failure. Do not repeat full setup or change
its tolerance based solely on these numbers.

Run this on Brown with the existing e0f9d874 checkout; no new commit is needed.
It submits one30-minute diagnostic allocation with one GTX1080Ti,6CPU and16GB.
It authenticates source and consumed data/checkpoint artifacts, evaluates four
evenly spaced full original TRAIN batches plus the tail, and compares GPU
FP32 default, GPU FP32 math attention and CPU FP32 math attention. The raw
reference is reconstructed from saved standardized targets and TRAIN moments,
so its error includes their float32 storage rounding. This is diagnostic
inference only: no preparation, training, target rewrite or science release.
Only the Slurm log is written. Keep both completed-preparation roots.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
LOG="$(mktemp "$BASE/runs/u100_numerical_probe.XXXXXX.out")"
JOB="$(sbatch --parsable --no-requeue --nodes=1 --ntasks=1 \
  --partition=gpus --account=default --qos=normal \
  --gres=gpu:geforce_gtx_1080_ti:1 --cpus-per-task=6 \
  --mem=16000M --time=00:30:00 --export=NONE \
  --job-name=lukaeb_numprobe --output="$LOG" --chdir="$BASE" <<'SLURM'
#!/bin/bash
set -euo pipefail
SPEC=/cs/home/rlyang/hlt_u100_pairs/runs/embedding_bridge_e0f9d874_1080ti_v3_r1/campaign_spec.json
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PROJECT/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
"${ENV_DIR}/bin/python" -u -B -s - "$SPEC" <<'PY'
"""Read-only frozen-U100 numerical diagnostic; cannot authorize campaign release."""
import json
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from hlt_classification.luka_embedding_bridge_hydra import campaign, contracts, inputs, response
from hlt_classification.luka_embedding_bridge_hydra.model import Model
from hlt_classification.luka_u100_pair_hydra.precision import backend_settings, fixed_batches, strict_math


def compare(actual, expected):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid comparison arrays')
    delta = np.abs(a - b)
    return dict(max_abs=float(delta.max()), rms=float(np.sqrt(np.mean(delta**2))),
                mean_abs=float(delta.mean()), p999=float(np.quantile(delta, .999)))


def probe(model, batches, rows, expected, mean, scale):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    values = {}
    raw_reference = expected.astype(np.float64) * scale + mean
    for mode in ('gpu_fp32_default', 'gpu_fp32_math', 'cpu_fp32_math'):
        print('Starting ' + mode, flush=True)
        started = time.perf_counter()
        device = 'cpu' if mode.startswith('cpu') else 'cuda'
        model.to(device).float().eval()
        parts = []
        with (strict_math() if mode.endswith('_math') else nullcontext()):
            settings = backend_settings()
            with torch.inference_mode(), torch.autocast(device, enabled=False):
                for batch in batches:
                    x = {k: torch.from_numpy(batch[k]).to(device) for k in ('features', 'vectors', 'mask')}
                    _, h = model.encode(**x)
                    parts.append(h.float().cpu().numpy())
        z = np.concatenate(parts)
        normalized = response.normalize(z, mean, scale)
        delta = np.abs(normalized.astype(np.float64) - expected)
        worst = np.unravel_index(delta.argmax(), delta.shape)
        values[mode] = normalized
        print(json.dumps(dict(mode=mode, seconds=time.perf_counter()-started,
            standardized=compare(normalized, expected), raw=compare(z, raw_reference),
            exceeding_coordinates=int(np.count_nonzero(delta > 1e-4 + 1e-4*np.abs(expected))),
            worst=dict(train_row=int(rows[worst[0]]), coordinate=int(worst[1]),
                       scale=float(scale[worst[1]]), expected=float(expected[worst]),
                       actual=float(normalized[worst])), settings=settings), allow_nan=False), flush=True)
    for name in ('gpu_fp32_default', 'gpu_fp32_math'):
        print(json.dumps(dict(comparison=name+'_vs_cpu_fp32_math',
                              standardized=compare(values[name], values['cpu_fp32_math']))), flush=True)
    print(json.dumps(dict(comparison='gpu_default_vs_gpu_math',
                          standardized=compare(values['gpu_fp32_default'], values['gpu_fp32_math']))), flush=True)
    return values


def main():
    print('Authenticating saved source, TRAIN inputs, target bank and frozen checkpoint.', flush=True)
    with open(sys.argv[1]) as handle:
        spec = json.load(handle)
    contracts.validate(spec, 'SPEC')
    if spec['recipe'] != contracts.recipe() or campaign.current(spec['project_dir'], spec['source_commit']) != spec['source_snapshot']:
        raise ValueError('Saved source/recipe differs')
    ds = inputs.Dataset(spec)
    view = ds.view('train', 'U100')
    u = ds.u100('train')
    indices = fixed_batches(len(view))
    rows = np.concatenate(indices)
    batches = [view.batch(ix) for ix in indices]
    transfer = inputs.transferred(spec)
    path = transfer.path(transfer.manifest['teacher'])
    model = Model('TR').eval()
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    model.requires_grad_(False)
    if not torch.cuda.is_available():
        raise RuntimeError('Run inside the requested GPU allocation')
    torch.set_num_threads(6)
    print(json.dumps(dict(diagnostic_only=True, spec=spec['content_hash'],
        teacher=transfer.manifest['teacher']['sha256'], original_bank=spec['embedding_hash'],
        rows=len(rows), batch_starts=[int(ix[0]) for ix in indices],
        batch_lengths=[len(ix) for ix in indices], gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())), flush=True)
    probe(model, batches, rows, np.array(u['targets'][rows]), u['mean'], u['scale'])
    print('Diagnostic complete. No training, preparation, target changes or campaign release.', flush=True)


if __name__ == '__main__':
    main()
PY
SLURM
)"
printf 'Diagnostic job: %s\nLog: %s\nFollow it with: tail -f %q\n' "$JOB" "$LOG" "$LOG"
BASH
```

The log prints per-mode raw/standardized errors and pairwise CPU/GPU differences.
Interpret those before deciding whether a new qualification policy is justified.
The command was locally checked with native Weaver CPU/CUDA inference, then
executed on Brown in164226. The revised full two-GPU qualification remains pending.

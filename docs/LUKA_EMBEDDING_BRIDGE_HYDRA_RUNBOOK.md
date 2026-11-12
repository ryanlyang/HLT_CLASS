# Queue the embedding bridge, response and regional campaign

This campaign has **30 scientific fits, including a fresh U100-w5 control**,
plus setup, automatic release and report:33 jobs total. Setup and TP request
**two RTX A6000s**; the other29 scientific fits request **two GTX1080Ti cards**.
Each GPU job uses one node,6CPU,48000MiB and FP32/SyncBatchNorm/global batch128.
The [plan](plans/LUKA_EMBEDDING_BRIDGE_RESPONSE_REGION_HYDRA_PLAN.md) fixes
the scientific comparisons; see the [contract](contracts/LUKA_EMBEDDING_BRIDGE_HYDRA.md)
and [validation record](LUKA_EMBEDDING_BRIDGE_HYDRA_VALIDATION.md).

Keep the original Hydra bundle, teacher-sweep spec and its authenticated inputs,
environment, original assignments and source worktrees. Execution v2 reuses the
completed physical export already on Brown and adds a minimal native delta.
It copies exact historical tensors instead of re-encoding them on Brown.
Do not rerun the failed v1 source or reuse its partial preparation as complete.

**For failed setup165857, use the recovery block below after committing v6.**
Preserve both the original92397736 preparation and the27a83412 completed
miniatures. V6 reuses the29 unaffected miniatures/resource profiles, batches
the expensive crop-input calculations, and remeasures TP on A6000s first. It requires no
new SPORC export, raw-particle reconstruction or persistent crop cache.
The complete campaign releases automatically only after the new checks pass.
The faster TP has not yet been timed on Brown. If still above48h, preserve its
new `setup/resource_TP.json` (or `tp_training_timing.json` if training alone
already exceeds the limit). Hardware is explicitly registered now; resource
admission still requires measured A6000 results. Other tasks retain their
original Pascal measurements and allocation requests.

## 1. Commit and push from Windows

Review the scoped diff. The source guard preserves donor runtime at
58d0c0edf5f41cff2398fb168fc16fc92e169294 for this repair (original scientific
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
    'tests/test_luka_embedding_bridge_activation.py',
    'tests/test_luka_embedding_bridge_regional_inputs.py',
    'tests/test_luka_embedding_bridge_qualified.py',
    'tests/test_luka_embedding_bridge_hardware.py',
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
git commit -m 'Move TP to paired A6000s and recover completed bridge qualification'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Existing inputs for this recovery

Both transfers already exist on Brown. Do not repeat SPORC export or rsync.
Keep `embedding_bridge_raw_0a00441f`, `embedding_bridge_native_92397736`,
the92397736 prepared-data root and the27a83412 completed-qualification root.
V6 requires the latter's genuine Pascal measurements, so use the complete
recovery block below to register and queue this execution.

The queue helper reviews the complete bounded graph and asks for `YES` once.
It submits A6000 setup and an afterok CPU release. Setup success automatically
queues TP on A6000s, the29 other fits on Pascal, and the CPU report. Only real
producer dependencies are used. No later confirmation or login polling is
needed. A finite weak teacher still releases its children.

## 3. Read results and progress

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
SPECS=("$HOME"/hlt_u100_pairs/runs/embedding_bridge_*_mixed_v6_r1/campaign_spec.json)
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
its strict numerical replay. Diagnostic164226 supports the explicitly
versioned v4 numerical policy. Setup164517 passed that check but ran out of
GPU memory in TP. V5 setup165857 passed memory and all miniatures/parity, then
rejected TP's runtime. Commit/push v6 before running the block below. The
original92397736 root supplies prepared arrays;27a83412 supplies the29
unaffected completed miniatures and conservative resource measurements.

After165857, first commit/push the scoped repair as above, then run this on
Brown. Paste that new40-character commit. Registration checks the old terminal
jobs and all completed preparation hashes; allow several minutes for NFS reads.
The new setup links49 array payloads (32 view arrays,12 physical arrays and
5 U100 arrays), remeasures TP on two A6000s, and repeats current-runtime parity. It does not
rerun the other29 fits. The storage reservation credits
those links, so it does not demand a second full33GiB allocation. A fresh
root and source snapshot distinguish this execution from the failed run.

With the existing payload inventory, recovery needs approximately15.88GiB
free including the1GiB reserve; keep20GiB free for breathing room. This is
additional space alongside the preserved inputs and old roots. The inherited
`--quota-gib` value is the original total-output reservation (at least33);
recovery credits the already allocated hard-linked arrays when checking new
disk demand. Do not replace that inherited value with16 or20. The block below
prints the exact remaining requirement from the saved array inventory.

```bash
bash <<'BASH'
set -euo pipefail
trap 'printf "Stopped; preserve the old and new roots and paste the error.\n" >&2' ERR
BASE="$(cd "$HOME/hlt_u100_pairs" && pwd -P)"
OLD="${BASE}/runs/embedding_bridge_92397736_1080ti_v2_r1/campaign_spec.json"
QUALIFIED="${BASE}/runs/embedding_bridge_27a83412_1080ti_v5_r1/campaign_spec.json"
test -f "$OLD"
test -f "$QUALIFIED"
python3 -B -s - "$OLD" "$QUALIFIED" "$BASE" <<'PY'
import json, shutil, sys
from pathlib import Path
old_path, qualified_path, base = sys.argv[1:]
old = json.loads(Path(old_path).read_text())
qualified = json.loads(Path(qualified_path).read_text())
reuse = qualified['preparation_reuse']
if Path(reuse['source_spec']['path']).resolve() != Path(old_path).resolve():
    raise SystemExit('Unexpected prepared-data source; stop and inspect.')
credit = sum(ref['bytes'] for ref in reuse['arrays'])
if credit != reuse['payload_bytes'] or not 0 < credit < old['storage_budget_bytes']:
    raise SystemExit('Invalid saved reuse inventory; stop and inspect.')
needed = old['storage_budget_bytes'] - credit + 2**30
free = shutil.disk_usage(base).free
print(f'Recovery needs {needed/2**30:.2f} GiB free including reserve; filesystem reports {free/2**30:.2f} GiB.')
print('Keep both original run roots. Registration authenticates this inventory before reuse.')
if free < needed:
    raise SystemExit('Insufficient free space; no campaign registered or queued.')
PY
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
ROOT="${BASE}/runs/embedding_bridge_${COMMIT:0:8}_mixed_v6_r1"
test ! -e "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
REG_LOG="$(mktemp "${BASE}/runs/embedding_bridge_${COMMIT:0:8}_recovery.XXXXXX.log")"
printf 'Recovery registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s - "$OLD" "$PROJECT" "$ROOT" "$COMMIT" "$QUALIFIED" >"$REG_LOG" 2>&1 <<'PY'
import json, subprocess, sys
old_path, project, root, commit, qualified = sys.argv[1:]
with open(old_path) as handle:
    old = json.load(handle)
args = [sys.executable, '-u', '-B', '-s', project+'/scripts/luka_embedding_bridge_hydra.py', 'create',
        '--root', root, '--teacher-sweep-spec', old['parent_spec']['path'],
        '--transfer-root', old['transfer_root'], '--transfer-hash', old['transfer_hash'],
        '--native-root', old['native_root'], '--native-hash', old['native_hash'],
        '--expected-commit', commit, '--quota-gib', str(old['available_quota_gib']),
        '--resume-spec', old_path, '--qualified-spec', qualified]
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
reconstruction`, followed by crop-input byte-parity checks, six `TP checkpoint parity` checks and eighteen
`TP early backward` checks (six conditions at each of128/11/98 rows), then
`TP early memory passed`. The frozen-U100 check and new TP miniature follow.
The other29 miniatures are authenticated in their original locations. Any remaining numerical mismatch
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

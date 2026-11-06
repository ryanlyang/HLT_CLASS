# Direct OFFLINE reconstruction on Brown Hydra

Twelve new native CE+U100-w5 fits, with complete measured-set reconstruction.
See the [plan](plans/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_IMPLEMENTATION_PLAN.md),
[scientific v1 / execution R3 contract](contracts/LUKA_DIRECT_OFFLINE_RECONSTRUCTION.md) and
[validation evidence](LUKA_DIRECT_OFFLINE_RECONSTRUCTION_VALIDATION.md).

Use the existing Group A spec, raw transfer, environment and C0/C1 references.
No new SPORC export, teacher training or reference rerun is needed. Keep those
inputs and old source worktrees. The new R setup is separate from A/B/C jobs.

R3 repairs the matching/loss execution overhead exposed by failed setup153208
on commit57789d36. It retains all298 measured output slots for that dataset,
the FP32 decoder, exact SciPy assignments, full30epochs and48-hour ceiling.
Use the newly pushed commit and a fresh v3 root below. The failed v2 root,
its setup/release IDs, claims and diagnostics must remain separate; do not
requeue that setup or edit its spec. No new SPORC transfer is needed.

## Commit and push the reviewed files

In the local repository's PowerShell terminal, review and stage only this work.
The README/handoff/plans index may contain other local documentation changes;
inspect their diff before including them. Do not use `git add -A`.

```powershell
$paths = @(
    'src/hlt_classification/luka_next_supervision_hydra',
    'scripts/luka_next_supervision_hydra.py',
    'scripts/queue_luka_next_supervision_hydra.sh',
    'tests/test_luka_direct_reconstruction.py',
    'tests/test_luka_reconstruction_release.py',
    'tests/test_luka_reconstruction_batched.py',
    'docs/contracts/LUKA_DIRECT_OFFLINE_RECONSTRUCTION.md',
    'docs/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_RUNBOOK.md',
    'docs/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_VALIDATION.md',
    'docs/plans/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_IMPLEMENTATION_PLAN.md',
    'docs/plans/LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md',
    'docs/LEGACY_SOURCE_MAP.md', 'docs/HANDOFF.md',
    'docs/plans/README.md', 'README.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin $paths -and -not $_.StartsWith('src/hlt_classification/luka_next_supervision_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Batch exact reconstruction loss and assignments for Hydra'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Register R and queue setup plus automatic release

Paste the complete block in the Brown terminal. The Group A default is the
user's342e5d3a execution-v2 run; change it only if the actual spec differs.
Registration reads/checksums the raw TRAIN inventory before calculating the
space requirement. It can take several minutes; the printed log can be tailed
from another terminal. Available quota means personally available space after
the existing environment/data, not total filesystem size.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
GROUP_A_SPEC="${BASE}/runs/next_A_342e5d3a_a6000_v2_r1/campaign_spec.json"
test -f "$GROUP_A_SPEC"
read -r -p 'New pushed 40-character R commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_next_R_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$GROUP_A_SPEC")"
ROOT="${BASE}/runs/next_R_${COMMIT:0:8}_a6000_v3_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for R outputs after existing data/environment: ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REG_LOG="$(mktemp "${BASE}/runs/next_R_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s \
    "${PROJECT}/scripts/luka_next_supervision_hydra.py" create-r \
    --root "$ROOT" --group-a-spec "$GROUP_A_SPEC" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    exit 1
fi
SPEC="${ROOT}/campaign_spec.json"
python3 -B -s -c 'import json,sys; s=json.load(open(sys.argv[1])); print("R campaign:",s["root"]); print("New fits:",s["remaining"]); print("Reused:",list(s["reused"])); print("Fixed slots:",s["target_inventory"]["maximum"]); print("Reserved output GiB:",s["storage_budget_bytes"]/2**30)' "$SPEC"
bash "${PROJECT}/scripts/queue_luka_next_supervision_hydra.sh" "$SPEC" bootstrap </dev/tty
squeue --me -o '%.18i %.30j %.12T %R'
BASH
```

Review the exact bootstrap plan, twelve-fit recipe and resource ceilings, then
type only `YES`. One `lukanr_setup` job uses
one RTX A6000,6CPUs,32000MiB host RAM and a12-hour setup limit. It prepares
complete OFFLINE/HLT TRAIN targets, common slot support, native parity evidence
and six genuine warmed production measurements. No decoder capacity, particles
or epochs are silently cut to pass resources. Poor finite fit quality is not a
failure condition. Preserve a failed setup's root/claims/logs for diagnosis.

The same submission queues `lukanr_release` on compute, requesting one CPU,
4000MiB and one hour, with `afterok:<exact setup job>`. **Once both IDs appear
in the completed bootstrap ledger, you can log off.** This is a Slurm job,
not a background process on the login host.

After setup finishes `COMPLETED/0:0`, release checks the full receipts, target
support, resources, source and original authorization. It writes and dry-reviews
the measured twelve-fit plan, then queues it automatically. No second command
or confirmation is required. If setup fails, science is not submitted. Poor
finite model quality does not fail setup. Release can wait for compute capacity;
the fits then wait for normal GPU scheduling.

Submission order is `R1 R2 R3 R4 R11 R12 R5 R6 R7 R8 R9 R10`.
All twelve submit independently, without dependencies, holds, lanes or caps.
The only dependency is setup-to-release. Existing A/B/C jobs are not changed.

## Inspect the automatic release

Use the same commit as registration. This reads the accepted exact IDs and logs;
it does not submit anything.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
read -r -p 'Same 40-character R commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
PROJECT="${BASE}/code_next_R_${COMMIT:0:8}"
ROOT="${BASE}/runs/next_R_${COMMIT:0:8}_a6000_v3_r1"
SPEC="${ROOT}/campaign_spec.json"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" inspect-submission --spec "$SPEC"
for LOG in "${ROOT}"/release-*.out; do
    if [[ -f "$LOG" ]]; then tail -n 80 -- "$LOG"; fi
done
squeue --me -o '%.18i %.30j %.12T %.12l %R'
BASH
```

The bootstrap ledger contains setup/release IDs; the science ledger contains
all twelve fit IDs after release. If release fails, preserve its log, claims
and journals. An interrupted partial journal needs exact-ID inspection before
recovery; do not erase its claim, requeue setup or launch another release blindly.
Already registered R1/R2 specs retain their own pinned implementation. This
R3 workflow requires a fresh root and the newly pushed code. Its setup must
remeasure all six paths; local timing is not a substitute for Hydra acceptance.

## Read results and running epochs

Set SPEC to the exact new campaign path, then use either final line:

```bash
SPEC="/absolute/path/to/next_R_COMMIT_a6000_v3_r1/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" progress --spec "$SPEC"
```

These are read-only. They show native validation AUC/recovery, same-seed
U100-w5 comparisons, published epochs, reconstruction components and unresolved
job states. Full TRAIN mechanism probes are under each row's `fit/probe_NN.json`;
they are not independent reconstruction validation. Original50k overlaps
validation. No new inference, training, file changes or job changes occur.

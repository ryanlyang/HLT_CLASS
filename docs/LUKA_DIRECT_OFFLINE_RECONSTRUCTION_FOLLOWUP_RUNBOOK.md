# Queue the R8-centered reconstruction follow-up on Brown Hydra

This is a separate R5 / R8_FOLLOWUP_V2 campaign with eighteen scientific fits. It reuses the authenticated Group A input
bundle and existing C0/C1 reference proofs; it creates a fresh output root and
rebuilds the full TRAIN reconstruction banks in a measured setup. It does not
change the currently registered R3 campaign or any held/running jobs.

See the [scientific plan](plans/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP_PLAN.md),
[contract](contracts/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP.md) and
[validation boundary](LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP_VALIDATION.md).

## Commit and push only this implementation

Review the exact paths first. Do not use `git add -A`; the working tree has
unrelated edits. The shared `docs/HANDOFF.md` also has pre-existing edits, so
review and stage its follow-up hunk separately if desired; it is not required
for Brown source validation.

```powershell
$paths = @(
    'src/hlt_classification/luka_next_supervision_hydra/campaign.py',
    'src/hlt_classification/luka_next_supervision_hydra/contracts.py',
    'src/hlt_classification/luka_next_supervision_hydra/submission.py',
    'src/hlt_classification/luka_next_supervision_hydra/resources.py',
    'src/hlt_classification/luka_next_supervision_hydra/reconstruction.py',
    'src/hlt_classification/luka_next_supervision_hydra/reconstruction_batched.py',
    'src/hlt_classification/luka_next_supervision_hydra/reconstruction_loss.py',
    'src/hlt_classification/luka_next_supervision_hydra/reconstruction_training.py',
    'src/hlt_classification/luka_next_supervision_hydra/reconstruction_release.py',
    'src/hlt_classification/luka_next_supervision_hydra/reconstruction_followup.py',
    'src/hlt_classification/luka_next_supervision_hydra/reconstruction_model.py',
    'src/hlt_classification/luka_next_supervision_hydra/reconstruction_targets.py',
    'src/hlt_classification/luka_next_supervision_hydra/model.py',
    'src/hlt_classification/luka_next_supervision_hydra/training.py',
    'src/hlt_classification/luka_next_supervision_hydra/worker.py',
    'src/hlt_classification/luka_next_supervision_hydra/reporting.py',
    'scripts/queue_luka_next_supervision_hydra.sh',
    'tests/test_luka_direct_reconstruction.py',
    'tests/test_luka_reconstruction_batched.py',
    'scripts/luka_next_supervision_hydra.py',
    'tests/test_luka_reconstruction_followup.py',
    'docs/contracts/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP.md',
    'docs/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP_RUNBOOK.md',
    'docs/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP_VALIDATION.md',
    'docs/plans/LUKA_DIRECT_OFFLINE_RECONSTRUCTION_FOLLOWUP_PLAN.md',
    'docs/plans/README.md', 'docs/LEGACY_SOURCE_MAP.md', 'README.md'
)
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Scoped staging failed' }
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git diff --cached --stat
git commit -m 'Add eighteen-fit R8 reconstruction campaign with automatic release'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

If `git diff --cached --name-only` shows paths beyond this list from already
staged work, stop and inspect that staging before committing.

## Register and queue setup plus automatic release

Run on Brown after pushing. The expected Group A input spec is the one used to
register the existing R3 campaign. If your actual path differs, set
`GROUP_A_SPEC` to that authenticated Group A `campaign_spec.json`. Do not point
it at the R3 output root or invent a new parent. Registration authenticates the
full TRAIN inventory and C0/C1 proofs; it may take a while.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
GROUP_A_SPEC="${BASE}/runs/next_A_342e5d3a_a6000_v2_r1/campaign_spec.json"
test -f "$GROUP_A_SPEC"
read -r -p 'New pushed 40-character follow-up commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_next_R8_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$GROUP_A_SPEC")"
ROOT="${BASE}/runs/next_R8_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for outputs (at least 8): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
REG_LOG="${ROOT}_registration.$(python3 -B -s -c 'import secrets; print(secrets.token_hex(4))').log"
mkdir -p "$(dirname "$ROOT")"
if ! "${ENV_DIR}/bin/python" -u -B -s \
    "${PROJECT}/scripts/luka_next_supervision_hydra.py" create-r8-followup \
    --root "$ROOT" --group-a-spec "$GROUP_A_SPEC" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 120 -- "$REG_LOG"
    printf 'Registration failed; preserve the root/log and inspect the cause.\n' >&2
    exit 1
fi
SPEC="${ROOT}/campaign_spec.json"
printf 'Registration log: %s\n' "$REG_LOG"
"${ENV_DIR}/bin/python" -B -s -c 'import json,sys; s=json.load(open(sys.argv[1])); print("Campaign:",s["root"]); print("Fits:",s["remaining"]); print("Reserved GiB:",round(s["storage_budget_bytes"]/2**30,2)); print("Reused controls:",list(s["reused"]))' "$SPEC"
bash "${PROJECT}/scripts/queue_luka_next_supervision_hydra.sh" "$SPEC" bootstrap </dev/tty
squeue --me -o '%.18i %.30j %.12T %.12l %R'
BASH
```

Review the exact bootstrap hash and type `YES` once. This submits the measured
setup and a durable CPU release job. After the exact job IDs are recorded, you
can log off. A successful setup causes the release job to dry-review and submit
all eighteen independent science fits and a dependent CPU report automatically. A failed setup cannot
release science. The science jobs have no fit-to-fit dependencies, hold state,
priority adjustment or concurrency cap. Dependencies are setup-to-release (`afterok`) and all fits-to-report (`afterany`).
The release uses the measured setup profile; if Hydra rejects the GPU shape,
preserve the root and inspect availability before registering another profile.

## Results and progress

Use the exact printed SPEC path:

```bash
SPEC="/absolute/path/to/next_R8_COMMIT_a6000_r1/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" progress --spec "$SPEC"
```

These commands authenticate and read saved epochs/results; they do not submit
jobs or run inference. Reports compare R13/R14 to C2/C3, and each seed-0
variant to C0. The original50k slice overlaps validation. Each follow-up row also prints its weight or strategy description.

## Registered jobs and hardware

This campaign uses one A6000 per GPU job, with the original native BF16 and
FP32 decoder policy. It does not silently substitute GTX1080Ti/FP32 or change
batch size. If gpu2201 is still draining, scheduler review can reject admission;
check `sinfo -N -p gpus` before queueing. Current availability is not established
by local implementation tests.

There are21 jobs: setup, release,18 fits and report. Setup measures13 execution
paths (two miniatures each), including weights5/10 and block4+8 memory. It can
take longer than the original setup. Scientific jobs are released only after
those checks pass; no further confirmation or login-shell process is needed.
A failed individual fit does not prevent other fits or the final `afterany`
report from running. Inspect `report.json` and `report-JOBID.out` afterward.

V2 task map: C2/R13 and C3/R14 are matched pairs; R15-R22 are weights
0.15/0.2/0.5/0.7/1/2/5/10; R23 slow ramp, R24 delayed ramp, R25 late fade,
R26 consistency, R27 track emphasis, R28 block4+8. Do not reuse an older v1
follow-up root or interpret its R15-R22 IDs using this map.

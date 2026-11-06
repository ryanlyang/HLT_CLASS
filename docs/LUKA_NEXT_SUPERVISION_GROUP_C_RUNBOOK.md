# Group C on Brown Hydra: densities and full-set denoising

This adds seven independent fits alongside A/B on the same RTX A6000 profile.
Every row retains native CE+U100-w5. C0/C1 are reused. Submission order:
`D1 S1 D2 S2 D5 S5 D3`. See the [v1 contract](contracts/LUKA_NEXT_SUPERVISION_GROUP_C.md)
and [validation evidence](LUKA_NEXT_SUPERVISION_GROUP_C_VALIDATION.md).

No new SPORC export, raw transfer, teacher fit or environment installation is
needed. The existing Group A spec authenticates D000 views, raw TRAIN HLT/OFFLINE
fields, original U100 targets, reference proofs and the pinned environment.
Keep old source worktrees and references. Use a fresh clean pushed worktree and
separate C root, reserving8GiB after existing data/environment plus1GiB headroom.

## Commit and push locally

Review only the Group C/shared infrastructure, tests and documentation changes.
Do not use `git add -A` in this workspace. From the repository in PowerShell:

```powershell
$paths = @(
    'src/hlt_classification/luka_next_supervision_hydra',
    'scripts/luka_next_supervision_hydra.py',
    'tests/test_luka_next_supervision_hydra.py',
    'tests/test_luka_next_supervision_group_b.py',
    'tests/test_luka_next_supervision_group_c.py',
    'docs/contracts/LUKA_NEXT_SUPERVISION_HYDRA.md',
    'docs/contracts/LUKA_NEXT_SUPERVISION_GROUP_C.md',
    'docs/LUKA_NEXT_SUPERVISION_GROUP_C_RUNBOOK.md',
    'docs/LUKA_NEXT_SUPERVISION_GROUP_C_VALIDATION.md',
    'docs/plans/LUKA_SEVEN_CANDIDATE_IMPLEMENTATION_PLAN.md',
    'docs/plans/LUKA_NEXT_SUPERVISION_CAMPAIGN_IDEAS.md',
    'docs/plans/README.md', 'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md', 'README.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin $paths -and -not $_.StartsWith('src/hlt_classification/luka_next_supervision_hydra/')
}
if ($unexpected) { throw ('Unrelated staged work; preserve and review it first: ' + ($unexpected -join ', ')) }
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check staged whitespace' }
git commit -m 'Implement Group C density and set denoising on Hydra'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

Check the staged stat for unrelated pre-existing staged work before committing.
All new modules/tests/contracts must be tracked in the exact pushed commit.

## Brown: register C and queue its technical setup

The default A path below is the user's342e5d3a execution-v2 root. If the actual
A spec is elsewhere, set GROUP_A_SPEC to that exact file. A science does not
need to finish. Registration authenticates its original source in a separate
process; the printed log path allows monitoring a quiet checksum phase.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
GROUP_A_SPEC="${BASE}/runs/next_A_342e5d3a_a6000_v2_r1/campaign_spec.json"
test -f "$GROUP_A_SPEC"
read -r -p 'New pushed 40-character Group C commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_next_C_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$GROUP_A_SPEC")"
ROOT="${BASE}/runs/next_C_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
REG_LOG="$(mktemp "${BASE}/runs/next_C_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s \
    "${PROJECT}/scripts/luka_next_supervision_hydra.py" create-c \
    --root "$ROOT" --group-a-spec "$GROUP_A_SPEC" \
    --expected-commit "$COMMIT" --quota-gib 8 >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    exit 1
fi
SPEC="${ROOT}/campaign_spec.json"
python3 -B -s -c 'import json,sys; s=json.load(open(sys.argv[1])); print("C campaign:",s["root"]); print("New fits:",s["remaining"]); print("Reused:",list(s["reused"]))' "$SPEC"
bash "${PROJECT}/scripts/queue_luka_next_supervision_hydra.sh" "$SPEC" setup </dev/tty
squeue --me -o '%.18i %.30j %.12T %R'
BASH
```

Review the exact command and type only YES. `lukanc_setup` prepares frozen
mixture centers and full normalized OFFLINE/HLT set banks, then measures four
real warmed production paths. Full TRAIN multiplicities are retained. Actual
cardinality tails, missing-channel support and measured limits appear in the
science review. A finite poor objective or AUC cannot reject science admission.

**Setup does not automatically queue science.** After exact COMPLETED/0:0,
run the next block. Preserve all diagnostics and claims if setup fails;
do not retry in the same root or modify its immutable spec.

## Queue all seven scientific fits after successful setup

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
read -r -p 'Same 40-character Group C commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
PROJECT="${BASE}/code_next_C_${COMMIT:0:8}"
SPEC="${BASE}/runs/next_C_${COMMIT:0:8}_a6000_r1/campaign_spec.json"
bash "${PROJECT}/scripts/queue_luka_next_supervision_hydra.sh" "$SPEC" science </dev/tty
squeue --me -o '%.18i %.30j %.12T %.12l %R'
BASH
```

Review the frozen target diagnostics and measured walltimes, then type YES.
All seven jobs submit together without dependencies, holds, nice adjustments,
arrays or family concurrency caps. They can run alongside A/B as Slurm permits.
Completed exact submission is durable after logout. A partial journal/claim
requires exact-ID inspection before recovery. No dependent report job or
automatic retry is added. This is a C-only release, not a requeue of all40 rows.

## Read completed results and running epochs

Set SPEC to the exact new campaign file:

```bash
SPEC="/absolute/path/to/next_C_COMMIT_a6000_r1/campaign_spec.json"
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" results --spec "$SPEC"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_next_supervision_hydra.py" progress --spec "$SPEC"
```

These read saved validation AUC, historical recovery, same-seed C0/C1 deltas
and published epochs without inference, writes or job changes. All incomplete
rows remain visible. Original50k overlaps validation; there is no independent
test. Better density/denoising loss alone is not evidence of better recovery.

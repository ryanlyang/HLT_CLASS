# Queue selective-information teachers on Hydra

This campaign adds **four teachers + eight HLT students**. Every student
predicts U100 at weight5 and shares another weight5 across its other targets.
The [plan](plans/LUKA_SELECTIVE_TEACHER_HYDRA_PLAN.md) lists exact configurations;
the [contract](contracts/LUKA_SELECTIVE_TEACHER_HYDRA.md) defines lineage and
the [validation record](LUKA_SELECTIVE_TEACHER_HYDRA_VALIDATION.md) states what
was tested locally.

**Normally no new SPORC transfer is needed.** Registration reuses the existing
bundle, frozen-adapter validation U100 delta, environment, U100 bank, C0 result,
and completed teacher-sweep T00/S00. Keep those artifacts and their original
worktrees. Other teacher-sweep tasks may still be running; only T00/S00 are
required completed parents. No existing jobs are modified or canceled.

If registration at `11499f08` stopped with a contrastive/frozen-adapter contract
mismatch, use the corrected source described here. That failure occurred
before campaign-directory creation and before submission. Commit/push the fix,
then run step 2 with the **new** commit; it selects a new checkout/output path.
Keep the earlier registration log. No job cancellation or new transfer is
needed for this error: Group A's cross-message parent identifies the correct
existing validation delta. The validator is not relaxed or bypassed.

## 1. Commit and push in the local PowerShell terminal

Run from this repository. This list excludes unrelated current work. The
source guard permits only additive selective-teacher runtime since donor
`6fe7e3499767f0a2216586d965efe39a05c24bf7`.

```powershell
$paths = @(
    'src/hlt_classification/luka_selective_teacher_hydra',
    'scripts/luka_selective_teacher_hydra.py',
    'scripts/queue_luka_selective_teacher_hydra.sh',
    'sbatch/run_luka_selective_teacher_hydra.sh',
    'tests/test_luka_selective_teacher_hydra.py',
    'docs/plans/LUKA_SELECTIVE_TEACHER_HYDRA_PLAN.md',
    'docs/contracts/LUKA_SELECTIVE_TEACHER_HYDRA.md',
    'docs/LUKA_SELECTIVE_TEACHER_HYDRA_RUNBOOK.md',
    'docs/LUKA_SELECTIVE_TEACHER_HYDRA_VALIDATION.md',
    'README.md', 'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md'
)
$unexpected = @(git diff --cached --name-only) | Where-Object {
    $_ -notin $paths -and -not $_.StartsWith('src/hlt_classification/luka_selective_teacher_hydra/')
}
if ($unexpected) { throw ('Review existing staged work first: ' + ($unexpected -join ', ')) }
git diff -- $paths
git add -- $paths
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Fix staged whitespace first' }
git commit -m 'Add selective-information teachers and multi-target Hydra students'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Register and queue in the Brown terminal

Paste the entire block, preserving newlines. Supply the new pushed full commit,
at least9 GiB of personally available output quota, and `YES` at the final
review. Registration checks original artifacts and source, so its log can be
quiet for several minutes. The printed log can be tailed in another terminal.

```bash
bash <<'BASH'
set -euo pipefail
BASE="$(cd "$HOME" && pwd -P)/hlt_u100_pairs"
MAIN="${BASE}/code_65952ed3"
PARENT="${BASE}/runs/teacher_sweep_6fe7e349_a6000_r1/campaign_spec.json"
test -f "$PARENT"
read -r -p 'New pushed selective-teacher commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$MAIN" fetch origin main
test "$(git -C "$MAIN" rev-parse origin/main)" = "$COMMIT"
PROJECT="${BASE}/code_selective_teacher_${COMMIT:0:8}"
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$MAIN" worktree add --detach "$PROJECT" "$COMMIT"
fi
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$PARENT")"
ROOT="${BASE}/runs/selective_teacher_${COMMIT:0:8}_a6000_r1"
test ! -e "$ROOT"
df -h "$BASE"
read -r -p 'Personally available GiB for outputs (at least 9): ' QUOTA_GIB </dev/tty
[[ "$QUOTA_GIB" =~ ^[1-9][0-9]*$ ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
CLI="${PROJECT}/scripts/luka_selective_teacher_hydra.py"
REG_LOG="$(mktemp "${BASE}/runs/selective_teacher_${COMMIT:0:8}_registration.XXXXXX.log")"
printf 'Registration log: %s\n' "$REG_LOG"
if ! "${ENV_DIR}/bin/python" -u -B -s "$CLI" create \
    --root "$ROOT" --teacher-sweep-spec "$PARENT" \
    --expected-commit "$COMMIT" --quota-gib "$QUOTA_GIB" >"$REG_LOG" 2>&1; then
    tail -n 100 -- "$REG_LOG"
    printf 'Registration stopped; preserve the root and log.\n' >&2
    exit 1
fi
printf 'Registered spec: %s/campaign_spec.json\n' "$ROOT"
bash "${PROJECT}/scripts/queue_luka_selective_teacher_hydra.sh" "${ROOT}/campaign_spec.json" </dev/tty
squeue --me -o "%.18i %.32j %.12T %.12l %R"
BASH
```

Expect `lukasi_setup` and `lukasi_release` IDs. Once both IDs are recorded you
can log off. A successful measured GPU setup automatically triggers the CPU
release, which queues all12 fits and CPU report. There is no later manual
approval. Four teacher jobs are independent. Students submit in order
`S6,S5,S4,S3,S2IT,S2T,S2I,S2K`; each waits only for its own required teachers.
This encourages larger target sets first but does not override Slurm scheduling.
Each fit requests one typed RTX A6000, six CPUs and32000MiB; walltime is measured.

Poor finite teacher accuracy does not block its students. A failed technical
setup cannot release science. A mistyped `YES` before submission can be corrected
by rerunning only the queue helper on the same spec. Preserve failed/ambiguous
claims and journals; do not recreate a campaign or delete evidence to retry.

## 3. Epochs and completed recoveries

Use `MODE=progress` for every published epoch, `MODE=results` for authenticated
completed selections, or `MODE=inspect-submission` for exact IDs.

```bash
bash <<'BASH'
set -euo pipefail
shopt -s nullglob
SPECS=("$HOME"/hlt_u100_pairs/runs/selective_teacher_*_a6000_r1/campaign_spec.json)
[[ ${#SPECS[@]} -eq 1 ]] || { printf 'Select one exact spec; found %s\n' "${#SPECS[@]}"; exit 1; }
SPEC="${SPECS[0]}"
MODE=progress
PROJECT="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "$SPEC")"
ENV_DIR="$(python3 -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$SPEC")"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_selective_teacher_hydra.py" "$MODE" --spec "$SPEC"
BASH
```

K/I/T/IT are privileged oracle teachers. S rows are native HLT-only students.
Compare them with `C0_U100_HYDRA` and `S00_U100_OFFLINE`; the latter isolates
whether selective targets improve over the already helpful extra OFFLINE
embedding. Recovery uses historical M0HLT/OFFLINE anchors, not the new teacher
as denominator. Original50k overlaps validation. Single-seed results do not
establish a reproducible gain.

## Recovery only: missing old validation delta

The normal path above reuses existing inputs. If registration identifies
missing validation U100 files, the new CLI also supports a minimal replacement
export on SPORC. Use the new clean pushed worktree and original SPORC environment:

```bash
"${ENV_DIR}/bin/python" -u -B -s "${PROJECT}/scripts/luka_selective_teacher_hydra.py" export \
    --origin-spec /home/ryreu/atlas/HLT_Classification/checkpoints/luka_u100_pair_correction_c08f6df6_debug_r1/campaign_spec.json \
    --output /home/ryreu/atlas/luka_selective_validation_NEW_SUFFIX \
    --expected-commit "$COMMIT"
```

This reconstructs only absent original validation U100 shards with saved
assignments, verifies their exact historical bytes and full fingerprint, and
publishes a completion manifest last. It never writes the old preparation.
Transfer this completed directory to Brown, preserving relative paths. Verify:

```bash
"${ENV_DIR}/bin/python" -B -s "$CLI" verify-transfer \
    --validation-root "$VALIDATION" --validation-hash "$VALIDATION_HASH" --bundle-root "$BUNDLE"
```

Then add `--validation-root "$VALIDATION" --validation-hash "$VALIDATION_HASH"`
to `create`, using a fresh output root. Existing corrupt files are errors, not
permission to rebuild/overwrite them. This alternative needs no new training
or full dataset transfer.

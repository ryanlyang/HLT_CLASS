# Queue the two-fit SupCon-5 campaign on SPORC debug

This queues one OFFLINE CE+5×SupCon teacher, then one fresh HLT student with
CE + U100 embedding weight5 + new-teacher embedding weight5. It reuses the saved
U100-w5 baseline. There are two additional support jobs: setup and report.
See the [plan](plans/LUKA_SUPCON5_DEBUG_PLAN.md),
[contract](contracts/LUKA_SUPCON5.md) and [validation evidence](LUKA_SUPCON5_VALIDATION.md).

## 1. Commit and push from Windows PowerShell

Run from the repository. The commands stage only this implementation. Review
the staged file list and exclude unrelated staged work before committing; the
new source validator permits only the new runtime files since donor
`00fbb92641553e8817b577b8765796e6618afc13`.

```powershell
$Files = @(
    'src/hlt_classification/luka_supcon5',
    'scripts/luka_supcon5.py',
    'scripts/queue_luka_supcon5.sh',
    'sbatch/run_luka_supcon5.sh',
    'tests/test_luka_supcon5.py',
    'docs/plans/LUKA_SUPCON5_DEBUG_PLAN.md',
    'docs/contracts/LUKA_SUPCON5.md',
    'docs/LUKA_SUPCON5_DEBUG_RUNBOOK.md',
    'docs/LUKA_SUPCON5_VALIDATION.md',
    'README.md', 'docs/HANDOFF.md', 'docs/LEGACY_SOURCE_MAP.md'
)
git add -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
$Staged = @(git diff --cached --name-only)
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect staged files' }
$Unexpected = @($Staged | Where-Object {
    $_ -notin $Files -and -not $_.StartsWith('src/hlt_classification/luka_supcon5/')
})
if ($Unexpected.Count) { throw "Unrelated staged files: $Unexpected" }
git diff --cached --stat
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Check the staged changes' }
git commit -m "Add two-fit OFFLINE SupCon-5 transfer campaign"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

Copy the full 40-character commit. A fresh detached SPORC worktree keeps other
campaign sources unchanged. Do not clean or reset the main checkout.

## 2. Inspect debug availability from the SPORC terminal

```bash
sinfo -p debug -N -o "%.18N %.12t %.35G %.18C %.12m"
squeue -p debug -o "%.18i %.12u %.30j %.2t %.12L %R"
df -h /home/ryreu/atlas/HLT_Classification/checkpoints
```

Availability is not assumed. Requests use one A100, six CPUs, 90000 MiB, account
`reu-aisocial`, QOS `qos_tier3`. Setup requests 12 hours; actual science limits
come from measurements and cannot exceed 24 hours per fit. The launcher tests
the exact resource shapes against the live SPORC controller before submission.
It needs at least 8 personally available output GiB plus filesystem headroom.

## 3. Register and queue from SPORC

Paste the complete block, preserving line breaks. Supply the pushed commit and
your available output space at its prompts. Existing datasets and U100 targets
are reused; there is no transfer or regeneration of the original control.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped; preserve outputs and paste the error." >&2' ERR
read -r -p "Pushed 40-character SupCon-5 commit: " COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
read -r -p "Personally available output GiB (at least 8): " QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ && "${QUOTA}" -ge 8 ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_supcon5_${COMMIT:0:8}"
REFERENCE="${MAIN}/checkpoints/luka_u100_contrastive_431341db_debug_r1/campaign_spec.json"
test -f "${REFERENCE}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_supcon5.sh" "${REFERENCE}" "${QUOTA}"
BASH
```

The reference must be the original contrastive campaign containing completed
`u100_w5_s0` job21993056, at its registered path. The importer verifies that
execution in its original worktree. A pair-correction or Hydra spec is not a
substitute. Authentication can take time while hashing historical artifacts.

The command performs the complete dry review and starts a detached dispatcher.
Tail the printed `submit.*.log` until it confirms the exact setup job ID. After
setup completes successfully, the dispatcher validates its measurements and
queues teacher, student and report automatically. The student waits for its
teacher's authenticated TRAIN bank; no further user action is needed for release.

The submit-host dispatcher survives logout via `nohup`, but not a submit-host
restart or process termination. If setup finishes without science appearing,
inspect the dispatcher log and exact submission journals. Preserve existing
claims and outputs; do not delete claims or blindly resubmit a campaign root.

## 4. Epoch histories and completed recoveries

Run on SPORC and enter the same commit. Change `MODE=epochs` to `MODE=results`
for authenticated selected-checkpoint results. Both commands are read-only and
perform no inference. Epoch summaries distinguish the privileged OFFLINE teacher
from the HLT student; the teacher's recovery is not deployable HLT performance.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p "Campaign 40-character commit: " COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MODE=epochs
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_supcon5_${COMMIT:0:8}_debug_r1"
SPEC="${ROOT}/campaign_spec.json"
ENV_DIR=/home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${ENV_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PROJECT="$("${ENV_DIR}/bin/python" -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PYTHONPATH="${PROJECT}/src"
"${ENV_DIR}/bin/python" -B -s "${PROJECT}/scripts/luka_supcon5.py" "${MODE}" --spec "${SPEC}"
BASH
```

Weighted loss averages and shared-encoder gradient diagnostics are in each
`<task>/fit/epoch-XX.json`. Norms already include the objective coefficients.
Scientific value is determined by HLT performance relative to the saved U100-w5
baseline, not the teacher's SupCon loss alone. With only these two fits, a gain
does not isolate teacher SupCon from the benefit of adding another OFFLINE head.

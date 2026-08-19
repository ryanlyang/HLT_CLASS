# HCWDL-MHPE Full-Data D000 Teacher-Distance Screen Runbook

This queues the 24-fit, 80-pass full-data C25P75 screen. It requires one
authenticated full-data source with U000, U100E, D066E, and D033E ready; the
source campaign itself need not be complete. It does not submit a new smoke.

```bash
set -euo pipefail
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
cd "${MAIN_REPO}"
git fetch origin main
export SCREEN_COMMIT="$(git rev-parse origin/main)"
export SCREEN_SHORT="${SCREEN_COMMIT:0:8}"
export SCREEN_WORKTREE="/home/ryreu/atlas/HLT_Classification_mhpe_d000_screen_${SCREEN_SHORT}"
export SCREEN_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_d000_teacher_screen_${SCREEN_SHORT}_r1"

test ! -e "${SCREEN_ROOT}"
if [ -e "${SCREEN_WORKTREE}" ]; then
  test "$(git -C "${SCREEN_WORKTREE}" rev-parse HEAD)" = "${SCREEN_COMMIT}"
  test -z "$(git -C "${SCREEN_WORKTREE}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach "${SCREEN_WORKTREE}" "${SCREEN_COMMIT}"
fi
export PYTHONPATH="${SCREEN_WORKTREE}/src"

export SOURCE_SPEC="$(
SCREEN_WORKTREE="${SCREEN_WORKTREE}" MAIN_REPO="${MAIN_REPO}" python -s - <<'PY'
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(os.environ["SCREEN_WORKTREE"]) / "src"))
from hlt_classification.scouting.hcwdl_mhpe_d000_schedule_screen import authenticate_source
candidates = []
for path in sorted((Path(os.environ["MAIN_REPO"]) / "checkpoints").glob("*/campaign_spec.json")):
    try:
        authenticate_source(path)
    except Exception:
        continue
    candidates.append(path.resolve())
if len(candidates) != 1:
    raise SystemExit(
        "Expected exactly one authenticated full-data C25P75 MHPE source with "
        "U000/U100E/D066E/D033E ready; "
        f"found {len(candidates)}:\n" + "\n".join(map(str, candidates))
    )
print(candidates[0])
PY
)"

python -s "${SCREEN_WORKTREE}/scripts/create_hcwdl_mhpe_d000_schedule_screen.py" \
  --source-campaign-spec "${SOURCE_SPEC}" --campaign-root "${SCREEN_ROOT}" \
  --project-dir "${SCREEN_WORKTREE}" --source-commit "${SCREEN_COMMIT}" \
  --authorize-live-submission --authorization-phrase \
    "AUTHORIZE HCWDL MHPE FULL C25P75 D000 TEACHER DISTANCE SCREEN EXACT SPEC" \
  --operational-evidence-waiver --waiver-phrase \
    "AUTHORIZE HCWDL MHPE D000 TEACHER DISTANCE SCREEN CARRIED OPERATIONAL EVIDENCE"

python -s "${SCREEN_WORKTREE}/scripts/submit_hcwdl_mhpe_d000_schedule_screen.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --output "${SCREEN_ROOT}/dry_run_submission_ledger.json"

python -s - "${SCREEN_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
spec = json.loads((root / "campaign_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())
assert spec["fit_count"] == 24
assert spec["heldout_evaluation_count"] == 96
assert len(spec["tasks"]) == 26 == len(plan["commands"])
assert spec["resources"]["gpu"] == {
    "cpus": 8, "memory": "96G", "walltime": "72:00:00", "gpu": "gpu:gh200:1",
}
assert spec["ordinary_access_role_counts"]["final_test"] == 0
print("Source:", spec["source"]["source_spec_path"])
print("Fits:", spec["fit_count"], "held-out evaluations:", spec["heldout_evaluation_count"])
print("Tasks:", len(plan["commands"]), "GPU request:", spec["resources"]["gpu"])
PY

python -s "${SCREEN_WORKTREE}/scripts/submit_hcwdl_mhpe_d000_schedule_screen.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --output "${SCREEN_ROOT}/submission_ledger.json" --execute \
  --authorization-phrase \
    "SUBMIT HCWDL MHPE FULL C25P75 D000 TEACHER DISTANCE SCREEN EXACT LEDGER"

squeue --me -o "%.18i %.55j %.2t %.10M %R" | grep -E 'JOBID|hcwd0s_'
```

If source discovery reports zero candidates, inspect which of U000, U100E,
D066E, or D033E is absent or invalid. Do not wait for unrelated later source
rungs or whole-campaign completion, and never substitute a 300k, C10P90, or
path-only source.

## Monitor and exact-ID cancellation

```bash
python -s "${SCREEN_WORKTREE}/scripts/monitor_hcwdl_mhpe_d000_schedule_screen.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --submission-ledger "${SCREEN_ROOT}/submission_ledger.json" \
  --output "${SCREEN_ROOT}/monitor_report.json"

# Print the exact campaign-bound cancellation command. Add --execute only if
# cancellation is actually intended.
python -s "${SCREEN_WORKTREE}/scripts/cancel_hcwdl_mhpe_d000_schedule_screen.py" \
  --submission-ledger "${SCREEN_ROOT}/submission_ledger.json"
```

## Failed/downstream recovery

Create recovery only from the immutable original ledger and a newly generated
monitor report. The recovery closure preserves completed fits and all four
horizon checkpoints.

```bash
export RECOVERY_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export RECOVERY_SHORT="${RECOVERY_COMMIT:0:8}"
export RECOVERY_WORKTREE="/home/ryreu/atlas/HLT_Classification_mhpe_d000_screen_recovery_${RECOVERY_SHORT}"
export RECOVERY_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_d000_teacher_screen_recovery_${RECOVERY_SHORT}_r1"

test ! -e "${RECOVERY_ROOT}"
if [ -e "${RECOVERY_WORKTREE}" ]; then
  test "$(git -C "${RECOVERY_WORKTREE}" rev-parse HEAD)" = "${RECOVERY_COMMIT}"
  test -z "$(git -C "${RECOVERY_WORKTREE}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach \
    "${RECOVERY_WORKTREE}" "${RECOVERY_COMMIT}"
fi

python -s "${RECOVERY_WORKTREE}/scripts/create_hcwdl_mhpe_d000_schedule_screen_recovery.py" \
  --campaign-spec "${SCREEN_ROOT}/campaign_spec.json" \
  --submission-ledger "${SCREEN_ROOT}/submission_ledger.json" \
  --monitor-report "${SCREEN_ROOT}/monitor_report.json" \
  --recovery-root "${RECOVERY_ROOT}" \
  --project-dir "${RECOVERY_WORKTREE}" \
  --source-commit "${RECOVERY_COMMIT}" \
  --authorization-phrase \
    "AUTHORIZE HCWDL MHPE D000 TEACHER DISTANCE SCREEN EXACT RECOVERY"

python -s "${RECOVERY_WORKTREE}/scripts/submit_hcwdl_mhpe_d000_schedule_screen_recovery.py" \
  --recovery-spec "${RECOVERY_ROOT}/recovery_spec.json" \
  --output "${RECOVERY_ROOT}/dry_run_submission_ledger.json"

# Execute only after inspecting the recovery spec, closure, and dry ledger.
python -s "${RECOVERY_WORKTREE}/scripts/submit_hcwdl_mhpe_d000_schedule_screen_recovery.py" \
  --recovery-spec "${RECOVERY_ROOT}/recovery_spec.json" \
  --output "${RECOVERY_ROOT}/submission_ledger.json" --execute
```

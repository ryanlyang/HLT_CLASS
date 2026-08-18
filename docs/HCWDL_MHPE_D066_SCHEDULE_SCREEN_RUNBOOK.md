# HCWDL-MHPE D066 Schedule-Screen Queue Runbook

This study submits 60 independent GH200 fits, one aggregate, and one
completion task. It imports exactly one completed `C25P75_300K60` campaign.
The final test remains sealed.

After pushing the implementation commit, run the block below on Tigris. It
creates a clean detached worktree, authenticates the unique completed source,
creates and validates the campaign, writes the canonical dry-run ledger, and
submits the live DAG.

```bash
(
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
export SCREEN_WORKTREE="/home/ryreu/atlas/HLT_Classification_mhpe_d066_screen_${SCREEN_SHORT}"
export SCREEN_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_d066_schedule_screen_${SCREEN_SHORT}_r1"

test ! -e "${SCREEN_WORKTREE}"
test ! -e "${SCREEN_ROOT}"
git -C "${MAIN_REPO}" worktree add --detach "${SCREEN_WORKTREE}" "${SCREEN_COMMIT}"
test -z "$(git -C "${SCREEN_WORKTREE}" status --porcelain)"
export PYTHONPATH="${SCREEN_WORKTREE}/src"

export SOURCE_SPEC="$(
  SCREEN_WORKTREE="${SCREEN_WORKTREE}" MAIN_REPO="${MAIN_REPO}" python -s - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(os.environ["SCREEN_WORKTREE"]) / "src"))
from hlt_classification.scouting.hcwdl_mhpe_schedule_screen import authenticate_source

candidates = []
for path in Path(os.environ["MAIN_REPO"]).glob("checkpoints/hcwdl_mhpe_*/campaign_spec.json"):
    try:
        authenticate_source(path)
        candidates.append(path.resolve())
    except Exception:
        pass
if len(candidates) != 1:
    raise SystemExit(
        f"Expected exactly one completed authenticated C25P75_300K60 source; found {len(candidates)}:\n  "
        + "\n  ".join(map(str, candidates))
    )
print(candidates[0])
PY
)"

python -s "${SCREEN_WORKTREE}/scripts/create_hcwdl_mhpe_schedule_screen.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${SCREEN_ROOT}" \
  --project-dir "${SCREEN_WORKTREE}" \
  --source-commit "${SCREEN_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL MHPE C25P75 D066 20 SCHEDULE SCREEN EXACT SPEC" \
  --operational-evidence-waiver \
  --waiver-phrase "AUTHORIZE HCWDL MHPE C25P75 D066 SCHEDULE SCREEN CARRIED OPERATIONAL EVIDENCE"

python -s "${SCREEN_WORKTREE}/scripts/submit_hcwdl_mhpe_schedule_screen.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --output "${SCREEN_ROOT}/dry_run_submission_ledger.json"

python -s "${SCREEN_WORKTREE}/scripts/submit_hcwdl_mhpe_schedule_screen.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --output "${SCREEN_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL MHPE C25P75 D066 20 SCHEDULE SCREEN EXACT LEDGER"

echo "Source: ${SOURCE_SPEC}"
echo "Screen: ${SCREEN_ROOT}"
squeue --me -o "%.18i %.60j %.2t %.10M %R" | grep -E 'JOBID|hcwsch_' || true
)
```

Monitor with:

```bash
python -s "${SCREEN_WORKTREE}/scripts/monitor_hcwdl_mhpe_schedule_screen.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --submission-ledger "${SCREEN_ROOT}/submission_ledger.json" \
  --output "${SCREEN_ROOT}/monitor.json"
```

Cancel only by the exact bound ledger:

```bash
python -s "${SCREEN_WORKTREE}/scripts/cancel_hcwdl_mhpe_schedule_screen.py" \
  --submission-ledger "${SCREEN_ROOT}/submission_ledger.json" --execute
```

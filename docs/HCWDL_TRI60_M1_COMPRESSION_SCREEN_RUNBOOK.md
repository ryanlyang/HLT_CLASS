# TRI60 M1 compression-screen runbook

This is a standalone full-data study. It does not wait for the source
campaign's aggregate/completion jobs and does not alter any currently running
TRI60 or dense-ladder job.

## Create and inspect

After committing and pushing, set the exact pushed commit on Tigris:

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export SOURCE_SPEC="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"

cd "${MAIN_REPO}"
git fetch origin main
export SCREEN_COMMIT="$(git rev-parse origin/main)"
export SCREEN_SHORT="${SCREEN_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_m1_screen_${SCREEN_SHORT}"
export SCREEN_ROOT="${MAIN_REPO}/checkpoints/hcwdl_tri60_m1_screen_${SCREEN_SHORT}_r1"

test -f "${SOURCE_SPEC}"
test ! -e "${SCREEN_ROOT}"
git worktree add --detach "${PROJECT_DIR}" "${SCREEN_COMMIT}"
export PYTHONPATH="${PROJECT_DIR}/src"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri60_m1_screen_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${SCREEN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SCREEN_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI60 M1 COMPRESSION SCREEN EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_m1_screen_campaign.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --output "${SCREEN_ROOT}/dry_run_submission_ledger.json"

python -s - "${SCREEN_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
spec = json.loads((root / "campaign_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())
dry = json.loads((root / "dry_run_submission_ledger.json").read_text())
assert len(spec["tasks"]) == 23
assert spec["condition_count"] == 20 and spec["fresh_fit_count"] == 19
assert spec["source_campaign_completion_required"] is False
assert spec["source_campaign_scheduler_dependency"] is False
assert spec["source_campaign_outputs_mutated"] is False
assert spec["standalone_smoke_required"] is False
assert spec["scheduler_nice"] == 5000
assert len(plan["commands"]) == len(dry["jobs"]) == 23
assert all("--nice=5000" in row["command"] for row in plan["commands"])
assert not plan["source_scheduler_dependencies"]
print("PASS: exact standalone 20-condition / 23-job dry run")
PY
```

## Live submit

Only after the dry-run audit passes:

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_m1_screen_campaign.py" \
  --spec "${SCREEN_ROOT}/campaign_spec.json" \
  --output "${SCREEN_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI60 M1 COMPRESSION SCREEN EXACT LEDGER"

squeue --me -o "%.18i %.64j %.2t %.10M %R" | grep -E 'JOBID|hcwm1scr_'
```

## Results

```bash
python -s - "${SCREEN_ROOT}/reports/validation_aggregate.json" <<'PY'
import json, math, sys
report = json.load(open(sys.argv[1]))
print(f"{'rank':>4} {'condition':<48} {'AUC':>10} {'R50':>10} {'dAUC M1':>11}")
rows = {row['row_id']: row for row in report['rows']}
for rank, name in enumerate(report['ranked_condition_ids'], 1):
    row = rows[name]
    metrics = row['metrics']
    r50 = math.exp(metrics['macro_mean_log_qcd_rejection_at_50pct_signal'])
    delta = row['delta_from_source_m1']['macro_ovr_auc']
    print(f"{rank:>4} {name:<48} {metrics['macro_ovr_auc']:>10.6f} {r50:>10.1f} {delta:>+11.6f}")
print('Final test accessed:', report['final_test_accessed'])
PY
```

Do not interpret `top_three_diagnostic_ids` as an automatic finalist lock;
the screen deliberately queues no follow-up jobs.

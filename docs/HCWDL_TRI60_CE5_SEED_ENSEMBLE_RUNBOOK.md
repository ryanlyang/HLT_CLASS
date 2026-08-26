# HCWDL TRI60 CE5 seed-ensemble reviewer runbook

This runbook launches the isolated full-data reviewer control defined by
[`HCWDL_TRI60_CE5_SEED_ENSEMBLE_REVIEWER_PLAN.md`](plans/HCWDL_TRI60_CE5_SEED_ENSEMBLE_REVIEWER_PLAN.md).
It creates seven fresh 60-pass exact-HLT fits: five independent CE teachers,
one C10P90/T1 ensemble-distilled student, and its seed-matched CE-only
control. It does not depend on, cancel, hold, reprioritize, or write into any
running TRI60 or dense-extension job.

## Commit and source requirements

Push the implementation commit to `origin/main`. The retained source campaign
spec must remain at:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json
```

The source campaign need not have completed. Its authenticated full-data
foundation, training recipe, and production resource lock are read-only
parents. The source spec itself and those parents must validate.

## Create, inspect, and submit

Run this single block on Tigris after the implementation commit is pushed:

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export CHECKPOINTS="${MAIN_REPO}/checkpoints"
export SOURCE_SPEC="${CHECKPOINTS}/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"

git -C "${MAIN_REPO}" fetch origin main
export CE5_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export CE5_SHORT="${CE5_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_tri60_ce5_${CE5_SHORT}"
export CAMPAIGN_ROOT="${CHECKPOINTS}/hcwdl_tri60_ce5_reviewer_${CE5_SHORT}_r1"

test -f "${SOURCE_SPEC}"
test ! -e "${CAMPAIGN_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${CE5_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach \
    "${PROJECT_DIR}" "${CE5_COMMIT}"
fi

export PYTHONPATH="${PROJECT_DIR}/src"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri60_ce5_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${CE5_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI60 CE5 REVIEWER EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_ce5_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/dry_run_submission_ledger.json"

python -s - "${CAMPAIGN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = json.loads((root / "campaign_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())
dry = json.loads((root / "dry_run_submission_ledger.json").read_text())

assert spec["fresh_fit_count"] == 7
assert spec["teacher_ids"] == [f"CE5_S{i:02d}" for i in range(1, 6)]
assert spec["paired_students"] == ["CE5_KD", "CE5_CONTROL"]
assert len(plan["commands"]) == 12
assert set(dry["jobs"]) == {row["task_id"] for row in plan["commands"]}
assert spec["source_campaign_scheduler_dependency"] is False
assert all(
    not row.get("subject_dependencies")
    for row in plan["commands"]
)
print("CE5 dry-run audit: PASS (7 fits, 1 reducer, 12 exact tasks)")
PY

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_ce5_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI60 CE5 REVIEWER EXACT LEDGER"

python -m json.tool "${CAMPAIGN_ROOT}/submission_ledger.json"
squeue --me -o "%.18i %.54j %.2t %.10M %R" | grep -E 'JOBID|hcwce5_'
```

The first GPU wave contains `CE5_S01` through `CE5_S05` and
`CE5_CONTROL`. `reduce_CE5E` follows only the five teachers. `CE5_KD` follows
the reducer. Aggregate waits for the reducer and both paired students.

## Read results

After `hcwce5_campaign_complete` succeeds:

```bash
export CAMPAIGN_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_tri60_ce5_reviewer_COMMIT8_r1

python -s - "${CAMPAIGN_ROOT}/reports/validation_aggregate.json" <<'PY'
import json
import math
import statistics
import sys

report = json.load(open(sys.argv[1]))
rows = report["rows"]
print(f"{'model':<16} {'kind':<22} {'accuracy':>10} {'AUC':>10} {'R50':>10}")
for row in rows:
    metrics = row["metrics"]
    r50 = math.exp(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])
    print(
        f"{row['row_id']:<16} {row['kind']:<22} "
        f"{metrics['accuracy']:>10.6f} {metrics['macro_ovr_auc']:>10.6f} "
        f"{r50:>10.1f}"
    )

print("\nPrimary paired deltas (CE5_KD - CE5_CONTROL):")
for name, value in report["comparisons"]["primary_CE5_KD_minus_CE5_CONTROL"].items():
    print(f"  {name}: {value:+.6f}")
print("\nTeacher AUC mean/sample std:")
summary = report["teacher_summary"]
print(f"  {summary['macro_ovr_auc_mean']:.6f} / {summary['macro_ovr_auc_sample_std']:.6f}")
print(f"Final test accessed: {report['final_test_accessed']}")
PY
```

## Failure recovery

Recovery is restart-from-zero for only the failed/downstream closure. First
cancel exact downstream IDs from this campaign ledger, wait for terminal
states, and create an immutable monitor with
`scripts/build_hcwdl_tri60_ce5_monitor.py`. Then create and dry-run a recovery
with `scripts/create_hcwdl_tri60_ce5_recovery.py` and
`scripts/submit_hcwdl_tri60_ce5_recovery.py`. A source-changing repair must
name only allowlisted execution files and use the exact phrase
`AUTHORIZE TRI60 CE5 EXECUTION-ONLY SOURCE REPAIR`. Live recovery submission
uses `SUBMIT TRI60 CE5 RECOVERY EXACT LEDGER`. Never cancel by job-name glob.

No final-test task exists in this graph.

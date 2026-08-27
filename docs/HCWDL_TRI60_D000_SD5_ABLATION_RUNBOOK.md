# HCWDL TRI60 D000 matched-seed diversity ablation runbook

This launches five full-data 60-pass D000 fits in parallel. It changes only
their seed domains, maps them to the five CE5 teacher seeds, and evaluates one
uniform validation-only ensemble. It does not cancel, hold, reprioritize,
depend on, or write into any running campaign.

## Create, audit, and submit

Push the implementation commit, then run this block on Tigris:

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
export CE5_SPEC="${CHECKPOINTS}/hcwdl_tri60_ce5_reviewer_925302de_r1/campaign_spec.json"

git -C "${MAIN_REPO}" fetch origin main
export SD5_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export SD5_SHORT="${SD5_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_tri60_d000_sd5_${SD5_SHORT}"
export CAMPAIGN_ROOT="${CHECKPOINTS}/hcwdl_tri60_d000_sd5_${SD5_SHORT}_r1"

test -f "${SOURCE_SPEC}"
test -f "${CE5_SPEC}"
test ! -e "${CAMPAIGN_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${SD5_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach \
    "${PROJECT_DIR}" "${SD5_COMMIT}"
fi

export PYTHONPATH="${PROJECT_DIR}/src"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri60_d000_sd5_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --ce5-campaign-spec "${CE5_SPEC}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SD5_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI60 D000 SD5 EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_d000_sd5_campaign.py" \
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

expected_teachers = [
    "U000", "LOGIT_U050E", "LOGIT_U100E", "LOGIT_D066E", "LOGIT_D033E",
]
assert spec["fresh_fit_count"] == 5
assert spec["source_teacher_order"] == expected_teachers
assert list(spec["ce5_seed_match"].values()) == [
    "CE5_S01", "CE5_S02", "CE5_S03", "CE5_S04", "CE5_S05",
]
assert spec["passes"] == 60 and spec["batch_size"] == 256
assert (spec["ce_weight"], spec["kd_weight"], spec["temperature"]) == (.25, .75, 2.0)
assert len(plan["commands"]) == 10
assert set(dry["jobs"]) == {row["task_id"] for row in plan["commands"]}
assert spec["source_campaign_scheduler_dependency"] is False
assert spec["ce5_campaign_scheduler_dependency"] is False
assert spec["reducer_publishes_train_probability_bank"] is False
assert spec["reducer_publishes_validation_probability_bank"] is False
assert all(not row.get("subject_dependencies") for row in plan["commands"])
train_rows = [row for row in plan["commands"] if row["task_id"].startswith("train_")]
assert len(train_rows) == 5
assert all(row["dependencies"] == ["preflight"] for row in train_rows)
print("SD5 dry-run audit: PASS (5 parallel fits, validation-only reducer, 10 tasks)")
PY

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_d000_sd5_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI60 D000 SD5 EXACT LEDGER"

python -m json.tool "${CAMPAIGN_ROOT}/submission_ledger.json"
squeue --me -o "%.18i %.64j %.2t %.10M %R" | grep -E 'JOBID|hcwsd5_'
```

The five `hcwsd5_train_...` jobs become eligible together after preflight.
Every job uses nice value 10000 and has only dependencies inside this new
ledger.

## Read results

After `hcwsd5_campaign_complete` succeeds:

```bash
python -s - "${CAMPAIGN_ROOT}/reports/validation_aggregate.json" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1]))
rows = {row["row_id"]: row for row in report["rows"]}
order = [
    "U000", "LOGIT_D000E", "CE5E",
    "SD5_LOGIT_D000_from_U000",
    "SD5_LOGIT_D000_from_U050E",
    "SD5_LOGIT_D000_from_U100E",
    "SD5_LOGIT_D000_from_D066E",
    "SD5_LOGIT_D000_from_D033E",
    "SD5_LOGIT_D000E",
]
print(f"{'model':<34} {'kind':<28} {'accuracy':>10} {'AUC':>10} {'R50':>10}")
for name in order:
    row = rows[name]
    metrics = row["metrics"]
    r50 = math.exp(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])
    print(
        f"{name:<34} {row['kind']:<28} "
        f"{metrics['accuracy']:>10.6f} {metrics['macro_ovr_auc']:>10.6f} {r50:>10.1f}"
    )

print("\nPrimary deltas:")
for comparison, metrics in report["comparisons"].items():
    print(f"  {comparison}")
    for metric, value in metrics.items():
        print(f"    {metric}: {value:+.6f}")
print(f"\nFinal test accessed: {report['final_test_accessed']}")
PY
```

No final-test task or durable reducer target bank exists.

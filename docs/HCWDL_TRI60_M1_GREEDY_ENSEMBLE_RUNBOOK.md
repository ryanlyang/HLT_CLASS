# TRI60 M1 greedy ensemble runbook

This standalone validation diagnostic reads the completed M1 screen but has
no dependency on or write path into any currently running campaign.

## Create, dry-run, and audit

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export SCREEN_SPEC="${MAIN_REPO}/checkpoints/hcwdl_tri60_m1_screen_356e2b94_r1/campaign_spec.json"

cd "${MAIN_REPO}"
git fetch origin main
export GREEDY_COMMIT="$(git rev-parse origin/main)"
export GREEDY_SHORT="${GREEDY_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_m1_greedy_${GREEDY_SHORT}"
export GREEDY_ROOT="${MAIN_REPO}/checkpoints/hcwdl_tri60_m1_greedy_${GREEDY_SHORT}_r1"

test -f "${SCREEN_SPEC}"
test -f "$(dirname "${SCREEN_SPEC}")/reports/campaign_complete.json"
test ! -e "${GREEDY_ROOT}"
git worktree add --detach "${PROJECT_DIR}" "${GREEDY_COMMIT}"
export PYTHONPATH="${PROJECT_DIR}/src"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri60_m1_greedy_ensemble_campaign.py" \
  --screen-campaign-spec "${SCREEN_SPEC}" \
  --campaign-root "${GREEDY_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${GREEDY_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI60 M1 GREEDY ENSEMBLE VALIDATION EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_m1_greedy_ensemble_campaign.py" \
  --spec "${GREEDY_ROOT}/campaign_spec.json" \
  --output "${GREEDY_ROOT}/dry_run_submission_ledger.json"

python -s - "${GREEDY_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
spec = json.loads((root / "campaign_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())
dry = json.loads((root / "dry_run_submission_ledger.json").read_text())
assert len(spec["tasks"]) == len(plan["commands"]) == len(dry["jobs"]) == 8
assert spec["candidate_count"] == 20
assert spec["inference_shard_count"] == 5
assert spec["candidates_per_shard"] == 4
assert spec["maximum_ensemble_size"] == 5
assert spec["objective_count"] == 3
assert spec["fresh_fit_count"] == 0
assert spec["source_campaign_scheduler_dependency"] is False
assert spec["screen_campaign_scheduler_dependency"] is False
assert all("--nice=10000" in row["command"] for row in plan["commands"])
assert not plan["source_scheduler_dependencies"]
print("PASS: isolated five-shard / three-objective greedy ensemble dry run")
PY
```

## Live submission

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_m1_greedy_ensemble_campaign.py" \
  --spec "${GREEDY_ROOT}/campaign_spec.json" \
  --output "${GREEDY_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI60 M1 GREEDY ENSEMBLE VALIDATION EXACT LEDGER"

squeue --me -o "%.18i %.64j %.2t %.10M %R" | grep -E 'JOBID|hcwm1ens_'
```

## Compact results

```bash
python -s - "${GREEDY_ROOT}/reports/greedy_ensemble.json" <<'PY'
import json, math, sys
report = json.load(open(sys.argv[1]))
for objective in report["objective_order"]:
    print("\n" + objective)
    print(f"{'size':>4} {'added':<46} {'AUC':>10} {'R50':>10} members")
    for row in report["paths"][objective]:
        metrics = row["metrics"]
        r50 = math.exp(metrics["macro_mean_log_qcd_rejection_at_50pct_signal"])
        print(
            f"{row['ensemble_size']:>4} {row['added_member']:<46} "
            f"{metrics['macro_ovr_auc']:>10.6f} {r50:>10.1f} "
            + ", ".join(row["members"])
        )
print("\nFinal test accessed:", report["final_test_accessed"])
print("Durable candidate predictions (GiB):", report["durable_candidate_prediction_bytes"] / 1024**3)
PY
```

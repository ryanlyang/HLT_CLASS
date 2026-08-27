# TRI60 D000-from-D033E 180-pass runbook

This standalone job reads the completed original TRI60 `LOGIT_D033E` teacher
bank and does not alter or depend on any queued campaign.

## Create, audit, and submit

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
export D180_COMMIT="$(git rev-parse origin/main)"
export D180_SHORT="${D180_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_d000_long180_${D180_SHORT}"
export D180_ROOT="${MAIN_REPO}/checkpoints/hcwdl_tri60_d000_long180_${D180_SHORT}_r1"

test -f "${SOURCE_SPEC}"
test ! -e "${D180_ROOT}"
if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${D180_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git worktree add --detach "${PROJECT_DIR}" "${D180_COMMIT}"
fi
export PYTHONPATH="${PROJECT_DIR}/src"

bash -n "${PROJECT_DIR}/sbatch/run_hcwdl_tri60_d000_long180.sh"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri60_d000_long180_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${D180_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${D180_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI60 D000 D033E LONG180 EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_d000_long180_campaign.py" \
  --spec "${D180_ROOT}/campaign_spec.json" \
  --output "${D180_ROOT}/dry_run_submission_ledger.json"

python -s - "${D180_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
spec = json.loads((root / "campaign_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())
dry = json.loads((root / "dry_run_submission_ledger.json").read_text())
command = plan["commands"][0]["command"]
assert len(plan["commands"]) == len(dry["jobs"]) == 1
assert spec["node_id"] == "LOGIT_D000_from_D033E_180"
assert spec["passes"] == 180 and spec["batch_size"] == 256
assert spec["ce_weight"] == .25 and spec["kd_weight"] == .75
assert spec["temperature"] == 2.0
assert spec["comparison_passes"] == [60, 120, 180]
assert "--time=3-00:00:00" in command
assert "--cpus-per-task=72" in command and "--mem=320G" in command
assert "--nice=10000" in command
assert not any(item.startswith("--dependency=") for item in command)
print("PASS: isolated exact D000 <- D033E 180-pass dry run")
PY

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri60_d000_long180_campaign.py" \
  --spec "${D180_ROOT}/campaign_spec.json" \
  --output "${D180_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI60 D000 D033E LONG180 EXACT LEDGER"

squeue --me -o "%.18i %.60j %.2t %.10M %R" | grep -E 'JOBID|hcwd180_'
```

## Results

```bash
python -m json.tool "${D180_ROOT}/reports/comparison.json"
```

Live progress is printed after every completed pass:

```bash
LOG="$(find "${D180_ROOT}" -type f -name 'slurm-*.out' -print -quit)"
grep 'HCWDL-TRI60 node=LOGIT_D000_from_D033E_180' "${LOG}" | tail -n 10
```

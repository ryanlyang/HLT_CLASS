# TRI60 Dense Extension Runbook

This queues the independent dense extension while the existing TRI60 campaign
continues unchanged. It submits no command against the source ledger. The
source campaign-complete job ID is used only as one `afterok` parent of the
read-only source gate.

## Create, dry-run, and submit

Run this from a Tigris login shell after the implementation commit is pushed
to `origin/main`:

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
export SOURCE_COMPLETE="$(dirname "${SOURCE_SPEC}")/reports/campaign_complete.json"

git -C "${MAIN_REPO}" fetch origin main
export DENSE_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export DENSE_SHORT="${DENSE_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_mhpe_tri60_dense_${DENSE_SHORT}"
export DENSE_ROOT="${CHECKPOINTS}/hcwdl_mhpe_tri60_dense_${DENSE_SHORT}_r1"

test -f "${SOURCE_SPEC}"
test ! -e "${DENSE_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${DENSE_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach \
    "${PROJECT_DIR}" "${DENSE_COMMIT}"
fi

export PYTHONPATH="${PROJECT_DIR}/src"

SOURCE_JOB_ARGS=()
if [ ! -f "${SOURCE_COMPLETE}" ]; then
  mapfile -t SOURCE_COMPLETE_JOBS < <(
    squeue --me -h -o '%A|%j' |
      awk -F'|' '$2 == "hcwtri60r_campaign_complete" || $2 == "hcwtri60_campaign_complete" {print $1}'
  )
  test "${#SOURCE_COMPLETE_JOBS[@]}" -eq 1 || {
    echo "Expected exactly one live TRI60 campaign-complete job"
    printf '%s\n' "${SOURCE_COMPLETE_JOBS[@]}"
    exit 1
  }
  export SOURCE_COMPLETE_JOB="${SOURCE_COMPLETE_JOBS[0]}"
  SOURCE_JOB_ARGS=(--source-completion-job-id "${SOURCE_COMPLETE_JOB}")
fi

python -s "${PROJECT_DIR}/scripts/create_hcwdl_mhpe_tri60_dense_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  "${SOURCE_JOB_ARGS[@]}" \
  --campaign-root "${DENSE_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${DENSE_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI60 DENSE EXTENSION EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_mhpe_tri60_dense_campaign.py" \
  --spec "${DENSE_ROOT}/campaign_spec.json" \
  --output "${DENSE_ROOT}/dry_run_submission_ledger.json"

python -s - "${DENSE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = json.loads((root / "campaign_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())
assert spec["fresh_fit_count"] == 48
assert spec["reducer_count"] == 15
assert len(spec["tasks"]) == 69
assert plan["source_campaign_commands"] == 0
assert plan["source_campaign_outputs_mutated"] is False
commands = "\n".join(" ".join(row["command"]) for row in plan["commands"])
assert "scancel" not in commands and "scontrol" not in commands
print("Dense extension audit passed: 48 fits, 15 reducers, 69 exact tasks")
PY

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_mhpe_tri60_dense_campaign.py" \
  --spec "${DENSE_ROOT}/campaign_spec.json" \
  --output "${DENSE_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI60 DENSE EXTENSION EXACT LEDGER"

squeue --me -o "%.18i %.58j %.2t %.10M %R" |
  grep -E 'JOBID|hcwtri60x_'
```

If the U-stage imports are not complete or their lineage differs, campaign
creation fails before any job is submitted. If the source campaign is already
complete, the block authenticates it immediately and does not add an external
Slurm dependency.

The first jobs eligible to run are the three LOGIT D083 specialists, two RSET
D075 specialists, and two RREL D075 specialists, plus source-independent skip
specialists. The `source_gate` waits for source completion; reducers that
import lower source specialists wait behind that gate.

## Monitor

```bash
python -s "${PROJECT_DIR}/scripts/monitor_hcwdl_mhpe_tri60_dense_campaign.py" \
  --spec "${DENSE_ROOT}/campaign_spec.json" \
  --ledger "${DENSE_ROOT}/submission_ledger.json" \
  --output "${DENSE_ROOT}/monitor.json"

python -m json.tool "${DENSE_ROOT}/monitor.json"
```

Recovery is restart-from-zero and exact-ledger only. Do not create recovery
while any row is active or unknown. The recovery CLIs are
`create_hcwdl_mhpe_tri60_dense_recovery.py` and
`submit_hcwdl_mhpe_tri60_dense_recovery.py`; use a newly pushed clean worktree
and a new recovery root. They preserve valid completed tasks and clean only
the registered failed task outputs inside `DENSE_ROOT`.


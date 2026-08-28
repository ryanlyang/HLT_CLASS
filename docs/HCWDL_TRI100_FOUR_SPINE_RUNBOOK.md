# HCWDL TRI100 Four-Spine Runbook

This queues the isolated 29-fit full-data study. Every fit is one four-node,
four-GH200 synchronous-DDP allocation while reducers remain single-GPU. It
does not alter current TRI60 or DX jobs.

## 1. Commit and push

From PowerShell, stage only the files listed in the implementation handoff,
commit, push `HEAD:main`, and record the full SHA.

## 2. Create the pinned campaign on Tigris

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
export SP4_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export SP4_SHORT="${SP4_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_tri100_spine4_${SP4_SHORT}"
export CAMPAIGN_ROOT="${CHECKPOINTS}/hcwdl_tri100_spine4_${SP4_SHORT}_r1"

test -f "${SOURCE_SPEC}"
test ! -e "${CAMPAIGN_ROOT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${SP4_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach \
    "${PROJECT_DIR}" "${SP4_COMMIT}"
fi

export PYTHONPATH="${PROJECT_DIR}/src"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri100_spine4_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SP4_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI100 FOUR SPINE EXACT SPEC"
```

Creation fails closed unless the source U000 checkpoint and both U000
probability roles authenticate with complete full-data identity coverage.

## 3. Materialize and inspect the dry run

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri100_spine4_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/dry_run_submission_ledger.json"

python -s - "${CAMPAIGN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = json.loads((root / "campaign_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())

assert spec["fresh_fit_count"] == 29
assert spec["reducer_count"] == 25
assert spec["branch_fit_counts"] == {
    "DIRECT": 1, "COARSE": 5, "DENSE": 8, "ULTRADENSE": 15,
}
assert spec["distributed_execution"] == {
    "kind": "synchronous_data_parallel_v1",
    "backend": "nccl", "world_size": 4, "nodes": 4,
    "ranks_per_node": 1, "global_batch_size": 256,
    "nominal_local_batch_size": 64,
    "partial_batch_policy": "exact_disjoint_row_weighted_v1",
    "validation_policy": "rank_zero_full_canonical_broadcast_v1",
    "publication_policy": "rank_zero_only_v1",
}
assert len(plan["commands"]) == 58
assert not any(row["external_dependencies"] for row in plan["commands"])

commands = {row["task_id"]: row["command"] for row in plan["commands"]}
for task in spec["tasks"]:
    command = commands[task["task_id"]]
    if task["kind"] in {"preflight", "train"}:
        assert "--nodes=4" in command
        assert "--ntasks=4" in command
        assert "--ntasks-per-node=1" in command
        assert any("HCWDL_SPINE4_DDP_WORLD_SIZE=4" in item for item in command)
    else:
        assert "--nodes=1" in command
        assert "--ntasks=1" in command
        assert any("HCWDL_SPINE4_DDP_WORLD_SIZE=1" in item for item in command)

print("Campaign:", root)
print("Spec:", spec["content_hash"])
print("Tasks:", len(plan["commands"]))
print("First branch fits:")
for row in spec["tasks"]:
    if row["kind"] == "train" and row["dependencies"] == ["preflight"]:
        print(" ", row["task_id"])
PY
```

Expected task count: 58 = source authentication + four-node NCCL acceptance
+ 29 four-node fits + 25 single-GPU reducers + aggregate + completion.

## 4. Submit the exact ledger

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri100_spine4_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI100 FOUR SPINE EXACT LEDGER"

squeue --me -o "%.18i %.58j %.2t %.10M %R" | grep -E 'JOBID|hcwsp4_'
```

The submitter is journaled and idempotent.  Rerunning it after a connection
loss resumes publication from the exact immutable submission events.

The four branch heads cannot start until `hcwsp4_preflight` publishes the
four-node NCCL acceptance lock. That job performs a real DDP backward pass,
checks one visible GH200 on each of four distinct nodes, and proves collective
rank topology. A failed acceptance job therefore prevents scientific fits
from starting; it is not a weak-metric veto.

## 5. Monitor

```bash
python -s "${PROJECT_DIR}/scripts/monitor_hcwdl_tri100_spine4_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --ledger "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --output "${CAMPAIGN_ROOT}/monitor.json"
```

Do not create recovery while any exact ledger job is active.  Recovery uses
the scripts documented by `--help` and the phrase
`SUBMIT HCWDL TRI100 FOUR SPINE RECOVERY EXACT LEDGER`.

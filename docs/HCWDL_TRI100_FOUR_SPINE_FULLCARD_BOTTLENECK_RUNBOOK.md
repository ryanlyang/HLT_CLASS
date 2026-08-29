# HCWDL TRI100 Full-Cardinality Bottleneck Four-Spine Runbook

This runbook queues the controlled pairing experiment specified by
[`HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_BOTTLENECK_MATCHING_IMPLEMENTATION_PLAN.md`](plans/HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_BOTTLENECK_MATCHING_IMPLEMENTATION_PLAN.md).

The execution has two isolated stages:

1. build and authenticate the full-cardinality assignment foundation, rebuild
   every assignment-dependent descendant, and prove all-row U000 equivalence;
2. create and submit the 29-fit, 25-reducer, single-GH200 four-spine campaign.

The second stage cannot be created before the first publishes
`locks/foundation.json`. Neither stage depends on, writes into, holds,
cancels, or reprioritizes the established four-spine jobs. There is no
standalone smoke campaign. Real-row matcher acceptance and a production-model
GH200 forward/backward check are registered gates in these DAGs.

## 1. Commit and push locally

From PowerShell in the main repository, inspect the exact staged set before
committing. Do not add unrelated worktrees, figures, paper files, or temporary
pytest outputs.

```powershell
git status --short --untracked-files=all
git diff --check
git add -- `
  docs/plans/HCWDL_TRI100_FOUR_SPINE_FULL_CARDINALITY_BOTTLENECK_MATCHING_IMPLEMENTATION_PLAN.md `
  docs/HCWDL_TRI100_FOUR_SPINE_FULLCARD_BOTTLENECK_RUNBOOK.md `
  sbatch/run_hcwdl_fullcard_bottleneck_foundation_task.sh `
  sbatch/run_hcwdl_tri100_spine4_bottleneck_task.sh `
  sbatch/run_hcwdl_tri100_spine4_bottleneck_recovery_task.sh `
  scripts/create_hcwdl_fullcard_bottleneck_foundation.py `
  scripts/create_hcwdl_tri100_spine4_bottleneck_campaign.py `
  scripts/create_hcwdl_tri100_spine4_bottleneck_recovery.py `
  scripts/monitor_hcwdl_tri100_spine4_bottleneck_campaign.py `
  scripts/run_hcwdl_fullcard_bottleneck_foundation_task.py `
  scripts/run_hcwdl_tri100_spine4_bottleneck_task.py `
  scripts/run_hcwdl_tri100_spine4_bottleneck_recovery_task.py `
  scripts/submit_hcwdl_fullcard_bottleneck_foundation.py `
  scripts/submit_hcwdl_tri100_spine4_bottleneck_campaign.py `
  scripts/submit_hcwdl_tri100_spine4_bottleneck_recovery.py `
  src/hlt_classification/scouting/hcwdl_assignment_store.py `
  src/hlt_classification/scouting/hcwdl_bottleneck_operations.py `
  src/hlt_classification/scouting/hcwdl_exact_dag_submission.py `
  src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_cache.py `
  src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_contracts.py `
  src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_diagnostics.py `
  src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_foundation.py `
  src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_foundation_campaign.py `
  src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_foundation_workflow.py `
  src/hlt_classification/scouting/hcwdl_fullcard_bottleneck_matcher.py `
  src/hlt_classification/scouting/hcwdl_homotopy.py `
  src/hlt_classification/scouting/hcwdl_homotopy_stream.py `
  src/hlt_classification/scouting/hcwdl_unified_balanced_builder.py `
  src/hlt_classification/scouting/hcwdl_unified_balanced_runner.py `
  src/hlt_classification/scouting/hcwdl_upper_builder.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_campaign.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_contracts.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_execution.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_graph.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_probability.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_recovery.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_runner.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_source.py `
  src/hlt_classification/scouting/hcwdl_tri100_spine4_bottleneck_workflow.py `
  tests/test_hcwdl_fullcard_bottleneck.py `
  tests/test_hcwdl_tri100_spine4_bottleneck.py
# These three files contain other in-progress documentation. Select only the
# full-cardinality bottleneck hunks; do not stage unrelated sections.
git add -p -- README.md docs/plans/README.md docs/HANDOFF.md
git diff --cached --check
git diff --cached --stat
git commit -m "Add full-cardinality bottleneck four-spine campaign"
git push origin HEAD:main
git rev-parse HEAD
```

## 2. Pin the clean Tigris worktree and source artifacts

Log into Tigris, then run:

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export CHECKPOINTS="${MAIN_REPO}/checkpoints"

git -C "${MAIN_REPO}" fetch origin main
export SP4B_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export SP4B_SHORT="${SP4B_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_spine4b_fullcard_${SP4B_SHORT}"

if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${SP4B_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach \
    "${PROJECT_DIR}" "${SP4B_COMMIT}"
fi

export PYTHONPATH="${PROJECT_DIR}/src"

# This is the completed TRI60 source that owns the shared read-only U000.
export SOURCE_SPEC="${CHECKPOINTS}/hcwdl_mhpe_tri60_full_d218961c_r1/campaign_spec.json"

# Set these two to the actual established single-GPU four-spine campaign and
# CE60 baseline on this filesystem. They are comparison artifacts only and do
# not become scheduler dependencies.
export ESTABLISHED_SPEC="/absolute/path/to/established_four_spine/campaign_spec.json"
export M0CE60_REPORT="/absolute/path/to/M0CE60/training_report.json"

test -f "${SOURCE_SPEC}"
test -f "${ESTABLISHED_SPEC}"
test -f "${M0CE60_REPORT}"
```

To locate the latter two without guessing:

```bash
find "${CHECKPOINTS}" -path '*/campaign_spec.json' \
  -path '*tri100*spine4*' -print
find "${CHECKPOINTS}" -path '*/training/M0CE60/training_report.json' -print
```

Choose the established single-GH200 four-spine spec, not the cancelled
four-node DDP attempt. Missing established fit reports are allowed and appear
as `pending`; they never block this campaign.

## 3. Local and pinned-source checks

```bash
cd "${PROJECT_DIR}"
python -m pytest -q -p no:cacheprovider \
  tests/test_hcwdl_fullcard_bottleneck.py \
  tests/test_hcwdl_tri100_spine4_bottleneck.py
bash -n sbatch/run_hcwdl_fullcard_bottleneck_foundation_task.sh
bash -n sbatch/run_hcwdl_tri100_spine4_bottleneck_task.sh
bash -n sbatch/run_hcwdl_tri100_spine4_bottleneck_recovery_task.sh
test -z "$(git status --porcelain)"
```

## 4. Create and dry-run the isolated foundation

```bash
export FOUNDATION_ROOT="${CHECKPOINTS}/hcwdl_spine4b_fullcard_foundation_${SP4B_SHORT}_r1"
test ! -e "${FOUNDATION_ROOT}"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_fullcard_bottleneck_foundation.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --foundation-root "${FOUNDATION_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SP4B_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI100 FOUR SPINE BOTTLENECK FOUNDATION EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_fullcard_bottleneck_foundation.py" \
  --spec "${FOUNDATION_ROOT}/foundation_spec.json" \
  --output "${FOUNDATION_ROOT}/dry_run_submission_ledger.json"

python -s - "${FOUNDATION_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = json.loads((root / "foundation_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())
assert len(spec["tasks"]) == 19
assert len(plan["commands"]) == 19
assert spec["ordinary_final_test_capability"] is False
assert spec["u000_retrained"] is False
assert spec["pairing_provenance"] == "validity_only_not_correspondence_confidence"
assert spec["existing_campaign_dependencies"] == []
assert all("final_test" not in row["task_id"] for row in spec["tasks"])
print("FOUNDATION DRY-RUN AUDIT: PASS")
PY
```

## 5. Submit the foundation

This is the first live mutation. Run it only after inspecting the dry ledger.

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_fullcard_bottleneck_foundation.py" \
  --spec "${FOUNDATION_ROOT}/foundation_spec.json" \
  --output "${FOUNDATION_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI100 FOUR SPINE BOTTLENECK FOUNDATION EXACT LEDGER"

squeue --me -o "%.18i %.56j %.2t %.10M %R" | grep -E 'JOBID|hcwsp4b_f' || true
```

The foundation performs, in order: source authentication, a genuine real-row
production/reference matcher check, complete train/validation assignments,
exact sampled recomputation, diagnostic reduction, scale/coupling/balanced
descendant rebuilds, an all-row old/new P0 equivalence scan, and the final
foundation lock. Assignment jobs use bounded process parallelism and never
persist dense edge matrices.

Wait for `foundation_lock` to complete, then audit:

```bash
test -f "${FOUNDATION_ROOT}/locks/foundation.json"
python -m json.tool "${FOUNDATION_ROOT}/locks/foundation.json"

python -s - "${FOUNDATION_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
lock = json.loads((root / "locks/foundation.json").read_text())
equiv = json.loads((root / "locks/u000_equivalence.json").read_text())
assert lock["u000_reused_read_only"] is True
assert lock["assignment_dependent_descendants_rebuilt"] is True
assert lock["pairing_provenance"] == "validity_only_not_correspondence_confidence"
assert lock["durable_dense_pair_matrices"] is False
assert lock["rolling_resume_persisted"] is False
assert equiv["identical_p0_tensors_all_rows"] is True
assert equiv["identical_labels_and_identity_order"] is True
print("FOUNDATION COMPLETION AUDIT: PASS")
PY
```

## 6. Create and dry-run the science campaign

```bash
export CAMPAIGN_ROOT="${CHECKPOINTS}/hcwdl_tri100_spine4b_fullcard_${SP4B_SHORT}_r1"
test ! -e "${CAMPAIGN_ROOT}"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri100_spine4_bottleneck_campaign.py" \
  --foundation-spec "${FOUNDATION_ROOT}/foundation_spec.json" \
  --established-campaign-spec "${ESTABLISHED_SPEC}" \
  --m0ce60-report "${M0CE60_REPORT}" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SP4B_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL TRI100 FOUR SPINE BOTTLENECK EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri100_spine4_bottleneck_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/dry_run_submission_ledger.json"

python -s - "${CAMPAIGN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
spec = json.loads((root / "campaign_spec.json").read_text())
plan = json.loads((root / "command_plan.json").read_text())
assert len(spec["tasks"]) == 58
assert len(plan["commands"]) == 58
assert spec["fresh_fit_count"] == 29
assert spec["reducer_count"] == 25
assert spec["execution"]["world_size"] == 1
assert spec["ensembles"] is False
assert spec["rolling_resume"] is False
assert spec["existing_campaign_dependencies"] == []
assert all("final_test" not in row["task_id"] for row in spec["tasks"])
print("SCIENCE DRY-RUN AUDIT: PASS")
PY
```

## 7. Submit the science campaign

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri100_spine4_bottleneck_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI100 FOUR SPINE BOTTLENECK EXACT LEDGER"

squeue --me -o "%.18i %.58j %.2t %.10M %R" | grep -E 'JOBID|hcwsp4b_' || true
```

The four branch-head fits depend only on this campaign's own preflight. They
can run concurrently. Every later fit depends only on the preceding
single-component probability reducer in the same branch.

## 8. Monitor and recover exact IDs

```bash
python -s "${PROJECT_DIR}/scripts/monitor_hcwdl_tri100_spine4_bottleneck_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --ledger "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --output "${CAMPAIGN_ROOT}/monitor.json"
python -m json.tool "${CAMPAIGN_ROOT}/monitor.json"
```

Only after every registered job is terminal, create a source-pinned
restart-from-zero recovery if the monitor contains retryable failures:

```bash
export RECOVERY_ROOT="${CHECKPOINTS}/hcwdl_tri100_spine4b_fullcard_recovery_${SP4B_SHORT}_r1"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_tri100_spine4_bottleneck_recovery.py" \
  --subject-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --subject-ledger "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --monitor-report "${CAMPAIGN_ROOT}/monitor.json" \
  --recovery-root "${RECOVERY_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${SP4B_COMMIT}"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri100_spine4_bottleneck_recovery.py" \
  --recovery "${RECOVERY_ROOT}/recovery_spec.json" \
  --output "${RECOVERY_ROOT}/dry_run_submission_ledger.json"

# Inspect the exact retry set before this live command.
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_tri100_spine4_bottleneck_recovery.py" \
  --recovery "${RECOVERY_ROOT}/recovery_spec.json" \
  --output "${RECOVERY_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL TRI100 FOUR SPINE BOTTLENECK RECOVERY EXACT LEDGER"
```

Never cancel by job-name patterns. Use only exact IDs from the corresponding
immutable submission ledger.

## 9. Completion and storage checks

```bash
test -f "${CAMPAIGN_ROOT}/reports/validation_aggregate.json"
test -f "${CAMPAIGN_ROOT}/reports/campaign_complete.json"
python -m json.tool "${CAMPAIGN_ROOT}/reports/campaign_complete.json"
du -sh "${FOUNDATION_ROOT}" "${CAMPAIGN_ROOT}"
find "${FOUNDATION_ROOT}" "${CAMPAIGN_ROOT}" \
  -type f \( -iname '*resume*' -o -iname '*optimizer*' \) -print
```

The last `find` must print nothing. Poor accuracy, AUC, R50, recovery, or
matching tails are reportable results and do not control completion. Final
test remains inaccessible throughout.

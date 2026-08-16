# HCWDL-MHPE-FULL Tigris Runbook

This runbook creates one clean detached worktree, authenticates the completed
all-mapped FULL3 foundation, writes the explicit no-new-smoke waiver, performs
a complete nonmutating Slurm dry run, and only then permits a separately
authorized live submission. It never submits final-test evaluation.

After pushing the implementation, create a clean detached worktree. The block
authenticates and selects exactly one completed all-mapped FULL3 foundation;
it stops if zero or multiple eligible roots exist.

```bash
set -euo pipefail
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
git -C "${MAIN_REPO}" fetch origin main
export MHPE_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export MHPE_SHORT="${MHPE_COMMIT:0:8}"
export MHPE_WORKTREE="/home/ryreu/atlas/HLT_Classification_hcwdl_mhpe_${MHPE_SHORT}"
export MHPE_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_full_${MHPE_SHORT}_r1"

test ! -e "${MHPE_WORKTREE}"
test ! -e "${MHPE_ROOT}"
git -C "${MAIN_REPO}" worktree add --detach "${MHPE_WORKTREE}" "${MHPE_COMMIT}"
export PYTHONPATH="${MHPE_WORKTREE}/src"

export FOUNDATION_LOCK="$(
python -s - "${MAIN_REPO}/checkpoints" <<'PY'
from pathlib import Path
import sys

from hlt_classification.data.cache_contracts import load_json
from hlt_classification.scouting.hcwdl_unified_balanced_full_campaign import (
    validate_foundation_campaign,
)
from hlt_classification.scouting.hcwdl_unified_balanced_full_contracts import (
    validate_foundation_lock,
)

candidates = []
for lock_path in Path(sys.argv[1]).rglob("locks/foundation.json"):
    try:
        lock = load_json(lock_path)
        validate_foundation_lock(lock)
        root = lock_path.parent.parent
        spec = load_json(root / "foundation_spec.json")
        spec_hash = validate_foundation_campaign(
            spec, executable=False, verify_source_tree=False,
        )
        if (
            spec["mode"] == "all_mapped_full3"
            and lock["foundation_spec_sha256"] == spec_hash
            and (root / "training/U000/training_report.json").is_file()
            and (root / "training/M0paired/training_report.json").is_file()
            and (root / "targets/u000_train/manifest.json").is_file()
        ):
            candidates.append(lock_path.resolve())
    except Exception:
        pass
if len(candidates) != 1:
    raise SystemExit(
        "Expected exactly one authenticated completed FULL3 foundation; found "
        f"{len(candidates)}:\n" + "\n".join(map(str, candidates))
    )
print(candidates[0])
PY
)"
echo "Foundation lock: ${FOUNDATION_LOCK}"

python -s "${MHPE_WORKTREE}/scripts/create_hcwdl_mhpe_campaign.py" \
  --foundation-lock "${FOUNDATION_LOCK}" \
  --campaign-root "${MHPE_ROOT}" \
  --project-dir "${MHPE_WORKTREE}" \
  --source-commit "${MHPE_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL MHPE FULL EXACT SPEC"

python -s "${MHPE_WORKTREE}/scripts/submit_hcwdl_mhpe_campaign.py" \
  --spec "${MHPE_ROOT}/campaign_spec.json" \
  --output "${MHPE_ROOT}/dry_run_submission_ledger.json"
```

Inspect the immutable graph and dry-run ledger before live submission. Live
submission requires the user's explicit authorization and this exact command:

```bash
python -s "${MHPE_WORKTREE}/scripts/submit_hcwdl_mhpe_campaign.py" \
  --spec "${MHPE_ROOT}/campaign_spec.json" \
  --output "${MHPE_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL MHPE FULL EXACT LEDGER"
```

Monitor or print exact cancellation IDs with:

```bash
python -s "${MHPE_WORKTREE}/scripts/monitor_hcwdl_mhpe_campaign.py" \
  --spec "${MHPE_ROOT}/campaign_spec.json" \
  --submission-ledger "${MHPE_ROOT}/submission_ledger.json" \
  --output "${MHPE_ROOT}/monitor.json"

python -s "${MHPE_WORKTREE}/scripts/cancel_hcwdl_mhpe_campaign.py" \
  --submission-ledger "${MHPE_ROOT}/submission_ledger.json"
```

Cancellation is preview-only unless `--execute` is supplied with
`CANCEL HCWDL MHPE EXACT IDS`. Recovery is created with
`create_hcwdl_mhpe_recovery.py` and submitted with
`submit_hcwdl_mhpe_recovery.py`; it must name exact failed tasks and preserves
completed outputs. A recovery dry run must be written to
`RECOVERY_ROOT/dry_run_submission_ledger.json` before live recovery. Repeated
recovery uses the previous recovery spec, ledger, and recovery monitor as its
new immutable subject.

The ordinary campaign never submits final-test work. After completion, the
human may separately create the execution lock with
`AUTHORIZE HCWDL MHPE SEALED FINAL TEST`; `submit_hcwdl_mhpe_final.py` first
prints the single exact-HLT final command and submits it only with the second
phrase `SUBMIT HCWDL MHPE SEALED FINAL TEST EXACT JOB`.

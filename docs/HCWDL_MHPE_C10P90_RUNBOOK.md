# HCWDL-MHPE C10P90 Parallel Tigris Runbook

This runbook creates and submits the full-data C10P90 companion without
touching the already running C25P75 campaign. It reuses only the authenticated
read-only FULL3 foundation. It creates a distinct detached worktree, campaign
root, immutable specification, dry-run ledger, live ledger, and `hcwmhpe90_`
Slurm jobs. It does not submit final-test evaluation.

Run this block after pushing the exact implementation commit:

```bash
set -euo pipefail
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
git -C "${MAIN_REPO}" fetch origin main
export MHPE90_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export MHPE90_SHORT="${MHPE90_COMMIT:0:8}"
export MHPE90_WORKTREE="/home/ryreu/atlas/HLT_Classification_hcwdl_mhpe90_${MHPE90_SHORT}"
export MHPE90_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_c10p90_full_${MHPE90_SHORT}_r1"

test ! -e "${MHPE90_WORKTREE}"
test ! -e "${MHPE90_ROOT}"
git -C "${MAIN_REPO}" worktree add --detach \
  "${MHPE90_WORKTREE}" "${MHPE90_COMMIT}"
test -z "$(git -C "${MHPE90_WORKTREE}" status --porcelain)"
export PYTHONPATH="${MHPE90_WORKTREE}/src"

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

python -s "${MHPE90_WORKTREE}/scripts/create_hcwdl_mhpe_campaign.py" \
  --foundation-lock "${FOUNDATION_LOCK}" \
  --campaign-root "${MHPE90_ROOT}" \
  --project-dir "${MHPE90_WORKTREE}" \
  --source-commit "${MHPE90_COMMIT}" \
  --recipe-profile C10P90 \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL MHPE C10P90 FULL EXACT SPEC"

python -s "${MHPE90_WORKTREE}/scripts/submit_hcwdl_mhpe_campaign.py" \
  --spec "${MHPE90_ROOT}/campaign_spec.json" \
  --output "${MHPE90_ROOT}/dry_run_submission_ledger.json"

python -s "${MHPE90_WORKTREE}/scripts/submit_hcwdl_mhpe_campaign.py" \
  --spec "${MHPE90_ROOT}/campaign_spec.json" \
  --output "${MHPE90_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL MHPE C10P90 FULL EXACT LEDGER"

echo "C10P90 root: ${MHPE90_ROOT}"
squeue --me -o "%.18i %.55j %.2t %.10M %R" | grep -E 'JOBID|hcwmhpe90_'
```

The block fails before mutation if the worktree/root already exists, the
pushed source is dirty or ambiguous, or the FULL3 foundation cannot be
authenticated uniquely. The live submit requires the canonical dry-run
ledger created immediately above it.

Monitor or preview exact cancellation IDs with:

```bash
python -s "${MHPE90_WORKTREE}/scripts/monitor_hcwdl_mhpe_campaign.py" \
  --spec "${MHPE90_ROOT}/campaign_spec.json" \
  --submission-ledger "${MHPE90_ROOT}/submission_ledger.json" \
  --output "${MHPE90_ROOT}/monitor.json"

python -s "${MHPE90_WORKTREE}/scripts/cancel_hcwdl_mhpe_campaign.py" \
  --submission-ledger "${MHPE90_ROOT}/submission_ledger.json"
```

Cancellation remains preview-only unless explicitly executed with the exact
campaign-bound IDs and the existing cancellation authorization phrase.

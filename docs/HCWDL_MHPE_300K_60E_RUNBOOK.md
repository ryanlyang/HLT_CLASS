# HCWDL-MHPE paired 300k/60-pass queue runbook

This runbook creates one clean detached worktree, authenticates exactly one
completed 300k unified-balanced foundation, creates two independent campaign
roots, renders both dry-run ledgers, and then submits both live DAGs. It does
not touch any full-data MHPE root.

Run the exact block supplied at handoff only after the implementation commit is
pushed. The expected live prefixes are:

```text
hcwmhpe25p_   C25P75_300K60
hcwmhpe90p_   C10P90_300K60
```

Each campaign has 23 jobs and its own `submission_ledger.json`. The first model
job in each is `train_U050_from_U000`; U000 is imported rather than retrained.
Use the ledger-aware monitor and cancellation CLIs. Never cancel by prefix.

After pushing, run:

```bash
(
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
cd "${MAIN_REPO}"
git fetch origin main
export MHPE_COMMIT="$(git rev-parse origin/main)"
export MHPE_SHORT="${MHPE_COMMIT:0:8}"
export MHPE_WORKTREE="/home/ryreu/atlas/HLT_Classification_mhpe_300k60_${MHPE_SHORT}"
export C25_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_c25p75_300k60_${MHPE_SHORT}_r1"
export C10_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_c10p90_300k60_${MHPE_SHORT}_r1"

test ! -e "${MHPE_WORKTREE}"
test ! -e "${C25_ROOT}"
test ! -e "${C10_ROOT}"
git -C "${MAIN_REPO}" worktree add --detach "${MHPE_WORKTREE}" "${MHPE_COMMIT}"
test -z "$(git -C "${MHPE_WORKTREE}" status --porcelain)"
export PYTHONPATH="${MHPE_WORKTREE}/src"

export FOUNDATION_LOCK="$(
  MHPE_WORKTREE="${MHPE_WORKTREE}" MAIN_REPO="${MAIN_REPO}" python -s - <<'PY'
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(os.environ["MHPE_WORKTREE"]) / "src"))
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.scouting.hcwdl_unified_balanced_contracts import (
    validate_foundation_lock,
    validate_foundation_spec,
)

candidates = []
for lock_path in Path(os.environ["MAIN_REPO"]).glob(
    "checkpoints/hcwdl_ub_foundation_*/locks/foundation.json"
):
    try:
        lock = load_json(lock_path)
        lock_hash = validate_foundation_lock(lock)
        root = lock_path.parent.parent
        spec = load_json(root / "foundation_spec.json")
        spec_hash = validate_foundation_spec(spec)
        if (
            lock["foundation_spec_sha256"] == spec_hash
            and spec["role_counts"] == {
                "train": 300_000,
                "validation": 100_000,
                "final_test": 100_000,
            }
            and (root / "training/U000/training_report.json").is_file()
            and (root / "training/M0paired/training_report.json").is_file()
            and (root / "targets/u000_train/manifest.json").is_file()
        ):
            candidates.append(lock_path.resolve())
    except Exception:
        pass
if len(candidates) != 1:
    raise SystemExit(
        "Expected exactly one completed authenticated 300k UB foundation; found "
        + str(len(candidates)) + ":\n  "
        + "\n  ".join(map(str, candidates))
    )
print(candidates[0])
PY
)"
echo "Foundation: ${FOUNDATION_LOCK}"

python -s "${MHPE_WORKTREE}/scripts/create_hcwdl_mhpe_campaign.py" \
  --foundation-lock "${FOUNDATION_LOCK}" \
  --campaign-root "${C25_ROOT}" \
  --project-dir "${MHPE_WORKTREE}" \
  --source-commit "${MHPE_COMMIT}" \
  --recipe-profile C25P75_300K60 \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL MHPE C25P75 300K60 EXACT SPEC"

python -s "${MHPE_WORKTREE}/scripts/create_hcwdl_mhpe_campaign.py" \
  --foundation-lock "${FOUNDATION_LOCK}" \
  --campaign-root "${C10_ROOT}" \
  --project-dir "${MHPE_WORKTREE}" \
  --source-commit "${MHPE_COMMIT}" \
  --recipe-profile C10P90_300K60 \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL MHPE C10P90 300K60 EXACT SPEC"

for ROOT in "${C25_ROOT}" "${C10_ROOT}"; do
  python -s "${MHPE_WORKTREE}/scripts/submit_hcwdl_mhpe_campaign.py" \
    --spec "${ROOT}/campaign_spec.json" \
    --output "${ROOT}/dry_run_submission_ledger.json"
done

python -s "${MHPE_WORKTREE}/scripts/submit_hcwdl_mhpe_campaign.py" \
  --spec "${C25_ROOT}/campaign_spec.json" \
  --output "${C25_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL MHPE C25P75 300K60 EXACT LEDGER"

python -s "${MHPE_WORKTREE}/scripts/submit_hcwdl_mhpe_campaign.py" \
  --spec "${C10_ROOT}/campaign_spec.json" \
  --output "${C10_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL MHPE C10P90 300K60 EXACT LEDGER"

echo "C25 root: ${C25_ROOT}"
echo "C10 root: ${C10_ROOT}"
squeue --me -o "%.18i %.60j %.2t %.10M %R" | grep -E 'hcwmhpe(25p|90p)_' || true
)
```

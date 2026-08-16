# HCWDL Unified-Balanced Full-Data Coarse Runbook

This runbook creates, dry-runs, and submits the three independent coarse
full-data arms. It reuses a completed authenticated FULL3 foundation; it does
not submit another foundation or smoke.

## Tigris preparation

After pushing the implementation commit:

```bash
set -euo pipefail
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
git -C "${MAIN_REPO}" fetch origin main
export COARSE_COMMIT="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export COARSE_SHORT="${COARSE_COMMIT:0:8}"
export COARSE_WORKTREE="/home/ryreu/atlas/HLT_Classification_hcwdl_ub_coarse_${COARSE_SHORT}"
export COARSE_ROOT="${MAIN_REPO}/checkpoints/hcwdl_ub_fullcoarse3_${COARSE_SHORT}_r1"
export COARSE_LEDGERS="${MAIN_REPO}/checkpoints/hcwdl_ub_fullcoarse3_ledgers_${COARSE_SHORT}_r1"
export FOUNDATION_ROOT="${MAIN_REPO}/checkpoints/hcwdl_ub_full3_foundation_8403b1a1_r1"

test -f "${FOUNDATION_ROOT}/locks/foundation.json"
test ! -e "${COARSE_WORKTREE}"
test ! -e "${COARSE_ROOT}"
git -C "${MAIN_REPO}" worktree add --detach \
  "${COARSE_WORKTREE}" "${COARSE_COMMIT}"
test -z "$(git -C "${COARSE_WORKTREE}" status --porcelain)"
export PYTHONPATH="${COARSE_WORKTREE}/src"
```

If the canonical completed foundation root differs, change only
`FOUNDATION_ROOT`; its lock is fully authenticated by the creator.

## Create and dry-run

```bash
python -s "${COARSE_WORKTREE}/scripts/create_hcwdl_unified_balanced_coarse_arms.py" \
  --foundation-lock "${FOUNDATION_ROOT}/locks/foundation.json" \
  --arms-root "${COARSE_ROOT}" \
  --project-dir "${COARSE_WORKTREE}" \
  --source-commit "${COARSE_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL UB FULLCOARSE3 THREE ARMS EXACT SPECS"

python -s "${COARSE_WORKTREE}/scripts/submit_hcwdl_unified_balanced_coarse_arms.py" \
  --arms-root "${COARSE_ROOT}" \
  --output-root "${COARSE_LEDGERS}/dry"
```

The dry run creates no Slurm jobs. Inspect the three specs and ledgers before
the one live mutation command.

## Submit all three arms

```bash
python -s "${COARSE_WORKTREE}/scripts/submit_hcwdl_unified_balanced_coarse_arms.py" \
  --arms-root "${COARSE_ROOT}" \
  --output-root "${COARSE_LEDGERS}/live" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL UB FULLCOARSE3 THREE ARMS PARALLEL EXACT LEDGERS"

squeue --me -o "%.18i %.70j %.2t %.10M %R"
```

This submits 14 jobs per arm: 12 training fits, one aggregate, and one
completion task. The two path roots and all three arms may run concurrently;
descendants stay dependency-bound. No final-test job is present.

Use `monitor_hcwdl_unified_balanced_coarse.py` on each canonical arm spec and
live ledger. Preview exact cancellation IDs with
`cancel_hcwdl_unified_balanced_coarse.py`. If a task fails, publish the monitor
report, create the exact closure with
`create_hcwdl_unified_balanced_coarse_recovery.py`, dry-run and then submit it
with `submit_hcwdl_unified_balanced_coarse_recovery.py` and phrase
`SUBMIT HCWDL UB FULLCOARSE3 RECOVERY EXACT CLOSURE`.


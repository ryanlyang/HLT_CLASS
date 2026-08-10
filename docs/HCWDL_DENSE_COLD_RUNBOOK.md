# HCWDL Dense Cold 300k Runbook

This supplemental campaign reuses the completed unweighted 300k pilot. It does
not rerun matching, qualification, M0, label-only D100, or TOFF, and it never
reads final test. Its exact semantics are in
[`contracts/HCWDL_DENSE_COLD_PILOT.md`](contracts/HCWDL_DENSE_COLD_PILOT.md).

## Create an isolated source worktree

Run after the implementation commit is pushed to `origin/main`:

```bash
cd /home/ryreu/atlas/HLT_Classification
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

git fetch origin main
export DENSE_COMMIT="$(git rev-parse origin/main)"
export DENSE_SHORT="${DENSE_COMMIT:0:8}"
export DENSE_WORKTREE="/home/ryreu/atlas/HLT_Classification_hcwdl_densecold300k_${DENSE_SHORT}"
export DENSE_ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_densecold300k_${DENSE_SHORT}_r1"
export PARENT_ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_pilot_b3154d67_unweighted_auto_r1"

test -f "${PARENT_ROOT}/campaign_spec.json"
test ! -e "${DENSE_WORKTREE}"
test ! -e "${DENSE_ROOT}"
git worktree add --detach "${DENSE_WORKTREE}" "${DENSE_COMMIT}"
```

## Create, dry-run, and submit

```bash
python -s "${DENSE_WORKTREE}/scripts/create_hcwdl_dense_pilot.py" \
  --parent-campaign-spec "${PARENT_ROOT}/campaign_spec.json" \
  --campaign-root "${DENSE_ROOT}" \
  --project-dir "${DENSE_WORKTREE}" \
  --source-commit "${DENSE_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL DENSE COLD 300K EXACT SPEC" \
  --output "${DENSE_ROOT}/campaign_spec.json"

python -s "${DENSE_WORKTREE}/scripts/submit_hcwdl_dense_pilot.py" \
  --campaign-spec "${DENSE_ROOT}/campaign_spec.json" \
  --output "${DENSE_ROOT}/dry_run_ledger.json"

python -s "${DENSE_WORKTREE}/scripts/submit_hcwdl_dense_pilot.py" \
  --campaign-spec "${DENSE_ROOT}/campaign_spec.json" \
  --output "${DENSE_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL DENSE COLD 300K EXACT SPEC"
```

The resulting DAG contains 12 sequential GPU jobs and one CPU aggregate. Each
job is checkpointable and receives the measured parent pilot's `gpu_single`
resource request. The ledger contains the exact IDs for monitoring or exact
cancellation.

## Read the completed result

```bash
python -m json.tool "${DENSE_ROOT}/reports/dense_cold_aggregate.json"
```

No final-test command exists in this supplemental DAG.

## Recover an interrupted dense closure

Do not requeue an old failed job after its source defect is fixed: its worker
remains pinned to the old commit. Generate an immutable monitor report with
`monitor_hcwdl_campaign.py`, then use `create_hcwdl_dense_recovery.py` and
`submit_hcwdl_dense_recovery.py` from a clean worktree at the repair commit.
The recovery validates and reuses the completed prefix, beginning at the first
failed task. Its exact safety contract is
[`contracts/HCWDL_DENSE_RECOVERY.md`](contracts/HCWDL_DENSE_RECOVERY.md).

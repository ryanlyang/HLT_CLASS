# HCWDL Five-Point Dense Cold 300k Runbook

This campaign runs beside the ten-point ladder from its own immutable worktree
and output root. It reuses the completed unweighted 300k pilot read-only and
does not rerun matching or endpoint qualification.

After pushing the implementation commit to `origin/main`, run:

```bash
(
set -euo pipefail
cd /home/ryreu/atlas/HLT_Classification
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

git fetch origin main
export DENSE5_COMMIT="$(git rev-parse origin/main)"
export DENSE5_SHORT="${DENSE5_COMMIT:0:8}"
export DENSE5_WORKTREE="/home/ryreu/atlas/HLT_Classification_hcwdl_dense5cold300k_${DENSE5_SHORT}"
export DENSE5_ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_dense5cold300k_${DENSE5_SHORT}_r1"
export PARENT_ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_pilot_b3154d67_unweighted_auto_r1"

test -f "${PARENT_ROOT}/campaign_spec.json"
test ! -e "${DENSE5_WORKTREE}"
test ! -e "${DENSE5_ROOT}"
git worktree add --detach "${DENSE5_WORKTREE}" "${DENSE5_COMMIT}"

python -s "${DENSE5_WORKTREE}/scripts/create_hcwdl_dense_pilot.py" \
  --parent-campaign-spec "${PARENT_ROOT}/campaign_spec.json" \
  --campaign-root "${DENSE5_ROOT}" \
  --project-dir "${DENSE5_WORKTREE}" \
  --source-commit "${DENSE5_COMMIT}" \
  --rung-step 5 \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL DENSE5 COLD 300K EXACT SPEC" \
  --output "${DENSE5_ROOT}/campaign_spec.json"

python -s "${DENSE5_WORKTREE}/scripts/submit_hcwdl_dense_pilot.py" \
  --campaign-spec "${DENSE5_ROOT}/campaign_spec.json" \
  --output "${DENSE5_ROOT}/dry_run_ledger.json"

python -s "${DENSE5_WORKTREE}/scripts/submit_hcwdl_dense_pilot.py" \
  --campaign-spec "${DENSE5_ROOT}/campaign_spec.json" \
  --output "${DENSE5_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL DENSE5 COLD 300K EXACT SPEC"

echo "DENSE5_ROOT=${DENSE5_ROOT}"
)
```

Jobs use the independent `hcddp5_*` prefix and are sequential, so this launch
does not alter the running primary or ten-point campaigns. Monitor with:

```bash
squeue --me
sacct -S today --format=JobID,JobName,State,ExitCode,Elapsed,End | grep hcddp5
```

After completion:

```bash
python -m json.tool \
  "/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_dense5cold300k_COMMIT_r1/reports/dense5_cold_aggregate.json"
```

Replace `COMMIT` with the eight-character source commit recorded at launch.

If a source defect interrupts this graph, use the source-pinned failed-closure
recovery in
[`contracts/HCWDL_DENSE_RECOVERY.md`](contracts/HCWDL_DENSE_RECOVERY.md).
It authenticates and retains completed D100offkd, then begins at D95 when D95
is the first failed task. Requeueing the old Slurm job would reuse its broken
source and is forbidden.

If the corrected recovery remains pending because it inherited the original
`320G`/72-hour request, replace it through the measured-resource reschedule in
`contracts/HCWDL_DENSE_RECOVERY.md`. Dense5 receives the identical
`96G`/six-hour profile and preserves its independent graph and output root.

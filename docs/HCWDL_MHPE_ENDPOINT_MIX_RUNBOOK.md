# HCWDL-MHPE Endpoint Mixture Runbook

This queues the seven-job 300k/60-pass endpoint teacher-mixture add-on. It
does not submit until both exact authorization phrases are supplied.

On Tigris, set `SOURCE_SPEC` to one completed dense C10P90 or C25P75 campaign,
then run:

```bash
set -euo pipefail
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
cd "${MAIN_REPO}"
git fetch origin main
export MIX_COMMIT="$(git rev-parse origin/main)"
export MIX_SHORT="${MIX_COMMIT:0:8}"
export MIX_WORKTREE="/home/ryreu/atlas/HLT_Classification_mhpe_endpoint_mix_${MIX_SHORT}"
export MIX_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_endpoint_mix_${MIX_SHORT}_r1"

# Set this explicitly; never discover a source by newest path.
export SOURCE_SPEC=/home/ryreu/atlas/HLT_Classification/checkpoints/REPLACE_WITH_COMPLETED_DENSE_ROOT/campaign_spec.json

test -f "${SOURCE_SPEC}"
test ! -e "${MIX_ROOT}"
if [ -e "${MIX_WORKTREE}" ]; then
  test "$(git -C "${MIX_WORKTREE}" rev-parse HEAD)" = "${MIX_COMMIT}"
  test -z "$(git -C "${MIX_WORKTREE}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach "${MIX_WORKTREE}" "${MIX_COMMIT}"
fi
export PYTHONPATH="${MIX_WORKTREE}/src"

python -s "${MIX_WORKTREE}/scripts/create_hcwdl_mhpe_endpoint_mix_campaign.py" \
  --source-campaign-spec "${SOURCE_SPEC}" \
  --campaign-root "${MIX_ROOT}" \
  --project-dir "${MIX_WORKTREE}" \
  --source-commit "${MIX_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL MHPE ENDPOINT MIX 300K60 EXACT SPEC"

python -s "${MIX_WORKTREE}/scripts/submit_hcwdl_mhpe_endpoint_mix_campaign.py" \
  --spec "${MIX_ROOT}/campaign_spec.json" \
  --output "${MIX_ROOT}/dry_run_submission_ledger.json"

python -s "${MIX_WORKTREE}/scripts/submit_hcwdl_mhpe_endpoint_mix_campaign.py" \
  --spec "${MIX_ROOT}/campaign_spec.json" \
  --output "${MIX_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL MHPE ENDPOINT MIX 300K60 EXACT LEDGER"

squeue --me -o "%.18i %.50j %.2t %.10M %R"
```

Monitor it with:

```bash
python -s "${MIX_WORKTREE}/scripts/monitor_hcwdl_mhpe_endpoint_mix.py" \
  --spec "${MIX_ROOT}/campaign_spec.json" \
  --submission-ledger "${MIX_ROOT}/submission_ledger.json" \
  --output "${MIX_ROOT}/monitor.json"
```

Cancellation uses the generic exact-ID command:

```bash
python -s "${MIX_WORKTREE}/scripts/cancel_hcwdl_campaign.py" \
  --submission-ledger "${MIX_ROOT}/submission_ledger.json"
```

If the authenticated monitor contains a terminal failure, create an exact
failed/downstream recovery (with a new empty `RECOVERY_ROOT`) and dry-run it:

```bash
export RECOVERY_ROOT="${MIX_ROOT}_recovery_r1"
python -s "${MIX_WORKTREE}/scripts/create_hcwdl_mhpe_endpoint_mix_recovery.py" \
  --campaign-spec "${MIX_ROOT}/campaign_spec.json" \
  --submission-ledger "${MIX_ROOT}/submission_ledger.json" \
  --monitor-report "${MIX_ROOT}/monitor.json" \
  --recovery-root "${RECOVERY_ROOT}" \
  --project-dir "${MIX_WORKTREE}" \
  --source-commit "${MIX_COMMIT}" \
  --authorization-phrase "AUTHORIZE HCWDL MHPE ENDPOINT MIX EXACT RECOVERY"

python -s "${MIX_WORKTREE}/scripts/submit_hcwdl_mhpe_endpoint_mix_recovery.py" \
  --recovery-spec "${RECOVERY_ROOT}/recovery_spec.json" \
  --output "${RECOVERY_ROOT}/dry_run_submission_ledger.json"
```

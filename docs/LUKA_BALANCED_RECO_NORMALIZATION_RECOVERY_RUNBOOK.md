# Recover the existing balanced reconstruction campaign

## Retry after the de6e9309 sbatch script rejection

The earlier estimates were `--test-only` output. The first live normalization
submission failed because sbatch received `bash` as its script filename.
No normalization/release/replacement job was accepted by that invocation.
Pending original reconstruction/report cancellations had already been requested.
Keep that failed recovery directory and its logs/intents; do not remove its
claim or retry the same commit/root. The corrected launcher uses the actual
shebang script for both dry review and live submission.

In local PowerShell, commit only this submission fix:

```powershell
$FixFiles = @(
  'src/hlt_classification/operations/hydra_reconstruction_normalization_recovery.py'
  'tests/test_luka_reconstruction_normalization_recovery.py'
  'docs/plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md'
  'docs/contracts/LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY.md'
  'docs/LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY_RUNBOOK.md'
  'docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md'
)
git add -- $FixFiles
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git commit --only -m "Submit recovery batch scripts directly to Slurm" -- $FixFiles
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

Use that new 40-character commit with the Hydra launch block below. The new
commit creates a fresh recovery root. It still uses the bundled exact
normalization, retains completed/running models, and replaces eligible
cancelled/failed original reconstruction attempts. No generic campaign restart.

## Normal recovery launch

This continues the dc5920f0 study. Do not launch the generic 18-fit campaign
again. Completed and running models remain in place. The operation cancels
only pending reconstruction/report jobs, then submits replacements for failed
or cancelled reconstruction fits after recovering the exact qualified normalizer.
Queued original baselines/teachers remain intact. See the
[recovery contract](contracts/LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY.md).

On 2026-11-23 the operator captured the complete normalization from running
R22 job 219510. Its content hash matches qualification 219148 exactly. The record
is preserved in `configs/operations/balanced_reco_dc5920f0_normalization.json`.
The queue script uses it by default: the normalization stage validates and
publishes this small record on a CPU allocation, without rebuilding inputs or
waiting for gpu1905. Each replacement still constructs its own training inputs.

The low-level CLI retains an explicit original-node replay fallback if no
candidate is supplied. That fallback must reproduce the exact qualified hash
before releasing fits; it saves the full statistics if the hash differs.

Local PowerShell, with unrelated staged changes reviewed first:

```powershell
git add -- configs/operations/balanced_reco_dc5920f0_normalization.json `
  src/hlt_classification/luka_balanced_reconstruction_hydra/normalization.py `
  src/hlt_classification/luka_balanced_reconstruction_hydra/inputs.py `
  src/hlt_classification/luka_balanced_reconstruction_hydra/resources.py `
  src/hlt_classification/luka_balanced_reconstruction_hydra/worker.py `
  src/hlt_classification/operations/hydra_reconstruction_normalization_recovery.py `
  scripts/recover_luka_reconstruction_normalization.py `
  scripts/queue_luka_reconstruction_normalization_recovery.sh `
  sbatch/run_luka_reconstruction_normalization_recovery.sh `
  tests/test_luka_reconstruction_normalization_recovery.py `
  tests/test_luka_balanced_reconstruction_hydra.py `
  docs/plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md `
  docs/contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA_INPUTS.md `
  docs/contracts/LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY.md `
  docs/LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY_RUNBOOK.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_RUNBOOK.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md
git diff --cached --stat
git commit -m "Freeze reconstruction normalization and recover existing Hydra fits"
git push origin HEAD:main
git rev-parse HEAD
```

Use the pushed 40-character commit in the Hydra terminal:

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
read -r -p 'Pushed normalization-repair commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$BASE/code_65952ed3" fetch origin main
git -C "$BASE/code_65952ed3" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_balanced_norm_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$BASE/code_65952ed3" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/queue_luka_reconstruction_normalization_recovery.sh" "$COMMIT"
BASH
```

For an existing full JSON normalization record, pass its absolute path as the
second argument of that final command. Its content must match the old qualified
hash; path existence does not authorize reuse.

After DETACHED, disconnecting or Ctrl-C only closes the log view. Run once.
Look for successful normalization, then replacement IDs. The preserved running
R22 or other original jobs are not resubmitted while live. No completed baseline
or teacher is retrained. Genuine filesystem errors still propagate.

Reconnect using the same commit:

```bash
ROOT=/cs/home/rlyang/hlt_u100_pairs/runs/balanced_reco_normalization_${COMMIT:0:8}_r1
tail -n 80 "$ROOT.launch/launcher.log"
cat "$ROOT/normalization/complete.json"
squeue --me -o '%.18i %.40j %.2t %R'
```

Read old and repaired scientific results with the unchanged original source:

```bash
PROJECT=/cs/home/rlyang/hlt_u100_pairs/code_balanced_reco_dc5920f0
export PROJECT_DIR="$PROJECT"
export ENV_DIR=/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv
source "$PROJECT/sbatch/luka_balanced_reconstruction_hydra_common.sh"
"$ENV_DIR/bin/python" -u -B -s "$PROJECT/scripts/luka_balanced_reconstruction_hydra.py" results \
  --spec /cs/home/rlyang/hlt_u100_pairs/runs/balanced_reco_600k_dc5920f0_r1/campaign_spec.json --rejections
```

Do not invoke the old scientific resume command. Keep all original and repair
records, including failed attempts. Missing/partial outputs are not success.

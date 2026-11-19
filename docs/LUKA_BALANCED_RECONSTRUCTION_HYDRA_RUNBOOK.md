# Queue the 600k balanced reconstruction screen on Hydra

This is **18 scientific fits**, two matched seeds,20 epochs each, on **four
GTX1080Ti GPUs per fit**. It trains new OFFLINE/M0HLT/U100 teachers and compares
U100-w5, R24, D20, R20, R22 and R28. Four readouts per trajectory include EMA05
as primary; EMA is not an additional training job. Dataset600–2600GeV, six
classes,600k TRAIN/150k natural validation. See the
[plan](plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md),
[contract](contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA.md) and
[validation](LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md).

Existing Oscar jobs and Hydra ladders stay queued/running. This campaign does
not reuse their scientific weights or create a persistent particle cache. The
initial three jobs prepare, qualify actual hardware, then automatically release
the18 fits plus report. The fits run serially. Qualification failure releases
no fits; it preserves diagnostic records. 48h is a per-fit envelope, not a
measured runtime guarantee for all18 fits.

## Commit and push from local PowerShell

These paths stage only this campaign's new files. Review any already staged
work before committing. Shared README/HANDOFF/index/donor-map notes can be
committed separately with your pending documentation changes; they are not
required for the executable campaign snapshot.

```powershell
git diff --cached --stat
git add -- src/hlt_classification/luka_balanced_reconstruction_hydra `
  scripts/luka_balanced_reconstruction_hydra.py `
  scripts/launch_luka_balanced_reconstruction_hydra.sh `
  scripts/queue_luka_balanced_reconstruction_hydra.sh `
  sbatch/luka_balanced_reconstruction_hydra_common.sh `
  sbatch/run_luka_balanced_reconstruction_hydra.sh `
  tests/test_luka_balanced_reconstruction_hydra.py `
  docs/plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md `
  docs/contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_RUNBOOK.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md
git diff --cached --stat
git commit -m "Add balanced 600k reconstruction screen for four Hydra GTX1080Ti GPUs"
git push origin HEAD:main
git rev-parse HEAD
```

Use that pushed40-character commit below. If another branch advanced main,
integrate it normally without resetting unrelated work. The remote launch uses
its own detached clean worktree, so other local unstaged work is preserved.

## Queue from your authenticated Hydra Bash terminal

The verified dataset already exists on Hydra. No new upload is needed.
The shared reservation can require26GiB currently free when none of the
protected ladder's14GiB has been spent; committed ladder outputs reduce that
unspent reservation. Declare current whole freeGiB, not the earlier27GiB value.
The launcher also checks actual filesystem space before submitting.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="$BASE/code_65952ed3"
df -h "$BASE"
read -r -p 'Pushed 40-character balanced-reconstruction commit: ' COMMIT </dev/tty
read -r -p 'Current available HOME space in whole GiB: ' FREE_GIB </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$FREE_GIB" =~ ^[0-9]+$ ]]
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_balanced_reco_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/queue_luka_balanced_reconstruction_hydra.sh" "$COMMIT" "$FREE_GIB"
BASH
```

**After `DETACHED balanced reconstruction launcher PID ...`, you can disconnect.**
Authentication, population replay, dry scheduler review and bootstrap submission
continue without your terminal. The tail is just a view; Ctrl-C closes the view.
Do not launch again. `LAUNCHER_EXIT=0` with three accepted IDs means the bootstrap
queued successfully. Later the successful release job adds18 science IDs and
one report. `LAUNCHER_EXIT=0` alone does not mean scientific results already exist.

The persistent runtime is
`/cs/home/rlyang/hlt_u100_pairs/environment_pair_hydra_v1/venv`.
Requests are partition`gpus`, account`default`, qos`normal`,
`gpu:geforce_gtx_1080_ti:4`,12 CPUs,64000MiB,48h. CPU prep has4h;
the four-GPU qualification has6h. GPU qualification measures full support and
timing before release; resource failure requires an explicit revision.

## Reconnect, progress and results

Set COMMIT to your launch commit in these blocks. Saved results need no inference.

```bash
COMMIT=PASTE_YOUR_40_CHARACTER_COMMIT
BASE=/cs/home/rlyang/hlt_u100_pairs
PROJECT="$BASE/code_balanced_reco_${COMMIT:0:8}"
ROOT="$BASE/runs/balanced_reco_600k_${COMMIT:0:8}_r1"
tail -n 60 "$ROOT.launch/launcher.log"
squeue --me -o '%.18i %.40j %.2t %.10M %R'
```

```bash
export PROJECT_DIR="$PROJECT"
export ENV_DIR="$BASE/environment_pair_hydra_v1/venv"
source "$PROJECT/sbatch/luka_balanced_reconstruction_hydra_common.sh"
PYTHON_EXE="$ENV_DIR/bin/python"
"$PYTHON_EXE" -B -s "$PROJECT/scripts/luka_balanced_reconstruction_hydra.py" progress \
    --spec "$ROOT/campaign_spec.json" --task R24_s10 --readout EMA05
"$PYTHON_EXE" -B -s "$PROJECT/scripts/luka_balanced_reconstruction_hydra.py" results \
    --spec "$ROOT/campaign_spec.json" --readout EMA05 --rejections
```

Choose any registered task, such as`U100_W5_s10`,`D20_s10` or`R24_s11`.
EMA starts at position3; earlier progress has RAW/RAW_BN only. Recovery becomes
available after the same-seed OFFLINE/M0HLT outputs exist. R50/R30 use each
readout's AUC-selected checkpoint. `results.json` contains paired aggregates,
late16..20 summaries and missing-row inventory. Incomplete afterany reports are
kept separately in`reports/<job>.json`;`results.json` is published only when all
fits are present. Native deployable checkpoints
and prediction files are referenced by`fits/<task>/result.json`.

## Technical continuation

Preserve all claims, journals and partial outputs. First inspect exact task logs
and accounting; a missing accepted-ID receipt requires reconciliation, not retry.
For an ordinary failed allocation after all accepted campaign jobs are terminal,
use the same source, spec and root. This resumes committed epochs and preserves
completed results; it does not change any other campaign's jobs.

```bash
bash "$PROJECT/scripts/launch_luka_balanced_reconstruction_hydra.sh" resume-detach \
    "$PROJECT" "$ROOT" "$COMMIT" "$FREE_GIB" \
    "$BASE/environment_pair_hydra_v1/venv" \
    --authorize AUTHORIZE_BALANCED_RECONSTRUCTION_18_FOUR_1080TI_V1
```

Refresh FREE_GIB from current storage first. An incomplete registration without
`campaign_spec.json` is not an epoch resume: inspect its log and preserve it.
Source/runtime drift, invalid data, ambiguous submissions and insufficient space
fail closed. Poor physics metrics are retained scientific results.

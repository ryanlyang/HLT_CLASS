# Lower-LR OFFLINE -> DE000 -> M0HLT on Oscar

Run three fresh20-epoch fits in that exact order, at batch128 and peakLR1e-4.
DE000 has CE1 on its deployed classifier, four embedding losses2.5 each,
four separate auxiliary KD losses0.25 each, and contrastive0.3 with ramp3.
New OFFLINE supplies its embedding/logit/key targets; completed CONCAT, UE100
and DE050 remain frozen historical teachers. The primary checkpoint rule
remains in-range macro AUC. Full-validation Hbb AUC/R50/R30 are saved every epoch.

See [plan](plans/LUKA_SIXCLASS_LOW_LR_OSCAR_PLAN.md),
[contracts](contracts/LUKA_SIXCLASS_LOW_LR_OSCAR.md) and
[local evidence](LUKA_SIXCLASS_LOW_LR_OSCAR_VALIDATION.md).

## Commit and push locally

These paths contain this campaign. Review the diff before committing. README,
HANDOFF and LEGACY_SOURCE_MAP contain unrelated edits; stage their new campaign
sections interactively rather than staging entire dirty files.

```powershell
git add -- src/hlt_classification/luka_sixclass_low_lr_oscar `
  scripts/luka_sixclass_low_lr_oscar.py scripts/launch_luka_sixclass_low_lr_oscar.sh `
  scripts/queue_luka_sixclass_low_lr_oscar.sh `
  sbatch/run_luka_sixclass_low_lr_oscar.sh sbatch/luka_sixclass_low_lr_oscar_common.sh `
  tests/test_luka_sixclass_low_lr_oscar.py `
  docs/plans/LUKA_SIXCLASS_LOW_LR_OSCAR_PLAN.md `
  docs/contracts/LUKA_SIXCLASS_LOW_LR_OSCAR.md `
  docs/LUKA_SIXCLASS_LOW_LR_OSCAR_RUNBOOK.md docs/LUKA_SIXCLASS_LOW_LR_OSCAR_VALIDATION.md
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Add lower-LR Oscar OFFLINE DE000 M0HLT mini campaign"
git push origin main
git rev-parse HEAD
```

## Queue on Oscar

This requires the healthy persistent HOME environment and the original completed
six-class campaign/source/population/banks. Raw data stay on scratch. No runtime
repair, data transfer, teacher retraining or separate preflight job is performed.
Reserve at least8GiB of currently free HOME space. Each fit requests oneL40S,
six CPUs,192000MiB and48hours. Startup in each allocation measures the real work.

Paste this in your Oscar terminal, using the40-character commit printed above:

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_sixclass_balanced_oscar_8a25f126"
checkquota
read -r -p 'Pushed 40-character mini-campaign commit: ' COMMIT </dev/tty
read -r -p 'Currently free HOME GiB (minimum 8): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
PROJECT="${BASE}/source/HLT_Classification_sixclass_low_lr_oscar_${COMMIT:0:8}"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
  git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/queue_luka_sixclass_low_lr_oscar.sh" "$COMMIT" "$QUOTA"
BASH
```

The launcher detaches before authentication, dry review and submission.
Ctrl-C after `DETACHED` ends only the log view. `LAUNCHER_EXIT=0` and three
`SUBMITTED` lines confirm submission; do not launch a duplicate.
M0HLT waits for DE000 by request. Recovery using new anchors remains unavailable
until M0HLT publishes. No other jobs are cancelled or modified.

## Progress and results

Set COMMIT to the exact pushed campaign commit in the same Oscar shell:

```bash
COMMIT=PASTE_THE_40_CHARACTER_COMMIT
BASE=/oscar/home/rlyang/hlt_classification
export PROJECT_DIR="${BASE}/source/HLT_Classification_sixclass_low_lr_oscar_${COMMIT:0:8}"
export ENV_DIR="${BASE}/environments/new_balanced_oscar_v1"
ROOT="${BASE}/checkpoints/luka_sixclass_low_lr_oscar_${COMMIT:0:8}_r1"
source "$PROJECT_DIR/sbatch/luka_sixclass_low_lr_oscar_common.sh"
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_low_lr_oscar.py" progress \
  --spec "$ROOT/campaign_spec.json" --stage DE000
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_low_lr_oscar.py" results \
  --spec "$ROOT/campaign_spec.json"
```

Progress reads small committed epoch records, including full Hbb metrics, and
does no inference. Results authenticates completed payloads and outputs JSON
with primary-selected in/out/full R50/R30 and separately labeled Hbb-selected
diagnostics. It supports partially completed campaigns.

## Interrupted execution

Preserve claims, attempts, logs and epoch files. Inspect exact campaign job IDs
and the failure before continuing. `resume-detach` requires every prior
registered job to be terminal; cancel pending dependent IDs only if explicitly
needed. It reuses published fits, restores complete optimizer/RNG state, and
can finish export after all20epochs without another training update.

```bash
bash "$PROJECT_DIR/scripts/launch_luka_sixclass_low_lr_oscar.sh" resume-detach \
  "$PROJECT_DIR" /oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1 \
  "$ROOT" "$COMMIT" CURRENT_FREE_HOME_GIB "$ENV_DIR" \
  --authorize AUTHORIZE_SIXCLASS_LOW_LR_OSCAR_V1
```

Never delete submission intents to retry an ambiguous sbatch response. Reconcile
the actual scheduler IDs first. Existing completed results and original source
must remain available for this experiment's frozen-teacher lineage.

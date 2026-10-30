# Oscar FullStat retained U100 predictor

Two fresh 20-epoch fits on the existing 3M/500k split: retained CE1/E5 first,
then the same expanded architecture with CE only. Both keep the predictor at
inference. Completed U100-w5 is a historical reference. No standalone preflight.
The control uses afterany ordering and is not suppressed by a failed retained
fit. [Scientific plan](plans/LUKA_FULLSTAT_RETAINED_PLAN.md).

## Commit and push locally

Stage only these implementation files plus the relevant documentation hunks
in HANDOFF, LEGACY_SOURCE_MAP and plans/README. Preserve unrelated work.

```powershell
git add -- src/hlt_classification/luka_fullstat_retained scripts/luka_fullstat_retained.py scripts/launch_luka_fullstat_retained.sh scripts/queue_luka_fullstat_retained.sh sbatch/run_luka_fullstat_retained.sh tests/test_luka_fullstat_retained.py tests/test_luka_fullstat_retained_launch.py docs/plans/LUKA_FULLSTAT_RETAINED_PLAN.md docs/contracts/LUKA_FULLSTAT_RETAINED.md docs/LUKA_FULLSTAT_RETAINED_RUNBOOK.md docs/LUKA_U100_RETAINED_PREDICTOR_HANDOFF.md
git diff --cached --stat
git commit -m "Add FullStat retained U100 predictor and matched CE control"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## One detached Oscar launch

Check `checkquota` before launch: at least **9 GiB** of home quota must remain
after other reserved/running outputs. The 8 GiB result budget includes both
fits' 20 full epoch states plus selected states, predictions and reports; the
extra 1 GiB is headroom. The filesystem check does not replace user quota.

Paste the pushed 40-character commit when prompted. That is the only prompt.
The existing repository path below is the same main checkout used for prior
FullStat launches; adjust REPO only if your checkout is elsewhere.

```bash
(
set -euo pipefail
read -r -p 'Paste the pushed retained-predictor commit: ' COMMIT
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
REPO=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_retained_${COMMIT:0:8}
ROOT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_retained_${COMMIT:0:8}_r1
PARENT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_readout_db9afe8b_r1/campaign_spec.json
git -C "$REPO" fetch origin main
git -C "$REPO" cat-file -e "${COMMIT}^{commit}"
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ ! -e "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/launch_luka_fullstat_retained.sh" \
    "$COMMIT" "$ROOT" "$PARENT" 9 \
    --authorize AUTHORIZE_FULLSTAT_RETAINED_AUTOSUBMIT_V1
)
```

After **DETACHED: launcher PID ...**, you may disconnect. The foreground is
only a log tail. Authentication, full dry review and both submissions continue
without another prompt. LAUNCH_COMPLETE means Slurm owns both jobs. Detachment
survives SSH disconnection, not a login-node restart or administrative kill.
This command never cancels existing jobs. Do not launch it twice. If it stops,
preserve `<ROOT>.launch/failure.json` and submission journals for inspection.

```bash
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
tail -n 60 -f "${ROOT}.launch/launcher.log"
```

The worker authenticates the inherited corrected runtime at
`/oscar/scratch/rlyang/hlt_classification/runtime_repair_v1/atlas_kd_oscar`.
Each job performs its own bounded graph/resource checks and then starts fresh
training. No historical source, target, checkpoint or job is modified.

## Results

With PROJECT and ROOT set to the exact launch paths above:

```bash
(
set -euo pipefail
export PROJECT_DIR="$PROJECT" JC2_SITE=oscar_l40s
source "$PROJECT_DIR/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "$PROJECT_DIR/scripts/luka_fullstat_retained.py" results \
    --spec "$ROOT/campaign_spec.json"
)
```

This reads saved validation predictions/reports, verifies completed receipts
and prints selected AUC/recovery plus JSON containing accuracy, R50/R30 and their
recoveries. The retained result can be read while the control is pending or
running; an absent control receipt is listed explicitly. It does not perform
new inference or open test data. For quick live epoch/progress inspection:

```bash
grep -E '^FULLSTAT RETAINED (task|progress)' "$ROOT"/train_retained-*.out | tail -n 40
```

The log is provisional until final publication succeeds. Every completed epoch
has its full graph/optimizer/RNG state under `<ROOT>/<task>/epochs/NN/`.
Final selected.pt must be loaded with the retained model, not native ParT.

# Queue the full D20 RAW campaign on Hydra

After the226613 qualification failure at commit0051dc8e, use a **new commit and
fresh study** for the input-schema repair. That job passed its four-GPU early
checks, then failed on the old `particles` field before particle construction
or scientific training. The repair consumes the existing `role_particles`
schema throughout allocation and coverage checks; no scientific setting changes.
Do not apply a new source to the old immutable spec or use its same-source
`resume` command. Leave the failed execution in place for provenance.

For this repair, the local commit block can use this smaller file list and
commit message (then use the same Hydra launch block below with the new hash):

```powershell
$Files = @(
  'src/hlt_classification/luka_d20_full_hydra/inputs.py'
  'tests/test_luka_d20_full_hydra.py'
  'docs/LUKA_D20_FULL_HYDRA_RUNBOOK.md'
  'docs/LUKA_D20_FULL_HYDRA_VALIDATION.md'
)
git add -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat -- $Files
git commit --only -m "Fix full D20 wide-population input builder" -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

This is the full2,488,148 TRAIN /497,954 natural validation campaign,200--2600GeV,
six classes, equal class/pT sampling, batch256, peakLR1e-4,20 epochs. Three seeds
train OFFLINE/M0HLT/U100/U100-w5/D20 and eight first-seed variants:23 fits.
Each fit has four five-epoch continuation jobs, each4 GTX1080Ti/12CPUs/96,000MiB/
48h. The92 segments are23 models; segment continuations preserve optimizer,
decoder, RNG and the same twenty-epoch schedule.

Read the [scientific plan](plans/LUKA_D20_FULL_HYDRA_PLAN.md),
[contract](contracts/LUKA_D20_FULL_HYDRA.md) and
[validation evidence](LUKA_D20_FULL_HYDRA_VALIDATION.md). RAW is primary;
OFFLINE/U100 retain EMA05 for outgoing targets. No historical checkpoint/bank
or old reconstruction normalizer is a scientific dependency. The verified new
dataset, persistent Hydra runtime and shared Git store must remain available.

In local PowerShell, commit only this campaign:

```powershell
Set-Location 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification'
$Files = @(
  'src/hlt_classification/luka_d20_full_hydra'
  'scripts/luka_d20_full_hydra.py'
  'scripts/launch_luka_d20_full_hydra.sh'
  'scripts/queue_luka_d20_full_hydra.sh'
  'sbatch/run_luka_d20_full_hydra.sh'
  'sbatch/luka_d20_full_hydra_common.sh'
  'tests/test_luka_d20_full_hydra.py'
  'docs/plans/LUKA_D20_FULL_HYDRA_PLAN.md'
  'docs/contracts/LUKA_D20_FULL_HYDRA.md'
  'docs/LUKA_D20_FULL_HYDRA_RUNBOOK.md'
  'docs/LUKA_D20_FULL_HYDRA_VALIDATION.md'
)
git add -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git diff --cached --stat -- $Files
git commit --only -m "Add full-population three-seed RAW D20 Hydra campaign" -- $Files
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

The shared README/HANDOFF/LEGACY_SOURCE_MAP contain unrelated prior changes;
their new entries can be reviewed separately. They are deliberately excluded
from whole-file staging above. The separate hardware benchmark is not needed
for this campaign's own mandatory four-card qualification.

In your Hydra SSH terminal:

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
read -r -p 'Pushed full D20 campaign commit (40 characters): ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
AVAILABLE=$(/usr/bin/python3 -I -B - "$BASE" <<'PY'
import shutil,sys
print(shutil.disk_usage(sys.argv[1]).free // 2**30)
PY
)
printf 'Available: %s whole GiB; initial admission requires 26.\n' "$AVAILABLE"
git -C "$BASE/code_65952ed3" fetch origin main
git -C "$BASE/code_65952ed3" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_d20_full_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$BASE/code_65952ed3" worktree add --detach "$PROJECT" "$COMMIT"
fi
ROOT="$BASE/runs/d20_full_${COMMIT:0:8}_r1"
bash "$PROJECT/scripts/launch_luka_d20_full_hydra.sh" "$COMMIT" "$ROOT" "$AVAILABLE"
BASH
```

Run once. DETACHED means the launcher survives disconnect. Ctrl-C closes only
the log view. First it authenticates source/runtime/transfer/scalar population
and prints a complete actual-script dry plan. It submits prepare -> qualify ->
release. Qualification has a48h outer limit because it constructs full views,
measures complete validation and all13 objectives; GPU transport/native checks
have a separate five-minute early limit before particle preparation.

Only successful real four-GPU qualification permits automatic release of92
science segments and a report. A benchmark result alone cannot release them.
Independent first segments may all wait at Priority; Dependency applies only
to their actual teacher or same-fit continuation. There is no whole-campaign
serial chain. Resource failure or uncertain submission preserves evidence;
do not delete claims/intents or blindly launch the same study twice.

After saving the same full commit in your Hydra shell:

```bash
BASE=/cs/home/rlyang/hlt_u100_pairs
PROJECT="$BASE/code_d20_full_${COMMIT:0:8}"
ROOT="$BASE/runs/d20_full_${COMMIT:0:8}_r1"
squeue --me -o '%.18i %.40j %.2t %.10M %R'
tail -n 60 -f "$ROOT.launch/launcher.log"
```

For saved results and R50/R30, no new inference:

```bash
export PROJECT_DIR="$PROJECT"
export ENV_DIR="$BASE/environment_pair_hydra_v1/venv"
source "$PROJECT_DIR/sbatch/luka_d20_full_hydra_common.sh"
"$ENV_DIR/bin/python" -I -u -B "$PROJECT/scripts/luka_d20_full_hydra.py" results \
  --spec "$ROOT/campaign_spec.json" --readout RAW --rejections
```

For epoch progress, substitute any registered fit such asD20_s20 orD20_E06_s20:

```bash
"$ENV_DIR/bin/python" -I -u -B "$PROJECT/scripts/luka_d20_full_hydra.py" progress \
  --spec "$ROOT/campaign_spec.json" --task D20_s20 --readout RAW
```

Explicit recovery uses the same immutable source/spec, only after all accepted
jobs are terminal. It preserves successful segments and resumes incomplete
ones from their last committed epoch. It neither cancels jobs nor accepts a
changed scientific recipe:

```bash
AVAILABLE=$(/usr/bin/python3 -I -B - "$BASE" <<'PY'
import shutil,sys
print(shutil.disk_usage(sys.argv[1]).free // 2**30)
PY
)
"$ENV_DIR/bin/python" -I -u -B "$PROJECT/scripts/luka_d20_full_hydra.py" resume \
  --spec "$ROOT/campaign_spec.json" --available-gib "$AVAILABLE" \
  --authorize AUTHORIZE_D20_FULL_23_FOUR_1080TI_V1
```

Preserve any uncertain submission journal for exact-ID reconciliation. Pending
or running jobs are not terminal and must not be replaced. No final-test
inference, cleanup of earlier campaigns or automatic HOME recheck is performed.

# New dataset: transfer and natural OFFLINE baseline on Brown CS Hydra

Use this only for the new QCD Run-2-combined / Higgs snapshot. See the
[plan](plans/LUKA_NEW_FULLSTAT_HYDRA_OFFLINE_PLAN.md) and
[contracts](contracts/LUKA_NEW_FULLSTAT_HYDRA.md).

Exactly one fresh OFFLINE CE fit: natural 3M TRAIN, 500k validation, sealed
1.5M test, 20 epochs, global batch 128, original warmup/cosine/AdamW schedule.
No flattening and no old recovery anchors. Default is one L40S; alternatives
are explicitly registered one L40 or one A6000, preserving BF16 and batch 128.

## 1. Upload from the local PowerShell terminal

The agreed split has already been created locally. Its content hash is
`2f05adcca889288e705a69fc1e54670f1713d29ec16ab75c4f694ab7b5a23d02`.
Do not rebuild it or delete its directory. Transfer 13,995 files / 26,912,173,193
bytes (~25.06 GiB), including every raw ROOT file, scalar memberships and
producer evidence. Manifest hash:
`7034f2153e585068000185ce4c1e7fb3e61fe96e5bbfeb5e650608a150071ae4`.

```powershell
cd C:\Users\22rya\ComputerScience\CERN\HLT_Classification
& C:\Users\22rya\miniconda3\envs\tagging-hlt\python.exe -B -s scripts\transfer_luka_new_fullstat.py send --root artifacts\datasets\luka_qcdrun2com_higgs2p_sb_v1 --identity-file "$env:USERPROFILE\.ssh\id_rsa_brown_cs"
```

This selects the same nondefault key as the user's working `ssh -i` command.
The transfer utility passes its path to OpenSSH and never reads or copies
private-key contents. Authenticate to Brown normally. Keep this terminal open until
`TRANSFER VERIFIED` appears. It streams directly into
`/cs/home/rlyang/hlt_u100_pairs/datasets/luka_qcdrun2com_higgs2p_sb_v1`;
there is no second remote tarball. A disconnected transfer can be retried with
the same command: existing files must match, and the entire arriving stream
is checked again. Mismatched files or a different destination manifest stop.
Do not delete partial directories or bypass verification.

## 2. Commit and push only this implementation locally

Preserve unrelated changes from other chats. These explicit paths omit shared
README/HANDOFF/source-map edits, which can be committed separately.

```powershell
git add -- src/hlt_classification/luka_new_fullstat_hydra scripts/luka_new_fullstat_hydra.py scripts/transfer_luka_new_fullstat.py scripts/launch_luka_new_fullstat_hydra.sh sbatch/run_luka_new_fullstat_hydra.sh tests/test_luka_new_fullstat_hydra.py docs/plans/LUKA_NEW_FULLSTAT_HYDRA_OFFLINE_PLAN.md docs/contracts/LUKA_NEW_FULLSTAT_HYDRA.md docs/LUKA_NEW_FULLSTAT_HYDRA_RUNBOOK.md docs/LUKA_NEW_FULLSTAT_HYDRA_VALIDATION.md
git diff --cached --stat
git commit -m "Add new FullSim dataset and natural OFFLINE Hydra baseline"
git push origin main
git rev-parse HEAD
```

Review the staged list before committing if there were previously staged
unrelated changes. Production uses a separate clean worktree of the full pushed
40-character commit. Do not update code directories used by existing jobs.

## 3. Compare current scheduling estimates on Brown

The user supplied Hydra inventory has mixed-use L40S, L40 and A6000 nodes.
Mixed state alone does not indicate how many cards are free. This read-only
command tests the measured scalar RAM envelope: six CPUs, 40000 MiB, one GPU,
48-hour ceiling. It submits no jobs.

```bash
python3 -I -B - <<'PY'
import os
import subprocess
env = {k: v for k, v in os.environ.items()
       if not k.startswith(('SBATCH_', 'SLURM_'))}
for gpu in ('nvidia_l40s', 'nvidia_l40', 'nvidia_rtx_a6000'):
    print('\n' + gpu, flush=True)
    subprocess.run([
        'sbatch', '--test-only', '--partition=gpus', '--account=default',
        '--qos=normal', '--nodes=1', '--ntasks=1', '--cpus-per-task=6',
        '--mem=40000M', '--gres=gpu:' + gpu + ':1', '--time=2880',
        '--export=NONE', '--wrap=true'], env=env, check=False)
print('Scheduling estimates only; no jobs submitted.')
PY
```

Prefer L40S when its start is reasonable. Set `GPU=l40` or `GPU=a6000` in the
launch block only if that explicit alternative is chosen before registration.
Estimates can move. No multi-GPU speedup or schedule prediction is promised.

## 4. Launch on Brown after transfer completes

The raw transfer uses about 25.06 GiB. Prepared OFFLINE TRAIN/validation RAM
is about 15.80 GiB; allow **27 GiB additional free quota** for prepared inputs,
epoch states, results and headroom. Check current space after uploading, not
the earlier 58 GiB figure. Keep all existing campaigns and environments.
The existing `environment_pair_hydra_v1/venv` is used without modification;
registration pins its installed package and Weaver source identity.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
DATASET="$BASE/datasets/luka_qcdrun2com_higgs2p_sb_v1"
test -f "$DATASET/transfer_complete.json"
df -h "$BASE"
read -r -p 'Pushed 40-character baseline commit: ' COMMIT </dev/tty
read -r -p 'Available additional HOME quota GiB after upload (minimum 27): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ && "$QUOTA" -ge 27 ]]
GPU=l40s
REPO="$BASE/code_65952ed3"
PROJECT="$BASE/code_new_fullstat_${COMMIT:0:8}"
ROOT="$BASE/runs/new_fullstat_offline_${COMMIT:0:8}_r1"
CACHE="$BASE/cache_new_fullstat_offline_${COMMIT:0:8}_r1"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
if [[ -e "$PROJECT" ]]; then
    test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
    test -z "$(git -C "$PROJECT" status --porcelain --untracked-files=all)"
else
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
mkdir -p "$BASE/runs"
bash "$PROJECT/scripts/launch_luka_new_fullstat_hydra.sh" detach \
    "$PROJECT" "$DATASET" "$ROOT" "$CACHE" "$COMMIT" "$GPU" "$QUOTA"
BASH
```

After `DETACHED`, disconnecting only stops the log tail. Registration,
full dry review, CPU preparation submission and dependent GPU submission
continue. `LAUNCHER_EXIT=0` and two `SUBMITTED` IDs establish submission;
they do not mean training completed. A launcher error preserves records;
inspect them rather than launching again blindly. The CPU job prepares only
OFFLINE inputs. The GPU job performs installed-Weaver/real-data miniature,
support, reload and timing checks before fresh scientific training.

## 5. Progress and completion

Use the PROJECT and ROOT values printed/selected above:

```bash
"$BASE/environment_pair_hydra_v1/venv/bin/python" -u -B -s \
    "$PROJECT/scripts/luka_new_fullstat_hydra.py" progress --spec "$ROOT/campaign_spec.json"
squeue --me -o '%.18i %.36j %.2t %.10M %R'
```

Each completed epoch has a hashed receipt plus full state under `epochs/`.
`result.json` appears only after selected checkpoint reload and prediction
export. Its report contains accuracy, macro AUC, R50, R30 and the complete
validation history. Both rejections use the same AUC-selected checkpoint.
HLT-to-OFFLINE recovery on this new dataset requires a future matched HLT fit.

If the GPU job fails after completed epochs, and preparation completed, an
explicit continuation can use the same immutable spec and source:

```bash
"$BASE/environment_pair_hydra_v1/venv/bin/python" -u -B -s \
    "$PROJECT/scripts/luka_new_fullstat_hydra.py" queue --spec "$ROOT/campaign_spec.json" \
    --resume --execute --authorize AUTHORIZE_NEW_FULLSTAT_HYDRA_OFFLINE_V1
```

This refuses live jobs, ambiguous submissions or a completed result. It runs
the GPU checks and restores the last committed epoch, including optimizer/RNG.
It does not repair scientific bugs or migrate source. Never edit a saved spec
or use resume to bypass a failing allocation/qualification check.

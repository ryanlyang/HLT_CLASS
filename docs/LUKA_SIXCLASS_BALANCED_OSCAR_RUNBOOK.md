# Six-class 600–2600 GeV Oscar ladder

Read [the scientific plan](plans/LUKA_SIXCLASS_BALANCED_OSCAR_PLAN.md) and
[validation evidence](LUKA_SIXCLASS_BALANCED_OSCAR_VALIDATION.md). This launches
six fresh models with direct main-head KD at each ladder rung. It does not rerun
the old eleven-class campaign or alter Hydra jobs.

## Commit and push locally

In PowerShell at the repository root:

```powershell
git add -- src/hlt_classification/luka_sixclass_balanced_oscar scripts/luka_sixclass_balanced_oscar.py scripts/launch_luka_sixclass_balanced_oscar.sh scripts/queue_luka_sixclass_balanced_oscar.sh scripts/bootstrap_luka_sixclass_balanced_oscar.sh sbatch/run_luka_sixclass_balanced_oscar.sh sbatch/luka_sixclass_balanced_oscar_common.sh tests/test_luka_sixclass_balanced_oscar.py docs/plans/LUKA_SIXCLASS_BALANCED_OSCAR_PLAN.md docs/contracts/LUKA_SIXCLASS_BALANCED_OSCAR.md docs/LUKA_SIXCLASS_BALANCED_OSCAR_RUNBOOK.md docs/LUKA_SIXCLASS_BALANCED_OSCAR_VALIDATION.md
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md docs/plans/README.md
git diff --cached --stat
git commit -m "Add six-class 600-2600 balanced Oscar main-KD ladder"
git push origin main
git rev-parse HEAD
```

Shared documents contain unrelated work; select only this campaign's additions
with `git add -p`. The first command scopes the new implementation files. Do not
use `git add .` or reset/clean the worktree. The runtime-restoration donor is
already committed at `9a633b7d`; this campaign pins donor files at
`d28c403360f7b4662f679f5564529a048d8db5e2`.

## Paste this block on Oscar

This uses the existing `f9dfc80b` checkout as a Git repository and the existing
verified scratch upload. Enter your newly pushed full commit and **currently
available** HOME quota in whole GiB. Need at least 14 GiB if the persistent HOME
runtime already exists and is verified, or 28 GiB if it must be restored.
Account separately for unrelated pending writes; `df` alone is not a user quota.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_new_balanced_oscar_f9dfc80b"
ENV_DIR="${BASE}/environments/new_balanced_oscar_v1"
checkquota
read -r -p 'Pushed 40-character SIX-CLASS commit: ' COMMIT </dev/tty
read -r -p 'Current free HOME GiB (14 with restored runtime; otherwise 28): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
RUNTIME_CONDA=''
if [[ ! -f "${ENV_DIR}.restore/complete.json" ]]; then
    unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
    module load miniforge3/25.3.0-3-a6hh
    RUNTIME_CONDA="$(conda info --base)/bin/conda"
    test -x "$RUNTIME_CONDA"
fi
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="${BASE}/source/HLT_Classification_sixclass_balanced_oscar_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/queue_luka_sixclass_balanced_oscar.sh" "$COMMIT" "$QUOTA" "$RUNTIME_CONDA"
BASH
```

After **`DETACHED six-class launcher PID`**, you may disconnect or Ctrl-C out of
the attached tail. Runtime restoration/verification, scalar population replay,
dry review and submission continue. Restoring exact packages can take time;
population registration also takes several minutes. Successful queueing means
six `SUBMITTED` lines and `BOOTSTRAP_EXIT=0`; DETACHED alone is not submission.

The log is:

```text
/oscar/home/rlyang/hlt_classification/checkpoints/luka_sixclass_balanced_oscar_<commit8>_r1.bootstrap/launcher.log
```

Watch later with `tail -n 60 -f` on that path. Do not launch again while its
status is uncertain. Existing partial runtime/launch records are preserved and
require inspection, never deletion to force a retry. This launcher verifies
all saved IDs of the previous `f9dfc80b` campaign are terminal; it cancels no jobs.
It never invokes the old recovery launcher, which would queue the wrong campaign.

```bash
squeue --me -o '%.18i %.12P %.42j %.2t %.10M %R'
```

Names begin `luka6kd_`. OFFLINE, M0HLT and CONCAT can start independently;
UE100 waits for all three, DE050 for its four teachers, DE000 for its four.
Every request is one L40S, six CPUs, 192000 MiB and 48 hours. The walltime is a
cap, not an estimate. There are six science jobs and **zero separate preflight
jobs**. Each science allocation performs its real technical acceptance before
starting fresh scientific optimization.

## Epoch progress and selected results

Set the commit prefix, then run these on Oscar:

```bash
COMMIT8=REPLACE_WITH_YOUR_COMMIT8
export PROJECT_DIR=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_sixclass_balanced_oscar_${COMMIT8}
ROOT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_sixclass_balanced_oscar_${COMMIT8}_r1
export ENV_DIR=/oscar/home/rlyang/hlt_classification/environments/new_balanced_oscar_v1
source "$PROJECT_DIR/sbatch/luka_sixclass_balanced_oscar_common.sh"
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_balanced_oscar.py" progress --spec "$ROOT/campaign_spec.json" --stage UE100
```

Replace UE100 with OFFLINE, M0HLT, CONCAT, DE050 or DE000. This fast path reads
epoch/report metadata and the latest progress marker. Recovery appears once
fresh six-class M0HLT and OFFLINE have published. It does not load model payloads.

For all published selected-checkpoint metrics, including R50/R30, passing QCD
counts, in/outside-range results, pT bins and recovery:

```bash
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_balanced_oscar.py" results --spec "$ROOT/campaign_spec.json"
```

This authenticates saved payloads and can take longer; no inference or final-test
access. Primary validation has **328,589** natural-distribution jets, not 500k.
All headline metrics and checkpoint selection are for the six-class 600–2600
region. Outside diagnostics use the same selected checkpoint.

## Explicit continuation

Inspect the exact failed job's log and preserved failure record first. All prior
saved jobs, including dependency jobs, must be terminal. Do not remove intent
journals, epoch commits, runtime receipts or result files. An uncertain sbatch
response needs exact scheduler reconciliation before a retry.

Using the same PROJECT_DIR, ROOT and ENV_DIR as above:

```bash
COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
checkquota
read -r -p 'Currently free HOME GiB: ' QUOTA
bash "$PROJECT_DIR/scripts/launch_luka_sixclass_balanced_oscar.sh" resume-detach \
    "$PROJECT_DIR" /oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1 \
    "$ROOT" "$COMMIT" "$QUOTA" "$ENV_DIR" \
    --authorize AUTHORIZE_SIXCLASS_BALANCED_OSCAR_V1
```

Only unpublished stages are queued. Committed epoch optimizer/model/RNG states
are restored; after epoch 20 an export retry performs no new scientific updates.
Completed models and old campaign artifacts are never overwritten.

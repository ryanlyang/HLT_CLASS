# New-data Oscar balanced ladder with main-head KD

Six fresh fits: OFFLINE, M0HLT, CONCAT, UE100, DE050 and DE000. The three students
add 75% predecessor KD / 25% CE to their deployed classifiers. Auxiliary
embedding/logit targets and final OFFLINE contrastive matching retain the
Hydra weights. See the [plan](plans/LUKA_NEW_BALANCED_OSCAR_PLAN.md).

## Local commit and push

Wait for the ongoing data transfer to print `TRANSFER VERIFIED` for the Oscar
scratch destination. That transfer is separate from committing code.

In local PowerShell, stage this campaign only. Shared HANDOFF/source-map/README
files contain other work; use `git add -p` for this campaign's additions there.

```powershell
git add -- src/hlt_classification/luka_new_balanced_oscar scripts/luka_new_balanced_oscar.py scripts/launch_luka_new_balanced_oscar.sh scripts/queue_luka_new_balanced_oscar.sh sbatch/run_luka_new_balanced_oscar.sh sbatch/luka_new_balanced_oscar_common.sh tests/test_luka_new_balanced_oscar.py docs/plans/LUKA_NEW_BALANCED_OSCAR_PLAN.md docs/contracts/LUKA_NEW_BALANCED_OSCAR.md docs/LUKA_NEW_BALANCED_OSCAR_RUNBOOK.md docs/LUKA_NEW_BALANCED_OSCAR_VALIDATION.md
git diff --cached --stat
git commit -m "Add new-data balanced Oscar ladder with main-head KD"
git push origin main
git rev-parse HEAD
```

## Paste on Oscar

Enter that pushed 40-character commit and current available HOME quota in whole
GiB. At least 14 GiB is reserved for this campaign, in addition to space needed
by unrelated running jobs. The raw 25 GiB dataset stays in scratch; no persistent
particle cache is built. The default runtime is now
`/oscar/home/rlyang/hlt_classification/environments/new_balanced_oscar_v1`;
it must already be provisioned and is checked for real imports and fingerprinted.
To use a different already provisioned persistent runtime, export
`LUKA_OSCAR_ENV=/absolute/path/to/environment` before this block.

For the failed `f9dfc80b_r1` execution, use the specific runtime recovery below
instead of the ordinary launch block.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_balanced_ff22b20f"
checkquota
read -r -p 'Pushed 40-character new-data Oscar commit: ' COMMIT </dev/tty
read -r -p 'Available HOME GiB (minimum 14): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
(( QUOTA >= 14 ))
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="${BASE}/source/HLT_Classification_new_balanced_oscar_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/queue_luka_new_balanced_oscar.sh" "$COMMIT" "$QUOTA"
BASH
```

Once **DETACHED launcher PID** and its log path appear, you may disconnect;
registration, population authentication, dry review and submission continue.
Ctrl-C exits only the attached tail. Successful submission is confirmed by six
`SUBMITTED` lines and `LAUNCHER_EXIT=0`, not by DETACHED alone. Do not launch again
while its status is uncertain. The job-name prefix is `lukanokd_`.

```bash
squeue --me -o '%.18i %.40j %.2t %.10M %R'
```

There are six scientific jobs and zero separate preflight jobs. The first three
can run independently; each student waits for all of its teachers. Real parity,
miniature, memory and timing checks run at the start of each science allocation.
The 48-hour request is a cap, not a prediction of runtime.

## Progress and results

Set COMMIT8 to the first eight characters from the successful launch:

```bash
COMMIT8=REPLACE_WITH_YOUR_COMMIT8
export PROJECT_DIR=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_new_balanced_oscar_${COMMIT8}
ROOT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_new_balanced_oscar_${COMMIT8}_r1
export ENV_DIR="$(/usr/bin/python3 -I -B -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment_dir"])' "$ROOT/campaign_spec.json")"
source "$PROJECT_DIR/sbatch/luka_new_balanced_oscar_common.sh"
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_new_balanced_oscar.py" progress --spec "$ROOT/campaign_spec.json" --stage UE100
```

Replace UE100 with any registered stage. Progress reads committed epoch metadata,
latest log markers and AUC/accuracy recovery once both fresh anchors are published.
To authenticate published results, including selected
R50/R30 and new-anchor recovery:

```bash
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_new_balanced_oscar.py" results --spec "$ROOT/campaign_spec.json"
```

The results command reads saved files; it performs no new inference or test
access. It includes every published stage and lists the unfinished ones; recovery
remains null until both anchors exist. Full payload authentication can take longer
than progress inspection.

## Recover the 2026-11-16 Python startup failure

Jobs 9167004/9167005/9167006 exited before the Python training worker because
the registered scratch environment was missing `os.py`, `codecs.py` and
`encodings/__init__.py`. The cause of their deletion is not established. Oscar
[documents individual-file scratch purging](https://docs.ccv.brown.edu/oscar/managing-files/filesystem),
so the recovery provisions a HOME runtime instead. The dataset is not implicated.

This path is specific to the untouched-training execution
`luka_new_balanced_oscar_f9dfc80b_r1`. It preserves its logs/spec/journals and
registers a new execution after runtime verification. If it finds saved training,
ambiguous submissions, or nonterminal old jobs, it stops before new submission.
No full-data model inference or separate preflight job is used for runtime repair.

Commit only this patch in local PowerShell (shared HANDOFF/source-map edits may
be staged separately with `git add -p`):

```powershell
git add -- src/hlt_classification/operations/oscar_runtime_restore.py scripts/restore_luka_oscar_runtime.py scripts/recover_luka_new_balanced_oscar.sh scripts/queue_luka_new_balanced_oscar.sh tests/test_oscar_runtime_restore.py docs/plans/LUKA_NEW_BALANCED_OSCAR_PLAN.md docs/contracts/LUKA_NEW_BALANCED_OSCAR.md docs/LUKA_NEW_BALANCED_OSCAR_RUNBOOK.md docs/LUKA_NEW_BALANCED_OSCAR_VALIDATION.md
git diff --cached --stat
git commit -m "Restore Oscar runtime in HOME and recover startup-only ladder failure"
git push origin main
git rev-parse HEAD
```

Then paste this on Oscar. The reported 69.42 GB HOME usage leaves enough for the
28 GiB conservative declaration below, provided unrelated writes have not consumed
it in the meantime. If `checkquota` shows less than 28 GiB available, stop before
launching. Downloads/Conda package cache go to fresh scratch directories. Installed
runtime payload is capped at 12 GiB, with 14 GiB outputs plus 2 GiB margin reserved.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_new_balanced_oscar_f9dfc80b"
checkquota
read -r -p 'Pushed 40-character runtime recovery commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
module load miniforge3/25.3.0-3-a6hh
CONDA_BASE="$(conda info --base)"
test -x "$CONDA_BASE/bin/python"
test -x "$CONDA_BASE/bin/conda"
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="${BASE}/source/HLT_Classification_new_balanced_oscar_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/recover_luka_new_balanced_oscar.sh" detach \
    "$COMMIT" 28 "$CONDA_BASE/bin/conda" \
    --authorize AUTHORIZE_NEW_BALANCED_OSCAR_SIX_V1
BASH
```

After `DETACHED runtime recovery PID`, disconnecting or Ctrl-C on the tail does
not stop reconstruction, verification, dry review or six-job submission. The log
is `checkpoints/luka_new_balanced_oscar_<commit8>_r1.runtime-launch/launcher.log`.
Success means `RUNTIME_RESTORED`, six `SUBMITTED` lines, and
`RECOVERY_LAUNCHER_EXIT=0`. DETACHED alone is not proof of completion. A failure
preserves diagnostics under `environments/new_balanced_oscar_v1.restore`; inspect
them rather than deleting or rerunning over an existing prefix. No automatic
package upgrades, prefix deletion, cancellation or dataset changes are performed.

## Explicit continuation after a failed job

Keep the same source, spec and root. First inspect its failure record/log and
the exact saved job IDs. All prior IDs must be terminal, including dependent
jobs; cancel only exact bound pending IDs if needed. Never remove submission
intents or epoch records to force a retry. An ambiguous sbatch response needs
accounting reconciliation before retrying.

With PROJECT_DIR, ROOT, ENV_DIR from above, a confirmed terminal campaign can
be continued using:

```bash
COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
read -r -p 'Currently available HOME GiB: ' QUOTA
bash "$PROJECT_DIR/scripts/launch_luka_new_balanced_oscar.sh" resume-detach \
    "$PROJECT_DIR" /oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1 \
    "$ROOT" "$COMMIT" "$QUOTA" "$ENV_DIR" \
    --authorize AUTHORIZE_NEW_BALANCED_OSCAR_SIX_V1
```

This authenticates completed models, queues only unpublished stages, restores
committed optimizer/RNG state, and finalizes already-trained fits without new
training updates. A runtime/source change or measured resource failure requires
inspection; this command does not silently alter scientific or resource settings.

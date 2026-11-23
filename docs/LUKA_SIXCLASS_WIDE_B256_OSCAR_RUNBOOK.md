# Queue the new-data 200–2600 GeV batch-256 ladder on Oscar

[Scientific plan](plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md) ·
[Contract](contracts/LUKA_SIXCLASS_WIDE_B256_OSCAR.md) ·
[Validation evidence](LUKA_SIXCLASS_WIDE_B256_OSCAR_VALIDATION.md).

Six fresh fits: OFFLINE, M0HLT, CONCAT, UE100, DE050, DE000. All have peak LR
1e-4, batch256, 20 epochs, equal class/pT-bin TRAIN sampling, and CE-only main
classifiers. Teacher embedding/logit branches remain; DE000 adds OFFLINE
contrastive matching. The final model is HLT-only. No standalone preflight job.

## Commit and push from local PowerShell

```powershell
git add -- src/hlt_classification/luka_sixclass_wide_b256_oscar scripts/luka_sixclass_wide_b256_oscar.py scripts/launch_luka_sixclass_wide_b256_oscar.sh scripts/queue_luka_sixclass_wide_b256_oscar.sh scripts/bootstrap_luka_sixclass_wide_b256_oscar.sh sbatch/run_luka_sixclass_wide_b256_oscar.sh sbatch/luka_sixclass_wide_b256_oscar_common.sh tests/test_luka_sixclass_wide_b256_oscar.py docs/plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md docs/contracts/LUKA_SIXCLASS_WIDE_B256_OSCAR.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_RUNBOOK.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_VALIDATION.md
git add -p -- README.md docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Add fresh six-class Oscar ladder at LR1e-4 batch256 and 200-2600 GeV"
git push origin main
git rev-parse HEAD
```

The shared documents already contain unrelated work. Select only this campaign's
additions when staging those files. Local raw data/audit artifacts are not pushed;
Oscar rebuilds and authenticates the population from the verified scratch upload.

## Paste on Oscar

The existing `f9dfc80b` checkout is only the Git source and original runtime
reference. All new model/target outputs go to a fresh campaign root. This block
requires current free HOME quota: **18 GiB** with the verified persistent runtime,
or **32 GiB** if it must also be created. Account for other pending writes.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="${BASE}/source/HLT_Classification_new_balanced_oscar_f9dfc80b"
ENV_DIR="${BASE}/environments/new_balanced_oscar_v1"
checkquota
read -r -p 'Pushed 40-character WIDE-B256 commit: ' COMMIT </dev/tty
read -r -p 'Current free HOME GiB (18 with runtime; otherwise 32): ' QUOTA </dev/tty
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
PROJECT="${BASE}/source/HLT_Classification_sixclass_wide_b256_oscar_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/queue_luka_sixclass_wide_b256_oscar.sh" "$COMMIT" "$QUOTA" "$RUNTIME_CONDA"
BASH
```

The detached launcher verifies the persistent runtime, authenticates scalar
membership and raw-transfer lineage, prints the exact plan, dry-reviews resources,
then submits six science jobs. Runtime/population checks can take several minutes.
`DETACHED` means the launcher started; **six `SUBMITTED` lines and
`BOOTSTRAP_EXIT=0`** mean submission succeeded. Do not launch twice. Ctrl-C from
the attached tail or disconnecting after DETACHED leaves the launcher running.

Log: `/oscar/home/rlyang/hlt_classification/checkpoints/luka_sixclass_wide_b256_oscar_<commit8>_r1.bootstrap/launcher.log`.
Jobs are named `luka6b256_train_<stage>`. OFFLINE/M0HLT/CONCAT may start
independently; later stages depend on all their teachers. Each requests one L40S,
six CPUs, 192000 MiB, 48 hours. This limit is not a runtime estimate. Batch-256
memory/timing qualification happens inside each allocation before fresh fitting.

```bash
squeue --me -o '%.18i %.12P %.45j %.2t %.10M %R'
```

## Read results without queueing an evaluation job

Set the new commit prefix and activate the same explicit environment:

```bash
COMMIT8=REPLACE_WITH_NEW_COMMIT_PREFIX
BASE=/oscar/home/rlyang/hlt_classification
export PROJECT_DIR=${BASE}/source/HLT_Classification_sixclass_wide_b256_oscar_${COMMIT8}
export ENV_DIR=${BASE}/environments/new_balanced_oscar_v1
ROOT=${BASE}/checkpoints/luka_sixclass_wide_b256_oscar_${COMMIT8}_r1
source "$PROJECT_DIR/sbatch/luka_sixclass_wide_b256_oscar_common.sh"
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_wide_b256_oscar.py" progress --spec "$ROOT/campaign_spec.json" --stage DE000
```

Replace DE000 with any stage. Epoch tables include in-range and full-validation
macro AUC, corresponding recovery once fresh anchors publish, full Hbb AUC and
sampling repetition. Anchors are their in-range-selected checkpoints, held fixed
for both populations. Validation is never class/pT resampled.

```bash
"$ENV_DIR/bin/python" -u -B -s "$PROJECT_DIR/scripts/luka_sixclass_wide_b256_oscar.py" results --spec "$ROOT/campaign_spec.json"
```

This prints selected epochs, accuracy/AUC/recovery, R50 and R30 for in-range,
pooled full validation and outside-only diagnostics. Add `--json` for per-bin
metrics, passing-QCD counts and all model recoveries. These commands authenticate
saved report/epoch metadata without inference or binary checkpoint/bank loading.
They also work with a partially completed ladder.

## Explicit recovery after a terminal failure

Inspect its exact log and failure record first. Keep all manifests, submission
intents and epoch states. Do not delete them to force a retry. All earlier job IDs
must be confirmed terminal; ambiguous sbatch receipts require reconciliation.
With the variables above and newly checked current quota:

```bash
COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
checkquota
read -r -p 'Currently free HOME GiB: ' QUOTA
bash "$PROJECT_DIR/scripts/launch_luka_sixclass_wide_b256_oscar.sh" resume-detach \
    "$PROJECT_DIR" /oscar/scratch/rlyang/datasets/luka_qcdrun2com_higgs2p_sb_v1 \
    "$ROOT" "$COMMIT" "$QUOTA" "$ENV_DIR" \
    --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1
```

Only unpublished stages are queued. Committed epochs resume model, optimizer and
RNG exactly. If all 20 epochs committed before export failed, recovery does no
new optimizer updates. No historical jobs/models/data are modified.

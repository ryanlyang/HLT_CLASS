# Queue the new-data 200–2600 GeV batch-256 ladder on Oscar

[Scientific plan](plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md) ·
[Contract](contracts/LUKA_SIXCLASS_WIDE_B256_OSCAR.md) ·
[Validation evidence](LUKA_SIXCLASS_WIDE_B256_OSCAR_VALIDATION.md).

Six fresh fits: OFFLINE, M0HLT, CONCAT, UE100, DE050, DE000. All have peak LR
1e-4, batch256, 20 epochs, equal class/pT-bin TRAIN sampling, and CE-only main
classifiers. Teacher embedding/logit branches remain; DE000 adds OFFLINE
contrastive matching. The final model is HLT-only. No standalone preflight job.

## Recover CONCAT in one 72-hour job (current SPEC/v5)

For failed job9620918, use this section. CONCAT gets one L40S/6CPUs/192000MiB
allocation for72h and prepares its inputs once in RAM. The full20 epochs and
export run in that allocation. The prior conservative estimate was51h45m;
runtime is measured again inside the job. No persistent particle cache is made.
The submitted sequence is CONCAT -> UE100 -> DE050 -> DE000; the latter three
retain48h each. OFFLINE9544070 and M0HLT9544071 are reused.

The supplied norm-gpu limits allow4,320 GPU-minutes and25,920 CPU-minutes for
this request, below5,760 and34,560 respectively. Two GPUs for72h would exceed
the GPU-minute budget. Scheduler dry review still runs before submission.

In local PowerShell, stage the owned recovery and review the staged changes:

```powershell
git add -- src/hlt_classification/luka_sixclass_wide_b256_oscar scripts/luka_sixclass_wide_b256_oscar.py scripts/recover_luka_sixclass_wide_b256_oscar.sh tests/test_luka_sixclass_wide_b256_oscar.py tests/test_luka_sixclass_wide_b256_segments.py tests/test_luka_sixclass_wide_b256_long.py tests/test_luka_sixclass_wide_b256_recovery.py docs/plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md docs/contracts/LUKA_SIXCLASS_WIDE_B256_OSCAR.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_RUNBOOK.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_VALIDATION.md
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Recover CONCAT in one 72-hour allocation with RAM-only inputs"
git push origin main
git rev-parse HEAD
```

Select only this campaign's additions in the shared documents. Then on Oscar:

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="$BASE/source/HLT_Classification_sixclass_wide_b256_oscar_d534fcfe"
checkquota
read -r -p 'Pushed 40-character single-job recovery commit: ' COMMIT </dev/tty
read -r -p 'Currently free HOME GiB (minimum 18): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/source/HLT_Classification_sixclass_wide_long_${COMMIT:0:8}"
ROOT="$BASE/checkpoints/luka_sixclass_wide_long_${COMMIT:0:8}_r1"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/recover_luka_sixclass_wide_b256_oscar.sh" long-detach \
    "$PROJECT" "$ROOT" "$COMMIT" "$QUOTA" \
    --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1
BASH
```

Expect four `SUBMITTED` lines: `train_CONCAT`, `train_UE100`, `train_DE050`,
`train_DE000`, then `LAUNCHER_EXIT=0`. There is no `train_CONCAT_part1` and
no separate preflight. `DETACHED` alone is not submission confirmation.
Ctrl-C only exits the tail view. Keep the old sources, specs, claims and baseline
artifacts. Do not launch the split alternative as well: both share one claim.
If interrupted, use the same-source `resume-detach` launcher documented below;
committed optimizer/model/RNG state is retained, including export-only recovery.

## Recover the walltime estimate stop (SPEC/v4, superseded split option)

The user selected SPEC/v5 above instead. Retain this section for interpreting
the split execution contract; do not use it for the current launch.

Use this for job9620918 in `luka_sixclass_wide_gradient_2bf345f9_r1`.
The batch256 gradient and memory checks passed, but the buffered estimate was
51h45m against48h. This recovery trains CONCAT epochs1--10 and11--20 in separate
dependent allocations, then UE100, DE050 and DE000. OFFLINE/M0HLT are reused.
There are five submitted jobs, four scientific fits, and no separate preflight.
The20-epoch schedule, RNG, optimizer and global checkpoint selection continue
through the boundary. Both halves rebuild inputs; allow that preparation time.

Local PowerShell:

```powershell
git add -- src/hlt_classification/luka_sixclass_wide_b256_oscar scripts/luka_sixclass_wide_b256_oscar.py scripts/recover_luka_sixclass_wide_b256_oscar.sh tests/test_luka_sixclass_wide_b256_segments.py tests/test_luka_sixclass_wide_b256_recovery.py docs/plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md docs/contracts/LUKA_SIXCLASS_WIDE_B256_OSCAR.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_RUNBOOK.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_VALIDATION.md
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Split CONCAT into checkpointed allocations without changing twenty-epoch training"
git push origin main
git rev-parse HEAD
```

Select only this recovery's additions in shared documents. On Oscar:

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="$BASE/source/HLT_Classification_sixclass_wide_b256_oscar_d534fcfe"
checkquota
read -r -p 'Pushed 40-character segmented-recovery commit: ' COMMIT </dev/tty
read -r -p 'Currently free HOME GiB (minimum 18): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/source/HLT_Classification_sixclass_wide_time_${COMMIT:0:8}"
ROOT="$BASE/checkpoints/luka_sixclass_wide_time_${COMMIT:0:8}_r1"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/recover_luka_sixclass_wide_b256_oscar.sh" time-detach \
    "$PROJECT" "$ROOT" "$COMMIT" "$QUOTA" \
    --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1
BASH
```

Expect `SUBMITTED train_CONCAT_part1`, `train_CONCAT`, `train_UE100`,
`train_DE050`, `train_DE000`, then `LAUNCHER_EXIT=0`. `DETACHED` alone does
not confirm submission. Ctrl-C leaves only the tail view. Preserve every earlier
source/spec/claim and the original baseline results; do not launch the old v3
execution. Each half retains48h and repeats integrated measured acceptance.

For an interrupted new segment use the existing `resume-detach` launcher below
with this exact new project/commit/spec. It authenticates terminal accounting,
keeps committed progress and omits a successfully published first segment.
No automatic resubmission or old-job cancellation is performed.

## Recover the failed backward check (historical SPEC/v3)

Use this for failed recovery `44aa1522_r1`, CONCAT9615391, diagnosed by9618095.
The repair keeps native forward/weight gradients and uses the equivalent direct
input gradient for oversized pointwise pair convolutions. It preserves batch256,
full particles, global BatchNorm and the complete scientific recipe. OFFLINE9544070
and M0HLT9544071 are reused. Only CONCAT -> UE100 -> DE050 -> DE000 are submitted.
All previous jobs must be terminal; no old claim or failure record is removed.

Local PowerShell:

```powershell
git add -- src/hlt_classification/luka_sixclass_wide_b256_oscar scripts/luka_sixclass_wide_b256_oscar.py scripts/recover_luka_sixclass_wide_b256_oscar.sh tests/test_luka_sixclass_wide_b256_recovery.py tests/test_luka_sixclass_wide_b256_convolution.py docs/plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md docs/contracts/LUKA_SIXCLASS_WIDE_B256_OSCAR.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_RUNBOOK.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_VALIDATION.md
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Recover CONCAT oversized pointwise input gradients without changing the recipe"
git push origin main
git rev-parse HEAD
```

Select only this repair's additions in the two shared documents. Then on Oscar:

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="$BASE/source/HLT_Classification_sixclass_wide_b256_oscar_d534fcfe"
checkquota
read -r -p 'Pushed 40-character gradient-recovery commit: ' COMMIT </dev/tty
read -r -p 'Currently free HOME GiB (minimum 18): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/source/HLT_Classification_sixclass_wide_gradient_${COMMIT:0:8}"
ROOT="$BASE/checkpoints/luka_sixclass_wide_gradient_${COMMIT:0:8}_r1"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/recover_luka_sixclass_wide_b256_oscar.sh" gradient-detach \
    "$PROJECT" "$ROOT" "$COMMIT" "$QUOTA" \
    --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1
BASH
```

Four `SUBMITTED` lines and `LAUNCHER_EXIT=0` confirm submission. The early
full-support check runs before full resident input construction; it is still
required on the actual L40S. `DETACHED` alone is not confirmation of submission.
Ctrl-C leaves the log view without stopping the detached launcher. Do not also
resume the old failed `44aa1522` execution. Keep its source, diagnostic log and
both original baseline outputs. No final-test inference is performed.
The large-convolution cuDNN warning can still appear because forward execution
remains native; the recorded finite backward checks determine acceptance.

## Recover the November 25 CONCAT memory failure (historical SPEC/v2)

This earlier procedure does not fix the subsequently observed backward failure.
For job9620918's walltime stop use the current SPEC/v5 instructions above.

Use this section for original campaign `d534fcfe_r1`, failed job9544072.
It creates a new SPEC/v2 execution and reuses OFFLINE9544070/M0HLT9544071.
Only CONCAT -> UE100 -> DE050 -> DE000 are submitted. Batch256, the full
particle inputs, draws, LR, losses and twenty epochs are unchanged. CONCAT
stores native pair autograd tensors in CPU RAM. A memory check on three real
maximum-support rows now precedes the hours-long full input construction.

In local PowerShell, stage the owned repair and only its shared-document additions:

```powershell
git add -- src/hlt_classification/luka_sixclass_wide_b256_oscar scripts/luka_sixclass_wide_b256_oscar.py scripts/recover_luka_sixclass_wide_b256_oscar.sh tests/test_luka_sixclass_wide_b256_recovery.py docs/plans/LUKA_SIXCLASS_WIDE_B256_OSCAR_PLAN.md docs/contracts/LUKA_SIXCLASS_WIDE_B256_OSCAR.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_RUNBOOK.md docs/LUKA_SIXCLASS_WIDE_B256_OSCAR_VALIDATION.md
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Recover batch256 CONCAT with lossless pair tensor storage and early memory check"
git push origin main
git rev-parse HEAD
```

Paste on Oscar, entering that newly pushed full commit and current free HOME
quota. Require18GiB for the new outputs; the existing persistent environment is
reused. The last observed HOME usage was65.88GiB; check current usage below.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/oscar/home/rlyang/hlt_classification
REPO="$BASE/source/HLT_Classification_sixclass_wide_b256_oscar_d534fcfe"
checkquota
read -r -p 'Pushed 40-character memory-recovery commit: ' COMMIT </dev/tty
read -r -p 'Currently free HOME GiB (minimum 18): ' QUOTA </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ && "$QUOTA" =~ ^[0-9]+$ ]]
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/source/HLT_Classification_sixclass_wide_memory_${COMMIT:0:8}"
ROOT="$BASE/checkpoints/luka_sixclass_wide_memory_${COMMIT:0:8}_r1"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
bash "$PROJECT/scripts/recover_luka_sixclass_wide_b256_oscar.sh" detach \
    "$PROJECT" "$ROOT" "$COMMIT" "$QUOTA" \
    --authorize AUTHORIZE_SIXCLASS_WIDE_B256_OSCAR_V1
BASH
```

`DETACHED` starts authentication and submission; four `SUBMITTED` lines plus
`LAUNCHER_EXIT=0` confirm it queued. Ctrl-C only exits the attached log view.
Preserve the original source, spec, results and new parent recovery claim.
Read progress/results with the same CLI below, using the new PROJECT/ROOT.
It resolves the preserved baselines automatically. For a later terminal failure
of this new execution, use its same-source `resume-detach` command below.
Never point the old resume command at a new source to bypass source checks.

The longest-row check remains a real L40S acceptance boundary, not a promise
that every resource limit is already measured locally. It adds no extra job.
CPU storage can cost transfer time; the worker measures remaining runtime and
memory before fitting. No historical results or final-test data are changed.

## Original six-fit registration: commit and push from local PowerShell

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

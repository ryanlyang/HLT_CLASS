# Queue the 600k balanced reconstruction screen on Hydra

**Existing dc5920f0 study,2026-11-23:** use the
[normalization recovery commands](LUKA_BALANCED_RECO_NORMALIZATION_RECOVERY_RUNBOOK.md).
Do not repeat the generic full-campaign launch below. The recovery preserves
completed/running models and restores the exact qualified normalization before
new reconstruction attempts. Historical fresh-campaign instructions follow.

**Current next action:** the operator verified Gloo/no-P2P diagnostic219052
COMPLETED/0:0 with saved four-rank OFFLINE parity. Original/Gloo-only profiles
returned incorrect tensor sums. Commit/push the v3 transport repair and use a
fresh study below. It pins NCCL_P2P_DISABLE=1 before Python, uses Gloo control,
and performs a five-minute actual GPU check before full decoding. That check
must pass all9 arm paths before qualification builds inputs. Full real-input
resource qualification still gates scientific release. Preserve old source and
diagnostics; no repeat of the three diagnostics is needed.

This is **18 scientific fits**, two matched seeds,20 epochs each, on **four
GTX1080Ti GPUs per fit**. It trains new OFFLINE/M0HLT/U100 teachers and compares
U100-w5, R24, D20, R20, R22 and R28. Four readouts per trajectory include EMA05
as primary; EMA is not an additional training job. Dataset600–2600GeV, six
classes,600k TRAIN/150k natural validation. See the
[plan](plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md),
[contract](contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA.md) and
[validation](LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md).

This campaign does not mutate scheduler state or reuse old scientific weights,
and does not create a persistent particle cache. For the newly authorized
retirement of old Hydra campaigns, use the
[clean-slate commands](LUKA_HYDRA_CLEAN_SLATE.md) before queuing. The
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
  src/hlt_classification/operations/hydra_reconstruction_gpu_diagnostic.py `
  scripts/luka_balanced_reconstruction_hydra.py `
  scripts/launch_luka_balanced_reconstruction_hydra.sh `
  scripts/queue_luka_balanced_reconstruction_hydra.sh `
  sbatch/luka_balanced_reconstruction_hydra_common.sh `
  sbatch/run_luka_balanced_reconstruction_hydra.sh `
  tests/test_luka_balanced_reconstruction_hydra.py `
  tests/test_luka_balanced_reconstruction_startup.py `
  tests/test_hydra_reconstruction_gpu_diagnostic.py `
  scripts/diagnose_luka_hydra_gpus.ps1 `
  docs/LUKA_HYDRA_GPU_DIAGNOSTIC.md `
  docs/contracts/LUKA_HYDRA_GPU_DIAGNOSTIC.md `
  docs/plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md `
  docs/contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA.md `
  docs/contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA_INPUTS.md `
  docs/contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA_LAUNCH.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_RUNBOOK.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md
git diff --cached --stat
git commit -m "Pin passing Hydra GPU transport and check all arms before decoding"
git push origin HEAD:main
git rev-parse HEAD
```

Use that pushed40-character commit below. If another branch advanced main,
integrate it normally without resetting unrelated work. The remote launch uses
its own detached clean worktree, so other local unstaged work is preserved.

## Queue from your authenticated Hydra Bash terminal

The verified dataset already exists on Hydra. No new upload is needed.
Current `STORAGE_POLICY/v2` needs10GiB initially: own8GiB output allowance plus
2GiB spare, with no old-ladder reservation or old-campaign file dependency.
Declare current whole freeGiB. Registration checks actual filesystem space.
There are no automatic HOME rechecks during
release, later submission, GPU-job startup or training epochs, and no runtime
output cap. A genuinely full filesystem can still cause a write failure. An
explicit manual resume performs its own initial admission check. Node-local
RAM scratch capacity and CPU/GPU resource qualification remain required.

The current allowance is based on the audited4.50GiB complete initialized
inventory. Older sources retained a14GiB ladder allowance and require its old
records. After the clean-slate cleanup, use the new standalone-policy pushed
commit and fresh study, never an old ebd15b7e retry. The historical one-file
cleanup section below records an earlier narrower authorization; do not run
it alongside or after the new clean-slate operation.

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
the four-GPU qualification has12h. GPU qualification measures full support and
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

### Torchrun argument repair after job217775

The input reader completed all three views. U100 took13547.9s; the entire job
ran5h49m28 before torchrun rejected the child worker's `--node` as an ambiguous
prefix of its own node-rank option. No GPU arm proof or scientific fit was run.
This was a launcher-argument error rather than the earlier diagnostic timer.

The corrected command puts `--` before the rank script. The installed PyTorch
parser now checks the generated command before registration's population work
and again before GPU input construction. New specs bind WORKER_LAUNCH_POLICY/v1.
Qualification requests12h, leaving room beyond the observed input preparation;
science48h and all physics/readout/resource checks remain. No separate preflight
job or persistent particle cache is added. Local smoke uses four real CPU worker
processes; actual four-GPU qualification remains required.

Commit/push the revised namespace/tests/plan/contracts/runbook/validation using
the file list above, including the new launch contract, then use its new pushed
commit in the general queue block. No dataset/environment transfer is needed.
Do not relaunch58b9e655 or modify its source/spec. Its temporary input arrays
were removed by the failed job's cleanup, so input construction will run again.
Keep all prior logs, preparation artifacts and submission records intact.

### Reader repair after diagnostic job217200

The53bab0c5 qualification log establishes slow D000 preparation:71433 of750000
rows after2h21m, across57697 tiny spans, then repeated waits on an unfinished
CPU worker until the6h Slurm limit. No scientific fit had started. This is not
an observed GPU-training performance limit or a physics-quality failure.

The revised source batches up to128 selected rows per same-file/role task,
reuses up to128MiB of temporary interpreted ROOT basket buffers per process,
and keeps original authorized final row spans and exact payload/normalization
order. Four CPU workers now have child stack samples,600s submitted-batch
deadlines and pool termination on failure. It preserves six-hour qualification,
48h science, the18-fit recipe, test sealing and initial-only HOME admission.
See the [input contract](contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA_INPUTS.md).

Commit/push the revised package/tests and campaign docs using the file list
above, including the new input contract, then run the general fresh-commit queue
block. No dataset upload is needed. Preserve both failed studies and all other
queued/running jobs; do not edit their source/spec or use old-root resume with
this revision. Qualification must still pass before scientific release.
Watch `input worker BEGIN/END` and row counts in the new qualification log.
If a worker times out, share its own stack samples and the named source/span;
no rows will be silently dropped. Local speed measurements are not a Hydra
runtime guarantee.

Local acceptance:50 tests passed/1 four-GPU skip. A128-selected-TRAIN QCD/Higgs
probe preserved exact features/vectors/targets and improved serial construction
from2.906s to0.453s (6.415x). This is not a full remote timing measurement.
The source snapshot remains source-pinned and qualification still gates release.

### Silent qualification timeout at4a264700

Operator output: prepare216290 completed in5m24, qualify216291 timed out at6h,
release216292 was cancelled, and no scientific fits were released. The GPU
working directory and qualification receipts are absent. The accounting's zero
CPU fields do not identify where startup stopped. Preserve that study unchanged.
The follow-up node SSH probe was reset before authentication, so neither GPU
health nor scratch manifests could be inspected. This is not evidence that the
node is faulty; the next useful action is a fresh-source diagnostic retry.

The diagnostic revision prints `LUKABREC_SHELL`, `LUKABREC_IMPORT` and
`LUKABREC_STAGE` records from shell entry through device/source checks, input
construction, rank loading, parity, miniatures and measurements. Input progress
includes decoded TRAIN/validation row counts. A five-minute gap in progress
dumps a stack sample during startup/probes; that is not a failed scientific
check. The common import check stops after300s plus15s termination grace if
hung. Normal science epochs and a parent waiting on workers do not produce
scheduled stack dumps. Qualification still has6h and science48h; this revision
does not claim to have identified or accelerated the original bottleneck.

Commit/push these changes with the campaign file list above, then use the
general fresh-commit queue block. There are no prior science fits to repeat.
Do not run the old-root resume block with revised source or alter its spec.
Use the new source-derived study root, leaving all other campaigns unchanged.
Watch the new `qualify-<accepted-ID>.out`; its last BEGIN/progress message and
any following stack sample identify the stalled phase. Do not merely increase
the old walltime or remove qualification.

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

## Historical one-file cleanup before the blocked01324411 retry

Superseded for the clean-slate workflow by [these commands](LUKA_HYDRA_CLEAN_SLATE.md).
This section documents the earlier protected-ladder policy; its old cleanup and
launch instructions are not the current clean-slate sequence.

The supplied inventory identifies169,968,554 bytes of derived old TRAIN keys at
`contrast_delta_54473f10/keys/keys.npz`. The revised helper authenticates that
exact payload against both its key manifest and transfer manifest, checks its
TRAIN-only teacher lineage and traces all current jobs through registered specs
and local metadata parents. Any live dependency, unresolved job, checksum error
or redirected path prevents deletion. It audits again immediately before unlink.

Only that one file can be deleted. Retain both manifests, all completed historical
results, raw datasets, source, environments, model checkpoints and predictions.
Old contrastive training/full replay will require restoring or re-exporting this
bank; the retained manifests are not rewritten to pretend the payload exists.
Intent/completion receipts are under
`retired_campaigns/contrast_delta_54473f10_keys_v1`. Do not erase or repeat an
existing operation record. This is an authorized derived-input cleanup, separate
from the new campaign's training and science outputs.

Commit and push the changed reconstruction package/tests and plan/contract/
runbook/validation docs, plus these new helper files and tests:

```powershell
git add -- src/hlt_classification/luka_balanced_reconstruction_hydra `
  tests/test_luka_balanced_reconstruction_hydra.py `
  src/hlt_classification/operations/hydra_contrast_key_retirement.py `
  scripts/retire_luka_hydra_contrast_keys.py `
  tests/test_hydra_contrast_key_retirement.py `
  docs/plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md `
  docs/contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_RUNBOOK.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md
git commit -m "Check HOME space only at launch and retire unused contrastive keys"
git push origin HEAD:main
git rev-parse HEAD
```

In Hydra Bash, use the new pushed commit. Cleanup runs before the detached queue
helper; it will refuse to proceed if it cannot authenticate inactivity. The new
commit makes a fresh study/launch path and preserves the failed01324411 log.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="$BASE/code_65952ed3"
read -r -p 'Pushed 40-character storage-fix commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
git -C "$REPO" fetch origin main
git -C "$REPO" merge-base --is-ancestor "$COMMIT" origin/main
PROJECT="$BASE/code_balanced_reco_${COMMIT:0:8}"
if [[ ! -e "$PROJECT" && ! -L "$PROJECT" ]]; then
    git -C "$REPO" worktree add --detach "$PROJECT" "$COMMIT"
fi
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
/usr/bin/python3 -I -B "$PROJECT/scripts/retire_luka_hydra_contrast_keys.py" \
    --execute --authorize RETIRE_UNUSED_HYDRA_CONTRAST_KEYS_V1
FREE_GIB="$(/usr/bin/python3 -I -B -c \
    'import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free // 2**30)' "$BASE")"
printf 'Currently available: %s whole GiB\n' "$FREE_GIB"
bash "$PROJECT/scripts/queue_luka_balanced_reconstruction_hydra.sh" "$COMMIT" "$FREE_GIB"
BASH
```

For a read-only cleanup preview, run the helper without`--execute --authorize ...`.
If space remains insufficient after cleanup, registration stops before any job
submission. Do not edit the source inside an existing pinned execution checkout.

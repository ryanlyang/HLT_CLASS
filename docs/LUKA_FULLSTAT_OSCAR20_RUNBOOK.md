# Luka FullStat: 3M / 500k regular ladder on Oscar

For the interrupted7486668 preparation array, use the dedicated
[sparse recovery + automatic follow-up runbook](LUKA_FULLSTAT_RECOVERY_RUNBOOK.md).
Do not run the fresh-gate instructions below against the existing campaign.

Authority: [plan](plans/LUKA_FULLSTAT_OSCAR20_PLAN.md) and
[contracts](contracts/LUKA_FULLSTAT_OSCAR20.md). This is a NEW experiment,
not recovery of the old trial or batch128 fusion runs.

Nine fresh fits: M0HLT, OFFLINE, U000, DIRECT_D000, COARSE_U050, COARSE_U100,
COARSE_D066, COARSE_D033, COARSE_D000. Exactly20 full epochs each, warmup2/cosine,
true batch128 for every model, no fusion. Best full-validation AUC checkpoint;
ordinary controls CE-only, KD uses .25CE+.75KL at T2. No automatic extension.

Split: 3,000,000 train /500,000 validation /1,500,000 sealed test /105,325 unused.
Class/QCD-bin proportions and HLT/offline pairings are preserved. The operator
approved the documented seeded block-ordered jet-row split. Event independence
and overlap with the old trial are **unverified**. No test command is provided.

## Storage and transfer

Run `checkquota` on Oscar first. Do not interpret filesystem-wide `df` capacity
as your quota. The local scalar scan measured a conservative **237.75 GiB**
cache-space reservation, including temporary publication and reserve. This is
not a measured final cache size. The raw snapshot is23.66GiB. Keep raw data and
campaign metadata/checkpoints on persistent home storage; prepared ragged
views may use scratch (not an archival copy). Leave additional persistent quota
for models, predictions, TRAIN banks and logs. Nothing is automatically deleted.

From Windows, only if this Oscar destination is NEW:

```powershell
$localData = 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification\artifacts\luka_fullsim_fullstat_20261015'
$remoteData = '/oscar/home/rlyang/datasets/luka_fullsim_fullstat_20261015'
ssh rlyang@ssh.ccv.brown.edu "test ! -e '$remoteData' && mkdir -p -- '$remoteData'"
if ($LASTEXITCODE -ne 0) { throw 'Destination exists or is unavailable. Preserve it and inspect.' }
scp -r -o ServerAliveInterval=30 -o ServerAliveCountMax=6 `
    "$localData/raw" "$localData/source.txt" "$localData/source.sha256" `
    "rlyang@ssh.ccv.brown.edu:$remoteData/"
if ($LASTEXITCODE -ne 0) { throw 'Transfer incomplete. Preserve partial files; do not recreate or delete the destination.' }
```

Use your already working Oscar SSH alias instead if required by your access
setup. Password/MFA are entered interactively, never pasted into scripts.
The CLI below authenticates all sixteen payload checksums after copying.

## Commit only the new workflow

The working tree may also contain unrelated batch128 report-recovery changes.
Do not `git add .` or bundle them into a source-compatible old recovery commit.
Review your changes first. New FullStat files plus the opt-in shared-runner
dispatch can be committed independently:

```powershell
git add -- src/hlt_classification/luka_fullstat `
    src/hlt_classification/jetclass2_delphes/runner.py `
    scripts/luka_fullstat.py scripts/queue_luka_fullstat_oscar.sh `
    scripts/check_luka_fullstat_local_witness.py sbatch/run_luka_fullstat_oscar.sh `
    tests/test_luka_fullstat.py docs/plans/LUKA_FULLSTAT_OSCAR20_PLAN.md `
    docs/contracts/LUKA_FULLSTAT_OSCAR20.md docs/LUKA_FULLSTAT_OSCAR20_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
# These shared docs contain other work: stage only the FullStat additions.
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git diff --cached --check
```

After reviewing the staged diff, commit/push and copy the full40-character hash:

```powershell
git commit -m 'Add FullStat Oscar regular ladder with fixed 20-epoch schedule'
if ($LASTEXITCODE -ne 0) { throw 'Inspect commit status before continuing' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Oscar: fresh pinned gate

Run the following **one complete block at a time** on an Oscar login node. The
actual Git repository is under `source/HLT_Classification`, not its parent.
No science jobs are submitted by this section.

```bash
read -r -p 'Pushed 40-character FullStat commit: ' COMMIT
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid commit'; false; }
export COMMIT
export MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
export PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_${COMMIT:0:8}"
export ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_3m500k_20e_${COMMIT:0:8}_r1"
export CACHE_ROOT="/oscar/scratch/rlyang/hlt_classification/caches/luka_fullstat_3m500k_20e_${COMMIT:0:8}_r1"
export CONTAINER=/oscar/home/rlyang/datasets/luka_fullsim_fullstat_20261015
```

```bash
(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -d "${MAIN}/.git"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" verify-transfer --container "${CONTAINER}"
)
```

Stop on errors; preserve everything. After successful verification and quota
inspection, create a fresh gate and full dry plan:

```bash
(
set -euo pipefail
checkquota
read -r -p 'Currently available scratch quota in whole GiB (not total filesystem space): ' QUOTA_GIB
[[ "${QUOTA_GIB}" =~ ^[0-9]+$ ]]
test "${QUOTA_GIB}" -ge 238
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" create \
    --container "${CONTAINER}" --root "${ROOT}" --cache-root "${CACHE_ROOT}" \
    --commit "${COMMIT}" --quota-gib "${QUOTA_GIB}" --acknowledge-row-split --batch-size 128
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" submit --spec "${ROOT}/gate_spec.json"
)
```

Review the printed plan: three individual jobs and one16-element array
(19 scheduled tasks): scalar foundation, prepare0--15%4 (4CPUs/32GB each),
assemble, GPU preflight (L40S/6CPUs/180000MiB/8h). CPU jobs use `batch/default`;
GPU uses `gpu/default/norm-gpu`. The helper checks each exact resource shape
with `sbatch --test-only` before submission. If permissions/partition/QOS differ,
stop and register a corrected site policy; do not silently edit a saved plan.

Paste the **plan** content_hash, not the spec hash, then submit:

```bash
(
set -euo pipefail
read -r -p 'Reviewed GATE plan content_hash: ' PLAN_HASH
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit the reviewed gate only? Type YES: ' CONFIRM
test "${CONFIRM}" = YES
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" submit \
    --spec "${ROOT}/gate_spec.json" --execute --plan-hash "${PLAN_HASH}" \
    --authorize 'AUTHORIZE LUKA FULLSTAT OSCAR20 EXACT PLAN'
)
```

The helper returns Slurm IDs in its live ledger. A lost SSH reply is not proof
of failed submission: inspect `gate_submission/submission_ledger.json` and the
journal before retrying. An exclusive claim blocks ambiguous partial retries.
Do not delete claims, directories, receipts or locks to force a retry.

## After all gate jobs complete: science

The preflight tests all seven views at true128, longest TRAIN backward and
separate longest-validation inference, installed-Weaver parity, a common
representative miniature and checkpoint/probability-bank readback. It records
full-cache reservations and conservative runtime projections, not a full3M
training epoch. Bad model accuracy cannot fail this gate; corrupt data, missing
lineage, insufficient memory or a request exceeding48h can.

Admission authenticates **every gate array element**, the registered preflight
job, cache, source and payloads. It writes a NEW immutable science spec once:

```bash
(
set -euo pipefail
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" admit --gate "${ROOT}/gate_spec.json"
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" submit \
    --spec "${ROOT}/science/campaign_spec.json"
)
```

If science already exists, preserve it; do not call `admit` again. Review the
new16-job science plan and its measured walltimes, then:

```bash
(
set -euo pipefail
read -r -p 'Reviewed SCIENCE plan content_hash: ' PLAN_HASH
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit all 16 reviewed science jobs? Type YES: ' CONFIRM
test "${CONFIRM}" = YES
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" submit \
    --spec "${ROOT}/science/campaign_spec.json" --execute --plan-hash "${PLAN_HASH}" \
    --authorize 'AUTHORIZE LUKA FULLSTAT OSCAR20 EXACT PLAN'
)
```

Controls and U000 may run concurrently when quota allows; KD tasks wait for the
correct selected teacher's TRAIN bank via afterok dependencies. No scientific
fit reopens ROOT particle arrays or rebuilds its views. No automatic science
followup/controller is armed, and there is no final-test job in this workflow.

## Progress and saved results

```bash
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
sacct -u "$USER" -S now-3days -X -n -P -o JobID,JobName%60,State,ExitCode,Elapsed
bash "${PROJECT}/scripts/queue_luka_fullstat_oscar.sh" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

Partial results show pending rows; completed ones show selected epoch, accuracy,
macro AUC, recoveries and per-class R50. These are development-validation results,
not test results. `science/train_NODE/training_report.json` retains all20 epoch
metrics; `validation_predictions.npz` retains selected predictions for later
read-only diagnostics. Intermediate offline-containing models remain oracles.

## Local verification versus cluster acceptance

The local full-snapshot scalar/checksum/split replay and bounded TRAIN-only
real-data witness live under ignored `artifacts/luka_fullstat_local_validation_20261016`.
`scripts/check_luka_fullstat_local_witness.py` can reproduce a fresh small witness
with a verified foundation. Its local Weaver parity is not Oscar admission,
does not establish full-batch memory, and never accesses test particles.
Do not copy its zero-parent diagnostic foundation into a production gate.

Focused tests: `tests/test_luka_fullstat.py`. Historical training-recipe tests
must also remain green. Local verification:112 combined regression tests plus
one independent longer-validation-support stress test passed; real128-jet TRAIN
witness and local Weaver CUDA parity passed. Production readiness still requires the real Oscar
gate, exact pushed clean source and explicit reviewed science submission.

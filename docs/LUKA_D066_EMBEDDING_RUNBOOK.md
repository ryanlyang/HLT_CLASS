# Frozen D066 embedding prediction, weight 5, SPORC

[Scientific plan](plans/LUKA_D066_EMBEDDING_PLAN.md) and
[contract](contracts/LUKA_D066_EMBEDDING.md) define this additive experiment.
One fresh HLT model, CE + 5 times auxiliary prediction loss, **no logit KD**.
The frozen target is LOGIT_D066, not ATTN_D066 or pure OFFLINE. Same authenticated
299234/74895 development split, batch128, fixed30, optimizer and initialization
as OFFLINE-target w5. No independent test remains. The original50k report uses
the same full-validation-selected model. The prediction head is removed at inference.

Prerequisites: retained LOGIT_D066 selected.pt, original reports/predictions and
source; completed OFFLINE-target w5 receipt; surviving D066 TRAIN/validation and
D000 TRAIN/validation caches. Deleted old attention banks are not required.
Presence alone is insufficient: registration authenticates provenance, and
the GPU worker reproduces teacher predictions before target export.
Reserve at least 2 GiB; no particle caches, attention banks or file deletions.

## Commit/push locally

Use PowerShell in the repository. Stage only the new experiment files; do not
include unrelated dirty work. These commands do not stage the living HANDOFF or
LEGACY_SOURCE_MAP since both may contain unrelated edits; stage their new sections
separately with `git add -p` if desired. Never force-push on a rejection.

```powershell
& {
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { throw 'Existing staged changes: inspect them before staging this experiment' }
git add -- src/hlt_classification/luka_d066_embedding scripts/luka_d066_embedding.py scripts/queue_luka_d066_embedding.sh scripts/queue_luka_d066_embedding_direct.sh sbatch/run_luka_d066_embedding.sh tests/test_luka_d066_embedding.py tests/test_luka_d066_embedding_priority.py tests/test_luka_d066_embedding_direct.py docs/plans/LUKA_D066_EMBEDDING_PLAN.md docs/contracts/LUKA_D066_EMBEDDING.md docs/LUKA_D066_EMBEDDING_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
$Confirm = Read-Host 'Commit and push only these reviewed experiment changes? Type YES'
if ($Confirm -ne 'YES') { throw 'Stopped; staged changes preserved' }
git commit -m "Allow direct tier3 D066 embedding fit using accepted resource evidence"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before continuing' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push' }
git rev-parse HEAD
}
```

## Current request: direct tier3 run, no new preflight

If registration at9a6163b7 stopped with the generic "Commit complete additive
D066 implementation" error, the guard rejected the separate FullStat U100
package. The source integration fix pins that package's exact reviewed tree;
it does not disable source checks. The failure occurred before direct root
publication or job submission. Commit/push the fix and use a fresh worktree
with the new commit below. The old21944417 cancellation check already passed
on this path: do NOT repeat cancellation. Nothing needs deleting/rebuilding,
and the retry still has no new preflight.

Use `queue_luka_d066_embedding_direct.sh`, NOT the older automatic preflight
helper. It submits exactly one `lukad6e_direct` job. That allocation authenticates
the frozen D066 teacher, exports its compact TRAIN targets, trains the fresh
HLT model for30 epochs, and writes comparisons. It does NOT run a stress suite,
probe epoch or second queued job. Prior genuine tier3 sweep evidence and the
completed matched weight5 run are authenticated, not rerun. Missing/corrupt
evidence fails closed. Native teacher readback and TRAIN alignment remain
necessary setup, not a hidden preflight. No teacher retraining is performed.

The request is at least6h (or measured w5 request+2h if larger), not an expected
duration; max24h. This allows setup and queue/node variability without silently
shortening the30-epoch schedule. Reserve2GiB. Leave existing debug sweep holds
alone. This job is not guaranteed priority over other tier3 users.

First push the updated source and retire only the still-pending21944417 using
the cancellation block in the next section. If it has started/completed, stop
and inspect instead; the direct command requires CANCELLED with zero elapsed.
Then run on sporcsubmit:

```bash
bash <<'BASH'
set -euo pipefail
MAIN=/home/ryreu/atlas/HLT_Classification
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_d066direct_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_d066_embedding_direct.sh"
BASH
```

Paste the **value** printed as DIRECT_PLAN_HASH, then YES. Inspect the printed
submission log for the single job ID. No controller or manual follow-up is
required. Do not also submit the legacy gate. Results use the same CLI:
`luka_d066_embedding.py results --spec <direct-root>/campaign_spec.json` in the
same pinned worktree/environment. Training logs show each epoch/best AUC.

## Retire pending debug job (shared prerequisite; no other jobs changed)

Do not change `Partition` using scontrol: old source explicitly checks debug.
Use a fresh tier3 registration after retiring only the still-pending job below.
This does not retrain LOGIT_D066 and does not change the experiment. Preserve
the old root and all its receipts. The old automatic controller exits when it
observes CANCELLED; it cannot queue a fit from a cancelled gate.

After pushing the updated source, first inspect and cancel **only21944417** on
sporcsubmit. The hold/recheck prevents cancelling a job that starts during the
inspection; a racing start is left running and requires a new decision.

```bash
bash <<'BASH'
set -euo pipefail
JOB=21944417
EXPECTED='ryreu|debug|lukad6e_preflight|PENDING'
BEFORE="$(squeue -h -j "${JOB}" -o '%u|%P|%j|%T')"
if [[ "${BEFORE}" != "${EXPECTED}" ]]; then
    printf 'STOP: not the expected pending preflight: %s\n' "${BEFORE}"
    exit 1
fi
read -r -p 'Cancel ONLY pending D066 preflight21944417 for tier3 replacement? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
scontrol hold "${JOB}"
AFTER="$(squeue -h -j "${JOB}" -o '%u|%P|%j|%T')"
if [[ "${AFTER}" != "${EXPECTED}" ]]; then
    scontrol release "${JOB}"
    echo 'STOP: job changed state; hold released. Inspect before continuing.'
    exit 1
fi
scancel "${JOB}"
sacct -j "${JOB}" -X -P -o JobID,JobName%40,State%40,ExitCode,Elapsed
BASH
```

Require `CANCELLED` and `00:00:00` before the replacement. If accounting is
delayed, rerun only the read-only sacct command. Do not rerun cancellation or
queue another run when status is uncertain. If the job is already running or
completed, stop here and inspect its controller/receipts instead.

For the current direct request use the block above. Only if explicitly choosing
the older **two-job preflight route**, use the queue block below, setting its helper invocation to:

```bash
SITE_NAME=sporc_a100 ORDER_PENDING=0 REPLACES_PREFLIGHT_JOB=21944417 \
    bash "${PROJECT}/scripts/queue_luka_d066_embedding.sh"
```

The replacement guard rechecks the exact cancelled ID/name/owner/account/debug
partition/zero elapsed time in sacct before registration. Missing, ambiguous,
running or completed accounting stops without submitting anything. It never
cancels a job itself. The helper creates `luka_d066_embedding_<commit>_tier3_r1`, dry-reviews one
tier3 preflight and one automatic tier3 fit, then requires the exact plan hash
and YES. Wait for `ARMED: tier3 preflight ...` in its log. The five old debug
sweep holds remain untouched and are not automatically released. Tier3 does
not guarantee an earlier start than debug; it is a different eligible queue.

## Queue a fresh run on sporcsubmit

Wait until existing `lukaesd_w5` finishes: the new setup authenticates its selected
predictions for the matched comparison. Do not stop w5 or w20. Paste the new
pushed full commit into this block; do not modify old worktrees.

```bash
bash <<'BASH'
set -euo pipefail
MAIN=/home/ryreu/atlas/HLT_Classification
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_d066embed_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
SITE_NAME=sporc_a100_debug ORDER_PENDING=0 bash "${PROJECT}/scripts/queue_luka_d066_embedding.sh"
BASH
```

The helper shows a full dry review. Paste the **64-character value**, not the
literal words FOLLOWUP_PLAN_HASH. Confirm the two jobs with YES. The helper
defaults to leaving all existing jobs/holds alone. For debug only, optional
`ORDER_PENDING=1` additionally asks HOLD FIVE and is valid only when none of
the selected pending jobs is already held. Only exact journal-bound IDs
21940433..21940437 (w50, w100,
w200, w500, w1000) are eligible for holds, and only while pending. Any already
running work is left alone. Confirm ARMED in the printed log before logging out.
The detached submit-host controller uses no compute allocation. It waits for
successful accounting AND authenticated technical/resource evidence, then queues
one fresh measured fit on the same registered partition. In optional ordering
mode it releases only its acknowledged holds when that fit terminates,
or on a handled controller/preflight error. No blind retries; no cancellation.

This orders the selected jobs, not other users or reservations. Slurm does not
offer an atomic pending-only hold: the controller checks before and after and
releases its own hold if a job starts during that narrow race. A hold does not
suspend running computation. See [Slurm scontrol documentation](https://slurm.schedmd.com/scontrol.html).
Do not independently hold/release those same five IDs while the controller owns
them; Slurm user holds have no per-controller ownership token.

## Monitoring and results

Set PROJECT and ROOT to the paths printed above, in the same activated
atlas_kd_sporc environment with `PYTHONPATH=${PROJECT}/src`:

```bash
squeue --me -o "%.18i %.12P %.40j %.2t %.10M %R"
tail -n 40 "${ROOT}"/auto-followup.*.log
python -u -B -s "${PROJECT}/scripts/luka_d066_embedding.py" results --spec "${ROOT}/science/campaign_spec.json"
```

Completed results compare M0HLT, OFFLINE, OFFLINE_EMBED_W5, LOGIT_D066 (oracle),
and HLT_D066_EMBED_W5, including both AUC deltas and recovery. Logs print every
epoch, current/best validation AUC and losses; provisional logs are not accepted
completed results. Poor performance does not fail admission or shorten fixed30.

## Interrupted setup / stranded holds

An interrupted prompt does not authorize any submission. Preserve the existing
root and review `followup/plan.json`, then rerun only the CLI `auto` command with
the same saved plan, NOT `create`. After a live claim exists, do not rerun auto or
manually submit science. Inspect `followup/failure.json`, exact submission journals,
and `followup/priority/*`. A lost sbatch/scontrol response requires manual review.

After a controller dies, recover only its acknowledged holds using the same
project/environment. The recovery command takes the same advisory lock and fails
if a live controller still owns it. It does not submit, cancel, or rerun anything:

```bash
python -u -B -s "${PROJECT}/scripts/luka_d066_embedding.py" release-holds --spec "${ROOT}/gate_spec.json" --confirm-release-owned
```

SIGKILL, host loss, accounting outages or a job purged from `scontrol` can require
manual exact-ID inspection. Holds with an intent but no acknowledged response
are deliberately not retried/released automatically. Never use a broad name,
partition or user cancellation; Fresh_check and tier3 are out of scope.

## Local and remote evidence

Focused tests cover production math, full fixed30/native deployment readback,
teacher/target provenance, normalization, exact submission journals and queue
failure paths. Synthetic small-population CUDA evidence is not a measured SPORC
admission; the new real debug preflight remains mandatory before science.

Local evidence (2026-10-21): 128-test expanded donor/new/archive/controller
regression passed; final new-experiment suite40passed/1Linux-only skip. The
production file-lock function was separately exercised successfully on local
WSL Linux, including concurrent exclusion and post-exit reacquisition. Bash,
PowerShell and CLI syntax/help checks passed. No real SPORC job was submitted
by the implementation or tests.

The tier3 execution amendment changes orchestration only. Scientific training,
target export, preflight measurements and native inference code remain unchanged.
See HANDOFF for the focused two-site tests; genuine tier3 acceptance still runs
inside the newly queued preflight, not in local fixtures.

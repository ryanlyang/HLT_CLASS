# Frozen D066 embedding prediction, weight 5, debug

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
Presence alone is insufficient: registration authenticates provenance, and the
new genuine GPU preflight reproduces teacher predictions before target export.
Reserve at least 2 GiB; no particle caches, attention banks or file deletions.

## Commit/push locally

Use PowerShell in the repository. Stage only the new experiment files; do not
include unrelated dirty work. These commands do not stage the living HANDOFF or
LEGACY_SOURCE_MAP since both may contain unrelated edits; stage their new sections
separately with `git add -p` if desired. Never force-push on a rejection.

```powershell
git add -- src/hlt_classification/luka_d066_embedding scripts/luka_d066_embedding.py scripts/queue_luka_d066_embedding.sh sbatch/run_luka_d066_embedding.sh tests/test_luka_d066_embedding.py tests/test_luka_d066_embedding_priority.py docs/plans/LUKA_D066_EMBEDDING_PLAN.md docs/contracts/LUKA_D066_EMBEDDING.md docs/LUKA_D066_EMBEDDING_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
git commit -m "Add frozen D066 embedding weight5 debug comparison"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect before continuing' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; do not force-push' }
git rev-parse HEAD
```

## Queue on sporcsubmit

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
bash "${PROJECT}/scripts/queue_luka_d066_embedding.sh"
BASH
```

The helper shows a full dry review. Paste the **64-character value**, not the
literal words FOLLOWUP_PLAN_HASH. Confirm the two jobs with YES, then separately
confirm HOLD FIVE. Only exact journal-bound IDs 21940433..21940437 (w50, w100,
w200, w500, w1000) are eligible for holds, and only while pending. Any already
running work is left alone. Confirm ARMED in the printed log before logging out.
The detached submit-host controller uses no compute allocation. It waits for
successful accounting AND authenticated technical/resource evidence, then queues
one fresh measured fit. It releases acknowledged holds when that fit terminates,
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

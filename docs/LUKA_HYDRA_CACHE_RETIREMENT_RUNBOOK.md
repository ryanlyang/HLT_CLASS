# Retire the completed baseline cache and resume balanced Hydra

This is the authorized recovery for `new_balanced_4a61a34c_r1` after its
completed OFFLINE fit and storage failures in M0HLT/CONCAT. It removes about
16 GiB of prepared baseline inputs and preserves the baseline's results and raw
data. The user requested cleanup first, so the command below submits no jobs.
The separate recovery action can later queue the five unfinished models. See the
[operational contract](contracts/LUKA_HYDRA_CACHE_RETIREMENT.md).

The previous cleanup already removed 5.41 GiB of old diagnostic parity
snapshots/browser cache, with the user reporting 14 GiB free afterward. Do not
repeat that cleanup. Its record is
`retired_campaigns/diagnostic_parity_cleanup_3tinsez0/intent.json`.

## Local PowerShell: commit only this change

```powershell
git add -- src/hlt_classification/operations/hydra_cache_retirement.py `
  scripts/luka_hydra_cache_retirement.py scripts/launch_luka_hydra_cache_recovery.sh `
  sbatch/run_luka_hydra_cache_recovery.sh tests/test_luka_hydra_cache_retirement.py `
  docs/contracts/LUKA_HYDRA_CACHE_RETIREMENT.md docs/LUKA_HYDRA_CACHE_RETIREMENT_RUNBOOK.md `
  docs/plans/LUKA_NEW_BALANCED_HYDRA_PLAN.md
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Retire completed Hydra baseline cache and resume balanced fits"
git push
git rev-parse HEAD
```

Other chats have unrelated local edits. Select only this operation's shared
documentation hunks; review the staged files before committing.

## Hydra: paste this cleanup-only block once

Enter current free home quota in **whole GiB rounded down**, before cleanup.
The declaration remains a conservative cap after cleanup; filesystem headroom
is rechecked. The last reported 14 GiB would cover the unfinished output budget
if it has not fallen. Source retrieval and authentication must succeed before
any deletion. Hashing the raw sources and prepared inputs can take several
minutes; this happens in a detached process.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="${BASE}/code_65952ed3"
df -h "${BASE}"
read -r -p 'Pushed 40-character cache cleanup commit: ' COMMIT </dev/tty
read -r -p 'Current free HOME quota, whole GiB rounded down: ' FREE_GIB </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${FREE_GIB}" =~ ^[0-9]+$ ]]
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/code_cache_recovery_${COMMIT:0:8}"
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
bash "${PROJECT}/scripts/launch_luka_hydra_cache_recovery.sh" detach-cleanup \
    "${PROJECT}" "${COMMIT}" "${FREE_GIB}"
BASH
```

After **DETACHED**, you may disconnect. Ctrl-C stops the log view only.
`CLEANUP_COMPLETE` and `LAUNCHER_EXIT=0` confirm completed cleanup. There should
be no `SUBMITTED` lines. The helper preserves models, reports, predictions,
epoch states, raw data and cache metadata; it only unlinks registered blocks.

```bash
tail -n 60 -f /cs/home/rlyang/hlt_u100_pairs/runs/new_balanced_4a61a34c_r1.cache-cleanup-launch/launcher.log
```

```bash
squeue --me -o '%.18i %.40j %.2t %.10M %R'
df -h /cs/home/rlyang/hlt_u100_pairs
```

## If it stops

Preserve the launcher log, `operations/baseline_cache_retirement_v1/` and the
original `submission/attempt_*` journals. Do not relaunch `detach`, delete claims,
change source in place or use the old balanced launcher after retirement.
An explicitly reviewed cleanup continuation uses launcher mode
`continue-cleanup` with the same project, commit and quota declaration.
It rechecks original accepted
jobs and refuses live or ambiguous attempts. A partial cleanup with the exact
saved intent can finish; a missing file without that intent stops.

The operation leaves every model, checkpoint, prediction and report intact.
It also retains `cache.json`, preparation intent and per-file cache receipts.
They document retired inputs, not a usable live cache. Raw data can rebuild
those inputs into a fresh directory if needed later.

When the user requests campaign resubmission, use the separate `continue-run`
mode with this same operational checkout, commit and recorded quota declaration.
That action rechecks the retired-cache receipt, original journals and all
completed stages, performs scheduler resource review, then queues only the
unfinished fits. It retains one L40S, six CPUs, 192000 MiB and 48 hours per job,
and the original integrated acceptance checks. No separate preflight job.
The combined `detach` mode remains available for a fresh retirement-plus-recovery
operation, but must not be invoked after the cleanup-only claim already exists.

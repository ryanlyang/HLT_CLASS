# Resume balanced Hydra after completed baseline-cache cleanup

This is the authorized recovery for `new_balanced_4a61a34c_r1` after its
completed OFFLINE fit and storage failures in M0HLT/CONCAT. Cleanup already
succeeded under `68dc0cb6eeb0f1f527bbeba11f4753b84789bac7`: the user reported
15.68 GiB freed,28.30 GiB available and `LAUNCHER_EXIT=0`. Do not repeat cleanup.
The command below registers current headroom and queues the five unfinished
models. It preserves the original campaign and cleanup records. See the
[operational contract](contracts/LUKA_HYDRA_CACHE_RETIREMENT.md).

The previous cleanup already removed 5.41 GiB of old diagnostic parity
snapshots/browser cache, with the user reporting 14 GiB free afterward. Do not
repeat that cleanup. Its record is
`retired_campaigns/diagnostic_parity_cleanup_3tinsez0/intent.json`.

## Local PowerShell: commit only this change

```powershell
git add -- src/hlt_classification/operations/hydra_cache_retirement.py `
  scripts/luka_hydra_cache_retirement.py scripts/launch_luka_hydra_cache_recovery.sh `
  sbatch/run_luka_hydra_balanced_resume.sh tests/test_luka_hydra_cache_retirement.py `
  docs/contracts/LUKA_HYDRA_CACHE_RETIREMENT.md docs/LUKA_HYDRA_CACHE_RETIREMENT_RUNBOOK.md `
  docs/plans/LUKA_NEW_BALANCED_HYDRA_PLAN.md
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Resume balanced Hydra using authenticated post-cleanup headroom"
git push
git rev-parse HEAD
```

Other chats have unrelated local edits. Select only this operation's shared
documentation hunks; review the staged files before committing.

## Hydra: paste this resume block once

Enter current free home quota in **whole GiB rounded down**. Enter28 if the
reported28.30 GiB is still current. The new declaration belongs to a separate
resume record; it does not overwrite the pre-cleanup declaration. Actual free
space is checked again at submission and every worker entry. Existing source,
cleanup evidence and completed OFFLINE artifacts are authenticated before
submission, which can take a few minutes in the detached process.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
REPO="${BASE}/code_65952ed3"
df -h "${BASE}"
read -r -p 'Pushed 40-character balanced recovery commit: ' COMMIT </dev/tty
read -r -p 'Current free HOME quota, whole GiB rounded down: ' FREE_GIB </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${FREE_GIB}" =~ ^[0-9]+$ ]]
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="${BASE}/code_balanced_resume_${COMMIT:0:8}"
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
bash "${PROJECT}/scripts/launch_luka_hydra_cache_recovery.sh" detach-resume \
    "${PROJECT}" "${COMMIT}" "${FREE_GIB}"
BASH
```

After **DETACHED**, you may disconnect. Ctrl-C stops the log view only.
Five `SUBMITTED` lines and `LAUNCHER_EXIT=0` confirm submission. Expected jobs:
M0HLT and CONCAT independently, then UE100, DE050 and DE000 with their original
teacher dependencies. OFFLINE is reused. Each keeps one L40S, six CPUs,
192000 MiB and48 hours. Sampling, losses, batch size and20 epochs stay fixed.
There is no new preflight job; existing integrated acceptance remains active.

```bash
tail -n 60 -f /cs/home/rlyang/hlt_u100_pairs/runs/new_balanced_4a61a34c_r1.post-cache-resume-launch/launcher.log
```

```bash
squeue --me -o '%.18i %.40j %.2t %.10M %R'
df -h /cs/home/rlyang/hlt_u100_pairs
```

## If it stops

Preserve the launcher log, both `operations/baseline_cache_retirement_v1/`
and `operations/post_cache_resume_v1/`, and original `submission/attempt_*`
journals. Do not repeat `detach-resume`, delete claims, edit an old source
checkout/spec, or use the original balanced launcher after retirement.
An explicitly reviewed continuation uses `continue-resume` with this new
project, commit and quota declaration. It refuses live or ambiguous previous
submissions. A different quota/source must not overwrite an existing record.

The operation leaves every model, checkpoint, prediction and report intact.
It also retains `cache.json`, preparation intent and per-file cache receipts.
They document retired inputs, not a usable live cache. Raw data can rebuild
those inputs into a fresh directory if needed later.

Keep `code_cache_recovery_68dc0cb6` available: the new continuation authenticates
and loads that immutable cleanup validator. It runs science from the original
`code_new_balanced_4a61a34c` checkout. The new source supplies only the bound
storage/submission adapter. No cache deletion or rebuild occurs on this path.

The original `continue-run` mode still exists for its old frozen quota. It is
not the post-cleanup28 GiB path. The old `detach`/`detach-cleanup` actions must
not be repeated for this already retired cache.

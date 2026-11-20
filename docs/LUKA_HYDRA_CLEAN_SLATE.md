# Keep the new Hydra study and retire prior campaigns

The new600k reconstruction study needs the new dataset/foundation, the whole
6.1GiB runtime/interpreter, a clean current source checkout and shared Git
history. It trains fresh teachers and needs no old300k data or old model weights.

This cleanup deletes old campaign models and predictions, including previous
full-size new-dataset baseline/ladder results, old inputs and pushed clean old
worktrees. Compact text reports/logs are archived first; binaries need another
copy or retraining to restore. It keeps every new reconstruction study/log,
retirement record, unknown directory and uncommitted source change. See the
[exact contract](contracts/LUKA_HYDRA_CLEAN_SLATE.md).

The revised campaign uses STORAGE_POLICY/v2:10GiB initial allowance, with no
dependency/reservation for the old ladder. No automatic per-job/epoch HOME
rechecks return. Science, reader repair,18 fits and four-GPU qualification stay
unchanged. Do not run either older one-file/stability cleanup alongside this one.

## 1. Commit and push from local PowerShell

Review any already staged work. These paths exclude the shared dirty README,
HANDOFF and legacy-source map, which may be committed separately by hunk.

```powershell
git diff --cached --stat
git add -- src/hlt_classification/luka_balanced_reconstruction_hydra `
  scripts/queue_luka_balanced_reconstruction_hydra.sh `
  tests/test_luka_balanced_reconstruction_hydra.py `
  src/hlt_classification/operations/hydra_clean_slate.py `
  scripts/cleanup_luka_hydra_clean_slate.ps1 tests/test_hydra_clean_slate.py `
  docs/contracts/LUKA_HYDRA_CLEAN_SLATE.md docs/LUKA_HYDRA_CLEAN_SLATE.md `
  docs/plans/LUKA_BALANCED_RECONSTRUCTION_HYDRA_PLAN.md `
  docs/contracts/LUKA_BALANCED_RECONSTRUCTION_HYDRA.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_RUNBOOK.md `
  docs/LUKA_BALANCED_RECONSTRUCTION_HYDRA_VALIDATION.md
git diff --cached --stat
git commit -m "Make reconstruction campaign standalone and retire old Hydra workspace"
git push origin HEAD:main
$CleanSlateCommit=(git rev-parse HEAD).Trim()
```

## 2. Run cleanup from that same local PowerShell window

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\cleanup_luka_hydra_clean_slate.ps1 -Commit $CleanSlateCommit
```

Use your Brown key passphrase when prompted. The launcher fetches the pushed
commit, creates the fresh kept worktree, uploads one SHA256-checked helper and
detaches deletion. It checks queue/runtime/source before archiving or deletion.
There is no full recursive size/hash scan of old binary payloads. Metadata
archiving and removing many small files can still take time.

The output prints KEEP/RETIRE/RETAIN, then ARCHIVE VERIFIED, individual RETIRED
paths and COMPLETE with actual free space. Ctrl+C ends only the log viewer.
Do not queue anything until COMPLETE. A STOP means inspect the log and any
`retired_campaigns/new_dataset_clean_slate_v1` intent/journal before retrying.
The script neither submits nor cancels jobs. No remote deletion has been
performed by local tests. Optional `-Preview` lists targets without deleting.

## 3. Queue the new campaign after COMPLETE, in Hydra Bash

Paste the same pushed40-character commit. The new source-derived r1 root is
fresh; failed older roots/logs remain.

```bash
bash <<'BASH'
set -euo pipefail
BASE=/cs/home/rlyang/hlt_u100_pairs
read -r -p 'Pushed clean-slate campaign commit: ' COMMIT </dev/tty
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
PROJECT="$BASE/code_balanced_reco_${COMMIT:0:8}"
test -f "$BASE/retired_campaigns/new_dataset_clean_slate_v1/complete.json"
test "$(git -C "$PROJECT" rev-parse HEAD)" = "$COMMIT"
FREE_GIB="$(/usr/bin/python3 -I -B -c \
  'import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free // 2**30)' "$BASE")"
printf 'Available: %s whole GiB\n' "$FREE_GIB"
bash "$PROJECT/scripts/queue_luka_balanced_reconstruction_hydra.sh" "$COMMIT" "$FREE_GIB"
BASH
```

Bootstrap queues prepare, genuine four-GPU qualification and release. Only
successful technical acceptance releases the18 scientific fits and report.
This operation has not established remote speed, GPU parity or physics results.

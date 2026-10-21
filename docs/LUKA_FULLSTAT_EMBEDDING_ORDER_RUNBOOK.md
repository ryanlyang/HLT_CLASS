# Order the already-armed Oscar FullStat embedding sweep

Only for the existing **fd687181** sweep with no science directory, after the
original controller has been verified and paused on **login010**. Keep preflight
**7690397** running. Do not resume PID300411 or submit science manually.

This preserves the same original source, accepted preflight, targets, seeds and
20-epoch fits. Scheduling is serial: **50 -> 500 -> 1000 -> control -> 100 -> 10
-> 2**. The control's recovery denominator becomes available after the fourth fit.
If one fit fails, the next independent fit may still run; aggregation requires
all seven to succeed. Ordinary SSH disconnects do not stop the detached helper,
but a controller/host failure needs inspection, never blind retry.

## 1. Commit/push from Windows

Stage these additions only; do not use `git add .` or include unrelated work.
The existing dirty HANDOFF/LEGACY_SOURCE_MAP entries may be reviewed/staged
separately with `git add -p`.

```powershell
git add -- scripts/order_luka_fullstat_embedding_sweep.py scripts/queue_luka_fullstat_embedding_order.sh src/hlt_classification/operations/__init__.py src/hlt_classification/operations/fullstat_embedding_order.py tests/test_luka_fullstat_embedding_order.py docs/plans/LUKA_FULLSTAT_EMBEDDING_ORDER_PLAN.md docs/contracts/LUKA_FULLSTAT_EMBEDDING_ORDER.md docs/LUKA_FULLSTAT_EMBEDDING_ORDER_RUNBOOK.md
git diff --cached --stat
git commit -m "Add ordered handover for existing Oscar FullStat embedding sweep"
if ($LASTEXITCODE -ne 0) { throw "Commit failed; inspect before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

## 2. Review and arm on login010

Paste the entire block. Prompts read `/dev/tty`, not heredoc stdin. Use the new
pushed 40-character commit, then the printed ORDER_PLAN_HASH. The dry review
does not signal the paused process or submit jobs. Execution retires only that
verified stopped Python controller and takes over waiting for the SAME preflight.

```bash
bash <<'BASH'
set -euo pipefail
test "$(hostname -s)" = login010
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
ROOT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_embedding_sweep_fd687181_r1
ORIGINAL=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_embedding_sweep_fd687181
read -r -p 'Paste the NEW pushed 40-character ordering commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" cat-file -e "${COMMIT}^{commit}"
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_embedding_order_${COMMIT:0:8}"
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
HELPER="${PROJECT}/scripts/queue_luka_fullstat_embedding_order.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${ORIGINAL}" 300411 login010
read -r -p 'Paste ORDER_PLAN_HASH printed above: ' HASH </dev/tty
[[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Retire the paused controller and automatically queue the ordered sweep after verified preflight success? Type YES: ' CONFIRM </dev/tty
test "${CONFIRM}" = YES
LOG="$(mktemp "${ROOT}/ordered-followup.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ROOT}" "${ORIGINAL}" 300411 login010 --execute "${HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Ordered controller PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Confirm ARMED ORDERED HANDOVER before disconnecting:\ntail -f "%s"\n' "${LOG}"
BASH
```

Wait for `ARMED ORDERED HANDOVER`. It means the old process is retired and the
new helper is waiting for successful authenticated gate acceptance. If preflight
has already passed, submission can proceed immediately after verification.
`ALL 8 JOBS QUEUED IN REQUESTED ORDER` records the exact job IDs. Ctrl-C stops
only tail. Do not rerun the launcher or resume the old controller.

Original preflight/science output locations and results CLI are unchanged.
New operational evidence lives at `ordered_followup_v1/{plan,retired,
measured_review,command_plan,submission_ledger,complete,failure}.json` (only
applicable records exist). These replace neither old followup records nor
scientific receipts. On an error, preserve everything and inspect the log and
exact journal; do not retry, delete claims, or submit jobs independently.

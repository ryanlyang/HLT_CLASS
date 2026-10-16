# Recover the completed batch128 probes without repeating GPU work

[Plan](plans/LUKA_BATCH128_REPORT_RECOVERY_PLAN.md) /
[contract](contracts/LUKA_BATCH128_REPORT_RECOVERY.md).

Job **21898881** completed all twelve child probes, then failed when its
process-tree sampler encountered an exiting process. This path preserves that
FAILED job and its original source. It authenticates the saved probes using
their original validators, checks accounting and the unchanged resource/time
limits, and queues **one CPU-only report job**. It does not rerun preprocessing
or GPU probes. Missing evidence or failed admission still blocks science.

After the CPU report succeeds, separately review/submit all **30 science jobs**:
17 fresh batch128 fits, 11 TRAIN reducers, aggregate and completion. Same split,
models, seeds, full particle support and 60--100-pass schedule; both fusion
endings and matched regular comparisons. No independent test set.

## 1. Publish the narrow zero-allocation fix from isolated source on Windows

The original recovery commit `ea7cfdfc0222768bdffd386058a807f6fac84de6`
incorrectly rejected GPU allocation zero in the two single-encoder no-op parity
phases. The correction allows just those two authenticated phases with empty
`parities`; real GPU phases still require positive measurements. No GPU rerun.

FullStat/shared-runner changes have since been committed on main. **Do not deploy
the combined main tip for this recovery.** Create a narrow repair commit based
on ea7cfdfc, then merge it into main using a separate clean worktree solely to
publish it. Deploy the narrow repair commit, NOT that integration merge commit.
This preserves both experiments and the strict source-compatibility check.

Run from the Windows repository containing the corrected files. Review the five
listed files first. These commands never switch/reset the current dirty worktree;
HANDOFF/LEGACY_SOURCE_MAP living-status edits remain for separate scoped review.
Existing helper directories cause a stop; preserve them instead of deleting them.

```powershell
$ErrorActionPreference = 'Stop'
$Repo = (Get-Location).Path
$Base = 'ea7cfdfc0222768bdffd386058a807f6fac84de6'
$RepairDir = Join-Path (Split-Path $Repo -Parent) 'HLT_Classification_luka_report_zero_fix'
$PublishDir = Join-Path (Split-Path $Repo -Parent) 'HLT_Classification_luka_report_publish'
$RepairFiles = @(
  'src/hlt_classification/luka_fullsim/alljets_report_replay.py',
  'tests/test_luka_batch128_report_recovery.py',
  'docs/plans/LUKA_BATCH128_REPORT_RECOVERY_PLAN.md',
  'docs/contracts/LUKA_BATCH128_REPORT_RECOVERY.md',
  'docs/LUKA_BATCH128_REPORT_RECOVERY_RUNBOOK.md'
)
if ((Test-Path -LiteralPath $RepairDir) -or (Test-Path -LiteralPath $PublishDir)) {
  throw 'Helper directory exists; preserve it and inspect before retrying.'
}
git -C $Repo fetch origin main
if ($LASTEXITCODE -ne 0) { throw 'Fetch failed' }
git -C $Repo merge-base --is-ancestor $Base origin/main
if ($LASTEXITCODE -ne 0) { throw 'Recovery base not on published main' }
git -C $Repo worktree add --detach $RepairDir $Base
if ($LASTEXITCODE -ne 0) { throw 'Repair worktree failed' }
foreach ($File in $RepairFiles) {
  Copy-Item -LiteralPath (Join-Path $Repo $File) -Destination (Join-Path $RepairDir $File)
}
git -C $RepairDir diff --stat
if ((Read-Host 'Commit only this reviewed recovery repair? Type YES') -ne 'YES') {
  throw 'Stopped; worktree preserved, no commit or push'
}
git -C $RepairDir add -- $RepairFiles
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git -C $RepairDir commit -m 'Accept zero allocation in no-op batch128 parity recovery'
if ($LASTEXITCODE -ne 0) { throw 'Inspect commit status; do not blindly retry' }
$RepairCommit = (git -C $RepairDir rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify repair commit' }
git -C $Repo worktree add --detach $PublishDir origin/main
if ($LASTEXITCODE -ne 0) { throw 'Publishing worktree failed' }
git -C $PublishDir merge --no-ff $RepairCommit -m 'Merge isolated batch128 report correction'
if ($LASTEXITCODE -ne 0) { throw 'Merge needs review; do not resolve by overwriting files' }
git -C $PublishDir push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; preserve both commits; never force-push' }
Write-Host "DEPLOY THIS REPAIR COMMIT: $RepairCommit"
```

Use the printed **repair commit** in every subsequent block. The normal merge
publishes it as an ancestor of main; the integrated tip includes FullStat and is
intentionally not a compatible recovery source. Neither worktree is deleted.

## 2. Dry-review the CPU recovery on sporcsubmit

This may take several minutes to hash saved payloads. It submits nothing.
Preserve both the old `922bd972` worktree and its failed campaign directory.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_luka_report_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_batch128_report_${COMMIT:0:8}_r1"
FAILED="${MAIN}/checkpoints/luka_matched_batch128_922bd972_r1/gate_spec.json"
test -f "${FAILED}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_batch128_report_recovery.sh" \
    "${COMMIT}" "${ROOT}" "${FAILED}" gate
printf '\nRecovery root: %s\n' "${ROOT}"
BASH
```

Expect `Admission: all original checks passed`, then a one-job plan with
1 CPU, 32768 MiB, four hours, debug, **no GPU request**. If admission reports
an error (including a >1440-minute projection), stop and inspect it. Do not
shorten the schedule or bypass the gate. Save the plan's `content_hash`.

## 3. Submit the reviewed plan

Use `gate` first. Only after step 4, use `science` with its **different** reviewed
hash. This single block supports either stage and has no wrapping subshell.

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New 40-character commit: ' COMMIT </dev/tty
read -r -p 'Reviewed stage (gate or science): ' MODE </dev/tty
read -r -p 'Reviewed PLAN content_hash: ' PLAN_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MODE}" =~ ^(gate|science)$ ]]
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_luka_report_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_batch128_report_${COMMIT:0:8}_r1"
FAILED="${MAIN}/checkpoints/luka_matched_batch128_922bd972_r1/gate_spec.json"
test -f "${ROOT}/gate_spec.json"
read -r -p "Submit the reviewed ${MODE} plan? Type YES: " CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || exit 0
LOG="${ROOT}/${MODE}-submit.$$.log"
set -o noclobber
nohup bash "${PROJECT}/scripts/queue_luka_batch128_report_recovery.sh" \
    "${COMMIT}" "${ROOT}" "${FAILED}" "${MODE}" --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Watch with:\ntail -f "%s"\n' "${LOG}"
BASH
```

Wait for the ledger/job ID and `QUEUE_HELPER_EXIT=0`; a helper PID alone is
not a submitted job. Ctrl-C stops `tail` only. The CPU job's log is
`ROOT/preflight-JOBID.out`. Check that exact job with `sacct -j JOBID -X -P
-o JobID,JobName,State,ExitCode,Elapsed`. Original 21898881 stays FAILED.

## 4. After the CPU report completes, dry-review science

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Same new 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_luka_report_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_batch128_report_${COMMIT:0:8}_r1"
FAILED="${MAIN}/checkpoints/luka_matched_batch128_922bd972_r1/gate_spec.json"
bash "${PROJECT}/scripts/queue_luka_batch128_report_recovery.sh" \
    "${COMMIT}" "${ROOT}" "${FAILED}" science
BASH
```

This requires the new CPU report's accepted exact journal, COMPLETED/0:0
accounting and authenticated admission. Review all 30 tasks and their resources;
then repeat step 3 with `science` and the science plan hash. No automatic
followup, cancellation or duplicate submission. Preserve any interrupted live
submission journal instead of deleting it or starting another helper.

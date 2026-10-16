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

## 1. Commit and push on Windows

Review changes first; do not stage unrelated files. HANDOFF has earlier user
edits, so review/stage its new entry separately if desired.
This recovery intentionally requires the model/data/recipe source to remain
identical to `922bd972`. In particular, do not bundle concurrent FullStat/shared
training-runner changes into this commit. If those are already committed on
your branch, stop and isolate the repair on the original source first; the
recovery compatibility check will reject a mixed source.

```powershell
git add -- src/hlt_classification/cms2jc2_response/measurement.py `
  src/hlt_classification/luka_fullsim/alljets_report_recovery.py `
  src/hlt_classification/luka_fullsim/alljets_report_replay.py `
  src/hlt_classification/luka_fullsim/alljets_fusion_campaign.py `
  src/hlt_classification/luka_fullsim/alljets_fusion_worker.py `
  src/hlt_classification/luka_fullsim/alljets_fusion_submission.py `
  scripts/luka_fullsim_alljets_fusion.py `
  scripts/queue_luka_batch128_report_recovery.sh `
  tests/test_process_measurement.py tests/test_luka_batch128_report_recovery.py `
  tests/test_luka_alljets_batch128.py tests/test_luka_fullsim_alljets_fusion.py `
  tests/test_luka_fusion_isolated_preflight.py tests/test_luka_fusion_cuda_diagnostic.py `
  docs/plans/LUKA_BATCH128_REPORT_RECOVERY_PLAN.md `
  docs/contracts/LUKA_BATCH128_REPORT_RECOVERY.md `
  docs/LUKA_BATCH128_REPORT_RECOVERY_RUNBOOK.md `
  docs/plans/LUKA_ALLJETS_BATCH128_PLAN.md docs/contracts/LUKA_ALLJETS_BATCH128.md `
  docs/LEGACY_SOURCE_MAP.md
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
git commit -m "Fix process sampler race and recover completed batch128 probes"
if ($LASTEXITCODE -ne 0) { throw 'Inspect commit status; do not blindly retry' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

If Git says nothing was staged, inspect `git log -1 --oneline` and status; the
changes may already be committed. Do not add unrelated files to force a commit.

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

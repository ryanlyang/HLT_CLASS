# SPORC matched batch128: controls, regular ladder and both fusion routes

[Plan](plans/LUKA_ALLJETS_BATCH128_PLAN.md) / [contract](contracts/LUKA_ALLJETS_BATCH128.md).
New experiment, not continuation of batch256 weights. Same299234/74895 rows and
60--100 passes; true128 for training/inference, accumulation1. All17 fits rerun,
11 TRAIN reducers, aggregate/complete =30 science jobs. No independent test.
No upload or preprocessing. Preserve every old source/artifact directory.

## Commit and push locally

Review the diff first. Stage only the following implementation files (do not use
`git add .`). `docs/HANDOFF.md` contains pre-existing edits; review separately.

```powershell
git add -- src/hlt_classification/jetclass2_delphes/runner.py `
  src/hlt_classification/jetclass2_delphes/batch128_recipe.py `
  src/hlt_classification/luka_fullsim/alljets_batch128.py `
  src/hlt_classification/luka_fullsim/alljets_fusion_campaign.py `
  src/hlt_classification/luka_fullsim/alljets_fusion_worker.py `
  src/hlt_classification/luka_fullsim/alljets_fusion_preflight.py `
  src/hlt_classification/luka_fullsim/alljets_fusion_isolated_preflight.py `
  src/hlt_classification/luka_fullsim/alljets_fusion_submission.py `
  scripts/luka_fullsim_alljets_fusion.py scripts/queue_luka_alljets_batch128.sh `
  tests/test_luka_alljets_batch128.py tests/test_luka_fullsim_alljets_fusion.py `
  tests/test_luka_fusion_isolated_preflight.py tests/test_luka_fusion_cuda_diagnostic.py `
  docs/plans/LUKA_ALLJETS_BATCH128_PLAN.md docs/contracts/LUKA_ALLJETS_BATCH128.md `
  docs/LUKA_ALLJETS_BATCH128_RUNBOOK.md docs/LEGACY_SOURCE_MAP.md `
  docs/plans/LUKA_FULLSIM_ALLJETS_FUSION_PLAN.md
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
git commit -m "Register matched Luka batch128 controls and both ladders"
if ($LASTEXITCODE -ne 0) { throw 'Inspect commit status before proceeding' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## Fresh worktree and dry review on sporcsubmit

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Paste the new 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_luka_batch128_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_matched_batch128_${COMMIT:0:8}_r1"
FAILED="${MAIN}/checkpoints/luka_fusion_isolated_5cab1830_r1/gate_spec.json"
test -f "${FAILED}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_alljets_batch128.sh" "${COMMIT}" "${ROOT}" "${FAILED}" gate
printf '\nStudy: %s\nProject: %s\n' "${ROOT}" "${PROJECT}"
BASH
```

Authenticates the existing failed job using its ORIGINAL source, then registers
ONE preflight (debug A10040GB,6CPU,250GiB,8h). It submits nothing. Review the
printed plan. Do not rerun preparation or edit the failed gate. Authentication
can take minutes; it is not yet a queued job.

## Submit the exact reviewed gate

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New commit: ' COMMIT </dev/tty
read -r -p 'Reviewed gate plan content_hash: ' PLAN_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_luka_batch128_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_matched_batch128_${COMMIT:0:8}_r1"
FAILED="${MAIN}/checkpoints/luka_fusion_isolated_5cab1830_r1/gate_spec.json"
LOG="$(mktemp "${MAIN}/checkpoints/luka-batch128-submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_alljets_batch128.sh" \
    "${COMMIT}" "${ROOT}" "${FAILED}" gate --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
BASH
```

Require `QUEUE_HELPER_EXIT=0` and a submitted job ID; a helper PID alone is not
submission. First D000/D000 parity and full128 stress run in separate fresh CUDA
processes. Then U050/U000, singleD000, singleU000; all stresses precede full
technical passes. Small local tests cannot prove the full-support A100 case.
Failure stops without retry or science. A measured >24h full-schedule request
also blocks admission; do not shorten the schedule or relax safety factors.

## After successful preflight: separately review and queue all30 science jobs

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_luka_batch128_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_matched_batch128_${COMMIT:0:8}_r1"
FAILED="${MAIN}/checkpoints/luka_fusion_isolated_5cab1830_r1/gate_spec.json"
bash "${PROJECT}/scripts/queue_luka_alljets_batch128.sh" "${COMMIT}" "${ROOT}" "${FAILED}" science
read -r -p 'Reviewed SCIENCE plan content_hash: ' PLAN_HASH </dev/tty
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Submit all30 matched batch128 science jobs? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]] || exit 0
LOG="$(mktemp "${MAIN}/checkpoints/luka-batch128-science.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_alljets_batch128.sh" \
    "${COMMIT}" "${ROOT}" "${FAILED}" science --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
BASH
```

Science creation authenticates complete v4 evidence and exact successful gate
accounting. No imported batch256 U000 targets or comparisons. Every fit starts
fresh; local reducers govern all KD dependencies. Nothing automatically cancels,
retries, deletes, requeues or submits another stage.

Results: in this worktree/environment run
`python -u -s scripts/luka_fullsim_alljets_fusion.py results --spec ROOT/science/campaign_spec.json`
with the actual absolute root. Full74895 validation selects, fixed50k reports
the same checkpoint. Compare single-HLT endpoints; label offline fusion oracle
and D000/D000 two-encoder HLT. All results remain exploratory/single-seed.

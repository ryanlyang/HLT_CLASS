# SPORC matched fixed30: trial ALLJETS, both ladders

This is the old trial299,234/74,895 population, **not new FullStat**. Seventeen
fresh fits: controls, direct, regular coarse, fusion and both HLT endings.
Batch128, exactly30 epochs, warmup3/cosine-to30. Full validation selects weights;
original50k is secondary. No independent test remains. Same per-fit budget,
unequal total route compute; no promise thirty epochs converges.

Keep original922bd972 worktree, failed job21898881 outputs, all ancestor
worktrees/artifacts and prepared cache. No dataset rebuild or GPU probe rerun.
The old100-pass report stays failed. New v6 CPU admission reauthenticates ALL
technical facts and recomputes the thirty-pass request with1.75 margin. It checks
time before queueing. The exact estimate, not3872*0.3 alone, governs admission.
No experiment has been submitted by local implementation.

## 1. Publish a scoped commit from the current main worktree

Unlike v5's isolated repair, v6 explicitly supports the concurrent FullStat
namespace from592a6dac. Its files must remain identical to that donor. Do not
cherry-pick this onto ea7 alone or include unrelated changes. In PowerShell:

```powershell
git add -- src/hlt_classification/jetclass2_delphes/batch128_30e_recipe.py src/hlt_classification/jetclass2_delphes/runner.py src/hlt_classification/luka_fullsim/alljets_30epoch.py src/hlt_classification/luka_fullsim/alljets_fusion_campaign.py src/hlt_classification/luka_fullsim/alljets_fusion_preflight.py src/hlt_classification/luka_fullsim/alljets_fusion_submission.py src/hlt_classification/luka_fullsim/alljets_fusion_worker.py src/hlt_classification/luka_fullsim/alljets_report_replay.py scripts/luka_fullsim_alljets_fusion.py scripts/queue_luka_alljets_30epoch.sh tests/test_luka_alljets_30epoch.py tests/test_luka_alljets_batch128.py tests/test_luka_batch128_report_recovery.py tests/test_luka_fullsim_alljets_fusion.py docs/plans/LUKA_ALLJETS_30E_PLAN.md docs/contracts/LUKA_ALLJETS_30E.md docs/LUKA_ALLJETS_30E_RUNBOOK.md
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git add -- tests/test_luka_fusion_isolated_preflight.py
if ($LASTEXITCODE -ne 0) { throw 'Stage failed' }
git diff --cached --stat
# Review this list before continuing. Stage only the new HANDOFF and
# LEGACY_SOURCE_MAP sections with git add -p; preserve unrelated edits.
git commit -m "Register matched batch128 fixed30 trial ladders"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect status before proceeding' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Dry review on sporcsubmit

Paste the new full commit when prompted. Fresh root; no overwrite/resume.

```bash
read -r -p 'New40-character commit: ' COMMIT
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 1
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka30_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_fixed30_${COMMIT:0:8}_r1"
OLD_GATE="${MAIN}/checkpoints/luka_matched_batch128_922bd972_r1/gate_spec.json"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main || exit 1
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}" || exit 1
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}" || exit 1
fi
bash "${PROJECT}/scripts/queue_luka_alljets_30epoch.sh" \
    "${COMMIT}" "${ROOT}" "${OLD_GATE}" gate
```

This authenticates source/bytes and may take minutes. It prints new train/reducer
requests and ONE CPU job. Stop and inspect any error; do not launch an old helper.

## 3. Submit the reviewed CPU admission job

Same shell/variables. Type the PLAN content_hash, not the GATE hash.

```bash
read -r -p 'Reviewed gate PLAN hash: ' PLAN_HASH
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]] || exit 1
LOG="$(mktemp "${ROOT}/gate-submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_alljets_30epoch.sh" \
    "${COMMIT}" "${ROOT}" "${OLD_GATE}" gate --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Log: %s\n' "${LOG}"
tail -f "${LOG}"
```

Ctrl-C stops tail only. Wait for this `luka30_preflight` to COMPLETE0:0 and its
authenticated `preflight/preflight.json`; science does not submit automatically.

## 4. Review and submit all thirty science jobs

```bash
bash "${PROJECT}/scripts/queue_luka_alljets_30epoch.sh" \
    "${COMMIT}" "${ROOT}" "${OLD_GATE}" science
```

Check all17 train nodes,11 reducers, aggregate/complete, requests<=24h,
batch128, recipe30. Then use the NEW science PLAN hash:

```bash
read -r -p 'Reviewed science PLAN hash: ' PLAN_HASH
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]] || exit 1
LOG="$(mktemp "${ROOT}/science-submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_alljets_30epoch.sh" \
    "${COMMIT}" "${ROOT}" "${OLD_GATE}" science --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Log: %s\n' "${LOG}"
tail -f "${LOG}"
```

If reconnecting, set COMMIT/MAIN/PROJECT/ROOT/OLD_GATE again to these same values;
do not create another experiment. Submission journals prevent duplicates; never
delete a live_claim to retry an interrupted submission. No blanket cancellations.

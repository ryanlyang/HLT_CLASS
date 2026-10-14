# Luka 299,234 / 74,895 regular ladder on SPORC debug

This new exploratory run retires the already-consumed historical test into TRAIN.
No independent test remains. It preserves original validation files, the physical
recipe and the nine-model regular ladder. See the [plan](plans/LUKA_FULLSIM_ALLJETS_PLAN.md)
and [contract](contracts/LUKA_FULLSIM_ALLJETS.md). Existing campaigns are untouched.

## 1. Commit and push the implementation from Windows

Review changes first. Do not use `git add .`: this checkout contains unrelated work.

```powershell
git add -- src/hlt_classification/luka_fullsim/alljets_data.py src/hlt_classification/luka_fullsim/alljets_campaign.py src/hlt_classification/luka_fullsim/alljets_reporting.py src/hlt_classification/luka_fullsim/regular_campaign.py src/hlt_classification/luka_fullsim/regular_cache.py src/hlt_classification/luka_fullsim/regular_preflight.py src/hlt_classification/luka_fullsim/regular_worker.py src/hlt_classification/luka_fullsim/regular_submission.py scripts/luka_fullsim_alljets.py scripts/queue_luka_fullsim_alljets.sh sbatch/run_luka_fullsim_alljets.sh tests/test_luka_fullsim_alljets.py tests/test_luka_fullsim_regular.py docs/plans/LUKA_FULLSIM_ALLJETS_PLAN.md docs/contracts/LUKA_FULLSIM_ALLJETS.md docs/LUKA_FULLSIM_ALLJETS_RUNBOOK.md docs/LEGACY_SOURCE_MAP.md
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
# HANDOFF already had unrelated edits: stage only the new all-jets section.
git add -p -- docs/HANDOFF.md
git diff --cached --stat
git commit -m "Add all-eligible Luka regular ladder development campaign"
if ($LASTEXITCODE -ne 0) { throw 'Inspect commit status before continuing' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

## 2. Fresh pinned worktree and gate dry review on sporcsubmit

Paste the full printed commit at the prompt. No dataset re-upload is needed.
The original restored 2.2 GiB raw snapshot is reused read-only. Review both free
space and your quota for the new durable seven-view cache; old caches are preserved.
The local full scalar-count check measured a conservative **22.624 GiB free-space
requirement** for preparation, including the reserve. The remote worker repeats
this check; `df` alone does not establish available personal quota.

```bash
read -r -p 'Pushed 40-character commit: ' COMMIT
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]] || { echo 'Invalid commit'; return 1 2>/dev/null || exit 1; }
MAIN=/home/ryreu/atlas/HLT_Classification
SHORT="${COMMIT:0:8}"
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_alljets_${SHORT}"
ROOT="${MAIN}/checkpoints/luka_alljets_${SHORT}_r1"
export COMMIT PROJECT ROOT
(
set -euo pipefail
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
df -h "${MAIN}"
quota -s 2>/dev/null || true
bash "${PROJECT}/scripts/queue_luka_fullsim_alljets.sh" "${COMMIT}" "${ROOT}" gate
)
```

Review three jobs only: `lukaa_foundation` (1 CPU, 16 GiB, 30min),
`lukaa_prepare` (8 CPUs, 64,000 MiB, 8h), `lukaa_preflight`
(6 CPUs, 90,000 MiB, one A100, 8h), all debug/`reu-aisocial`/`qos_tier3`.
Two `afterok` dependencies link them. CPU preparation must not request a GPU.

## 3. Submit the exact reviewed gate

Use the PLAN **content_hash**, not its parent hash. These variables survive only
in this shell; after reconnecting, restore COMMIT/PROJECT/ROOT from step 2 first.

```bash
read -r -p 'Reviewed gate plan content_hash: ' PLAN_HASH
(
set -euo pipefail
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
LOG=$(mktemp "${ROOT}/gate-submit.XXXXXX.log")
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_alljets.sh" \
    "${COMMIT}" "${ROOT}" gate --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
)
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
```

Inspect the log for `QUEUE_HELPER_EXIT=0` and the three IDs. `Ctrl-C` stops a
`tail -f`, not the background helper. Do not launch duplicate helpers on an
ambiguous reply; preserve the submission journal.

## 4. After all three gate jobs succeed: science dry review, then submit

The new resource measurements must pass first. This does **not** reuse the old
100k gate as evidence for the larger population.

```bash
bash "${PROJECT}/scripts/queue_luka_fullsim_alljets.sh" "${COMMIT}" "${ROOT}" science
```

Review the 16 jobs and measured walltimes (maximum 24h). Then:

```bash
read -r -p 'Reviewed SCIENCE plan content_hash: ' PLAN_HASH
(
set -euo pipefail
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
LOG=$(mktemp "${ROOT}/science-submit.XXXXXX.log")
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_alljets.sh" \
    "${COMMIT}" "${ROOT}" science --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
)
```

Graph: controls M0HLT/OFFLINE/U000; U000 -> DIRECT_D000 and
U000 -> COARSE_U050 -> COARSE_U100 -> COARSE_D066 -> COARSE_D033 -> COARSE_D000.
Each arrow uses the preceding selected model's TRAIN-only bank. No fusion.

## 5. Partial or complete results

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -u -s "${PROJECT}/scripts/luka_fullsim_alljets.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

Reports show full-validation accuracy/AUC/rejection50/recoveries and the original
50k subset's accuracy/AUC/recoveries at the **same** selected checkpoint. Preserve
`science/train_*/validation_predictions.npz`, diagnostics, reports, selected
checkpoints, receipts and campaign/foundation metadata for future comparisons.
Do not call these test results. Real GPU acceptance remains a remote gate, not
something established by local CPU tests.

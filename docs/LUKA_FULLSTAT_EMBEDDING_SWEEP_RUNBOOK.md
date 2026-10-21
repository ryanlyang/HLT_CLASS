# FullStat standalone embedding sweep on Oscar

[Plan](plans/LUKA_FULLSTAT_EMBEDDING_SWEEP_PLAN.md) /
[contracts](contracts/LUKA_FULLSTAT_EMBEDDING_SWEEP.md)

Fresh native HLT models: lambda 0 (CE-only), 2, 10, 50, 100, 500, 1000. Same 3M/500k,
batch 128, 20 epochs, M0HLT initialization/sampling, TRAIN-only standardized
pure-OFFLINE 128-vector. No logit KD, D033 targets, attention or warm start.
Training-only auxiliary predictor is removed from deployment. Final test sealed.

One L40S preflight (6 CPUs, 180,000 MiB, 8h) checks all seven technical full epochs
and measured resources. With explicit `auto` approval, it is followed by seven
GPU fits plus CPU aggregation after successful accounting AND artifact checks.
Actual fit requests are measured (12h floor/48h cap), not guessed from old runs.
One failed gate stops continuation; poor AUC is not a failure condition.

## Commit and push on Windows

Use the repository root. Preserve unrelated dirty work and any existing staged
changes. The additive source donor is `007885f7360f4cd28e9f91a690e04f0f4ea0afff`;
unrelated later runtime edits require a reviewed compatibility update, not bypass.

```powershell
& {
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { throw "Inspect existing staged changes first" }
    $files = @(
        "src/hlt_classification/luka_fullstat_embedding_sweep"
        "scripts/luka_fullstat_embedding_sweep.py"
        "scripts/queue_luka_fullstat_embedding_sweep.sh"
        "sbatch/run_luka_fullstat_embedding_sweep.sh"
        "tests/test_luka_fullstat_embedding_sweep.py"
        "tests/test_luka_fullstat_embedding_sweep_followup.py"
        "docs/plans/LUKA_FULLSTAT_EMBEDDING_SWEEP_PLAN.md"
        "docs/contracts/LUKA_FULLSTAT_EMBEDDING_SWEEP.md"
        "docs/LUKA_FULLSTAT_EMBEDDING_SWEEP_RUNBOOK.md"
    )
    git add -- $files
    if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
    git diff --cached --stat
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw "Inspect whitespace errors" }
    git commit -m "Add FullStat20 CE embedding sweep with gated automatic science"
    if ($LASTEXITCODE -ne 0) { throw "Inspect commit status" }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw "Push failed" }
    git rev-parse HEAD
}
```

The new HANDOFF/LEGACY_SOURCE_MAP sections can be staged separately with
`git add -p`; do not include unrelated sections blindly.

## One Oscar launch: review, authorize, detach

Reserve at least 12 GiB of **personal available quota**, not shared filesystem
free space. Source/cache/parent artifacts are reused read-only. This creates a
new root; it does not change the currently running gate/embedding experiments.

Paste the entire block, then answer its prompts. The helper checks the repaired
Oscar environment; no automatic package updates or source changes occur.

```bash
(
set -euo pipefail
read -r -p 'Pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_embedding_sweep_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_embedding_sweep_${COMMIT:0:8}_r1"
ORIGINAL=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_recovery_9bc3cae0_r1/science/campaign_spec.json
test -f "${ORIGINAL}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" cat-file -e "${COMMIT}^{commit}"
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
checkquota
read -r -p 'At least 12 GiB personally available? Type YES: ' SPACE_OK </dev/tty
[[ "${SPACE_OK}" == YES ]]
bash "${PROJECT}/scripts/queue_luka_fullstat_embedding_sweep.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" 12 auto
read -r -p 'Paste FOLLOWUP_PLAN_HASH printed above: ' HASH </dev/tty
[[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Authorize preflight plus ALL eight science jobs after verified success? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="${ROOT}/auto-followup.log"
test ! -e "${LOG}"
nohup bash "${PROJECT}/scripts/queue_luka_fullstat_embedding_sweep.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" 12 auto --execute "${HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Controller PID: %s\nStudy: %s\n' "$!" "${ROOT}"
printf 'Watch with:\ntail -f "%s"\n' "${LOG}"
printf 'Wait for ARMED. Do not also submit science manually.\n'
)
```

Run the printed tail command and confirm **ARMED** with an exact preflight ID,
not merely a PID. Ctrl-C stops tail only. Ordinary SSH disconnects are safe;
submit-host reboot/controller death is not covered. Poll every 60s, up to 14 days; no GPU
held by the controller. After success, **ALL SCIENCE JOBS QUEUED** lists eight
IDs; this means submission, not finished training. Failure records and exact
journals remain for inspection. Never blindly restart an uncertain submission.

Default manual `gate`/`science` modes still require separate dry/live plan hashes.
Do not combine a manual science submission with an armed controller.
The original preflight from another campaign cannot admit this sweep.

```bash
squeue --me -o "%.18i %.12P %.40j %.2t %.10M %.12l %R"
```

Prefix `lukafses_`; independent fits may still serialize under personal QOS
limits. Aggregate waits afterok for all seven; no quality-based selection/cancel.

## Results and live progress

Set PROJECT and ROOT to the exact paths printed at launch, then:

```bash
export PROJECT_DIR="${PROJECT}"
export JC2_SITE=oscar_l40s
unset LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT}/scripts/luka_fullstat_embedding_sweep.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

Completed arms have full-validation AUC/accuracy, selected epoch, delta and
recovery versus the NEW w0. Until w0 finishes, those comparisons are n/a;
pending arms are not skipped or replaced by provisional logs. Old M0HLT and
OFFLINE are labeled historical. All comparisons remain single-seed exploratory.

Logs under `science/wWEIGHT-JOBID.out` print `FULLSTAT CE EMBED weight=...`
once per epoch and `FULLSTAT CE EMBED progress` every 128 batches with training
ETA (excluding validation). No offline embeddings are read during validation.
Retain targets, selected.pt, training_report.json, validation_predictions.npz,
receipts and source. No per-epoch optimizer archive or automatic resumption.

Local CUDA miniatures do not establish an Oscar runtime envelope; the genuine
full-population gate is mandatory. No cluster job is submitted by implementation.

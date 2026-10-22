# Oscar FullStat U100 embedding weight5: no new full preflight

Authority: [plan](plans/LUKA_FULLSTAT_U100_EMBEDDING_PLAN.md) and
[contract](contracts/LUKA_FULLSTAT_U100_EMBEDDING.md).
Two queued jobs: export_U100 -> train_w5. Export includes new-target integrity,
maximum-support and three-update checks, NOT a technical training epoch.
Accepted fd687181 real Oscar resource evidence is reauthenticated and reused.
Missing/corrupt/incompatible evidence blocks registration; do not bypass it.
No old jobs are cancelled, resumed, held or reprioritized. The running w50 and
previously cancelled sweep remain unchanged. No detached followup controller.

Scientific recipe: original 3M/500k, HLT-only, fresh initialization with matched
seeds, batch128, fixed20,
CE+5*SmoothL1 predicting frozen COARSE_U100 TRAIN embeddings. TRAIN-only new
normalization; predictor removed at inference. Historical M0HLT-relative recovery
is descriptive, not a matched control or independent test. Test remains sealed.

## Commit and push on Windows

Do not stage unrelated HANDOFF/LEGACY_SOURCE_MAP changes wholesale; review the
new entries separately with git add -p if desired.

```powershell
& {
    git diff --cached --quiet
    if ($LASTEXITCODE -ne 0) { throw "Other changes are staged; inspect first" }
    git add -- src/hlt_classification/luka_fullstat_u100_embedding scripts/luka_fullstat_u100_embedding.py scripts/queue_luka_fullstat_u100_embedding.sh sbatch/run_luka_fullstat_u100_embedding.sh tests/test_luka_fullstat_u100_embedding.py docs/plans/LUKA_FULLSTAT_U100_EMBEDDING_PLAN.md docs/contracts/LUKA_FULLSTAT_U100_EMBEDDING.md docs/LUKA_FULLSTAT_U100_EMBEDDING_RUNBOOK.md
    if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
    git commit -m "Add FullStat U100 embedding weight5 with accepted-resource reuse"
    if ($LASTEXITCODE -ne 0) { throw "Commit failed" }
    git push origin HEAD:main
    if ($LASTEXITCODE -ne 0) { throw "Push failed" }
    git rev-parse HEAD
}
```

## Review and queue on Oscar

Keep original source and accepted sweep/preflight artifacts. Requires6GiB
personal durable headroom; checkquota reports shared-space limits separately.
Review/authentication may take minutes but schedules no repeated full preflight.
Use a new run suffix for a genuinely new authorized experiment; never retry a
partial submission by removing claims or inventing another root.

```bash
bash <<'BASH'
set -euo pipefail
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PREVIOUS=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_embedding_sweep_fd687181_r1/gate_spec.json
read -r -p 'Paste the pushed U100 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" cat-file -e "${COMMIT}^{commit}"
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_u100_embedding_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_u100_embedding_w5_${COMMIT:0:8}_r1"
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
checkquota
read -r -p 'Personally available durable result GiB (at least 6): ' QUOTA </dev/tty
[[ "${QUOTA}" =~ ^[1-9][0-9]*$ ]]
HELPER="${PROJECT}/scripts/queue_luka_fullstat_u100_embedding.sh"
bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PREVIOUS}" "${QUOTA}"
read -r -p 'Paste PLAN_HASH printed above: ' HASH </dev/tty
[[ "${HASH}" =~ ^[0-9a-f]{64}$ ]]
read -r -p 'Queue U100 export and automatic dependent weight5 training? Type YES: ' CONFIRM </dev/tty
test "${CONFIRM}" = YES
LOG="$(mktemp "${ROOT}/queue-submit.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ROOT}" "${PREVIOUS}" "${QUOTA}" \
    --execute "${HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nStudy: %s\nWatch with:\ntail -f "%s"\n' "$!" "${ROOT}" "${LOG}"
BASH
```

Wait until the submission log contains both exact job IDs. The training job
waits afterok on export and starts automatically after successful checks.
An export failure prevents the fit from running; no preflight waiting/manual
release step. Slurm cancellation of a failed dependency may not be immediate.
The submit process exits after the two-job journal is durable.

Logs: ROOT/export_U100-JOB.out and ROOT/train_w5-JOB.out. Training prints epoch
AUC and per128-batch progress/ETA. Finished artifacts are train_w5/{selected.pt,
training_report.json,validation_predictions.npz,results.json,task_receipt.json}.
Targets are ROOT/targets, export receipt ROOT/export_U100/receipt.json. CLI results
authenticates original reuse, export and selected predictions. No final-test CLI.

```bash
# Set ROOT and PROJECT to the exact paths printed at submission.
export PROJECT_DIR="${PROJECT}"
export JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT}/scripts/luka_fullstat_u100_embedding.py" results \
    --spec "${ROOT}/campaign_spec.json"
```

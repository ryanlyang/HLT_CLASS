# Queue the four-stage cumulative embedding ladder on Oscar

Four fresh fits: UE100 -> DE066 -> DE033 -> DE000. One submission; no separate
preflight or setup job and no login-node controller. All jobs are queued now,
with afterok dependencies. Each includes three short startup updates and its
scientific20 epochs; the first three also export their selected embeddings.

See the [scientific plan](plans/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_PLAN.md) and
[contract](contracts/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING.md). Earlier campaigns,
worktrees, caches and banks must remain intact. No old jobs are changed.

## 1. Commit and push on Windows

Do not use `git add .`. Review staged changes first and preserve unrelated work.
HANDOFF/LEGACY_SOURCE_MAP already had unrelated changes: selectively stage the
new cumulative sections if desired.

```powershell
git add -- src/hlt_classification/luka_fullstat_cumulative_embedding scripts/luka_fullstat_cumulative_embedding.py scripts/queue_luka_fullstat_cumulative_embedding.sh sbatch/run_luka_fullstat_cumulative_embedding.sh tests/test_luka_fullstat_cumulative_embedding.py docs/plans/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_PLAN.md docs/contracts/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING.md docs/LUKA_FULLSTAT_CUMULATIVE_EMBEDDING_RUNBOOK.md
git diff --cached --stat
git commit -m "Add FullStat cumulative embedding ladder without separate preflight"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

## 2. Review on Oscar

Paste the pushed40-character commit. This creates a clean detached worktree and
new immutable study, verifies the completed four-teacher run, and prints the
four commands. It does NOT submit jobs. Authentication may take several minutes;
it reads existing evidence, not a new GPU preflight.

Check that `checkquota` shows at least12GiB personal durable headroom before
declaring12. Filesystem-wide free space is not your quota. Preserve all existing
outputs; if this study already exists, the helper verifies the exact registration.

```bash
(
set -euo pipefail
read -r -p 'Pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="${MAIN}_fullstat_cumulative_embedding_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_cumulative_embedding_${COMMIT:0:8}_r1"
PREVIOUS=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiteacher_b21c2d4e_r1/campaign_spec.json
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
bash "${PROJECT}/scripts/queue_luka_fullstat_cumulative_embedding.sh" "${COMMIT}" "${ROOT}" "${PREVIOUS}" 12
printf '\nStudy: %s\nProject: %s\n' "${ROOT}" "${PROJECT}"
)
```

The plan must say `jobs: 4`, `preflight_jobs: 0`, `setup_jobs: 0`. Check the
registered input/teacher list and inherited walltime. No new timing estimate
has been measured for these fits; requests reuse the original native and
four-head evidence plus120 minutes. There is no automatic batch/epoch reduction.

## 3. Submit all four jobs together

Use the same commit and printed `PLAN_HASH`. Do not repeat after a disconnect
or uncertain submission; first inspect `submission/` journals and your queue.

```bash
(
set -euo pipefail
read -r -p 'Same pushed commit: ' COMMIT </dev/tty
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_cumulative_embedding_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_cumulative_embedding_${COMMIT:0:8}_r1"
PREVIOUS=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiteacher_b21c2d4e_r1/campaign_spec.json
read -r -p 'Authorize all four 20-epoch fits and conditional exports? Type YES: ' CONFIRM </dev/tty
test "${CONFIRM}" = YES
bash "${PROJECT}/scripts/queue_luka_fullstat_cumulative_embedding.sh" "${COMMIT}" "${ROOT}" "${PREVIOUS}" 12 --execute "${PLAN_HASH}"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
)
```

Expect `lukafsc_train_UE100`, `lukafsc_train_DE066`, `lukafsc_train_DE033`,
`lukafsc_train_DE000`. Dependencies are normal. Once all four exact IDs are
reported, Slurm owns the chain; disconnecting does not stop it. A technical
failure blocks/cancels dependent jobs, while a valid low-AUC result continues.
Do not manually submit later rungs or edit an immutable study to repair a failure.

## Progress and results

`train_STAGE-JOBID.out` prints epoch AUC, best-so-far AUC, CE and each auxiliary
loss. Every128 training batches it prints completed batches and estimated
training time remaining (excluding validation/checkpointing). Each first-three
job exports its selected TRAIN embedding after training, then publishes its
task receipt. `sacct` completion means this export finished too.

The following command authenticates completed selected validation results,
prints AUC/recovery and R50/R30 comparisons, and includes per-class recoveries
and pending stages in JSON. It performs no inference or test access.

```bash
(
set -euo pipefail
read -r -p 'Execution commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
export PROJECT_DIR="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_cumulative_embedding_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_cumulative_embedding_${COMMIT:0:8}_r1"
export JC2_SITE=oscar_l40s
unset PYTHONHOME PYTHONPATH LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_cumulative_embedding.py" results --spec "${ROOT}/campaign_spec.json"
)
```

Compare DE000 with completed U100-w5, four-teacher and historical M0HLT; OFFLINE
is the oracle. UE100/DE066/DE033 use richer inputs, not deployable HLT controls.
The 3M/500k split, 20 epochs and final-test sealing are unchanged. This is
single-seed adaptive development validation. Local installed-Weaver GPU tests
exercise all four production workers, but are not a new Oscar resource test;
the original completed Oscar evidence is checked at review and execution.

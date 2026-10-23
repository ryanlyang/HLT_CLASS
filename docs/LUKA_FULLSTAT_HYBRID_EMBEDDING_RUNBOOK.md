# Queue the five-rung hybrid ladder on Oscar

UE000 -> UE100 -> DE066 -> DE033 -> DE000, all fresh20-epoch fits.
Each uses75% logit KD /25% CE plus total embedding weight5 over the exact
[teacher table](plans/LUKA_FULLSTAT_HYBRID_EMBEDDING_PLAN.md).
See the [version1 contract](contracts/LUKA_FULLSTAT_HYBRID_EMBEDDING.md).

One submission queues all five jobs. No standalone preflight/setup/reducer
job and no login controller. Each job has three disposable startup updates;
scientific initialization is reset afterward. Applicable TRAIN target exports
are part of the scientific jobs. Old cumulative jobs are NOT changed or cancelled.

## 1. Commit and push locally

Only stage these new files, not `git add .`. Preserve unrelated work, including
the separately developed particle-prediction setup. Selectively stage the new
HANDOFF/LEGACY_SOURCE_MAP sections if desired; they already contain other edits.

```powershell
git add -- src/hlt_classification/luka_fullstat_hybrid_embedding scripts/luka_fullstat_hybrid_embedding.py scripts/queue_luka_fullstat_hybrid_embedding.sh sbatch/run_luka_fullstat_hybrid_embedding.sh tests/test_luka_fullstat_hybrid_embedding.py docs/plans/LUKA_FULLSTAT_HYBRID_EMBEDDING_PLAN.md docs/contracts/LUKA_FULLSTAT_HYBRID_EMBEDDING.md docs/LUKA_FULLSTAT_HYBRID_EMBEDDING_RUNBOOK.md
git diff --cached --stat
git commit -m "Add FullStat hybrid KD and embedding ladder without separate preflight"
if ($LASTEXITCODE -ne 0) { throw "Inspect commit status before proceeding" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

Use the printed40-character commit. The execution source check permits this
package's runtime additions beyond6b76a677, plus only NEW files from the
explicitly listed concurrent `luka_particle_prediction` trial package, which
this campaign never imports. Existing runtime changes remain forbidden.
Other changes require source compatibility review; do not weaken the guard.

## 2. Review on Oscar

The completed four-teacher study, original Oscar acceptance and original
FullStat references must still be intact. Reuse does not require completion
of the currently queued CE-only cumulative ladder. Authentication reads saved
evidence; it does not run a new GPU preflight.

Run `checkquota` and confirm at least16GiB personally available durable quota
before declaring16. Filesystem-wide free space is not your quota. No deletion
or overwrite is part of these commands.

```bash
(
set -euo pipefail
read -r -p 'Pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT="${MAIN}_fullstat_hybrid_embedding_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_hybrid_embedding_${COMMIT:0:8}_r1"
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
bash "${PROJECT}/scripts/queue_luka_fullstat_hybrid_embedding.sh" "${COMMIT}" "${ROOT}" "${PREVIOUS}" 16
printf '\nStudy: %s\nProject: %s\n' "${ROOT}" "${PROJECT}"
)
```

This submits nothing. Review `jobs:5`, `preflight_jobs:0`, `setup_jobs:0`,
the node/recipe teacher lists, input coordinates and per-task walltimes.
UE100 has only OFFLINE/M0HLT embedding heads; DE000 has no M0HLT head.
Every stage has a logit teacher. Copy the printed `PLAN_HASH`.

Requests reuse max(original native/KD training request, completed four-teacher
request)+120min, with180min instead for the first job's original-target exports.
L40S,6CPUs,180000MiB; reject above48h, never clamp. New combined runtime is
unmeasured. No automatic smaller batches, fewer epochs or lower particle support.

## 3. Submit the five jobs

Use the same pushed commit and reviewed hash. Do not repeat after an uncertain
submission/disconnect; inspect the durable `submission/` journal first.

```bash
(
set -euo pipefail
read -r -p 'Same pushed commit: ' COMMIT </dev/tty
read -r -p 'Reviewed PLAN_HASH: ' PLAN_HASH </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
PROJECT="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_hybrid_embedding_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_hybrid_embedding_${COMMIT:0:8}_r1"
PREVIOUS=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiteacher_b21c2d4e_r1/campaign_spec.json
read -r -p 'Authorize all five hybrid fits and their TRAIN exports? Type YES: ' CONFIRM </dev/tty
test "${CONFIRM}" = YES
bash "${PROJECT}/scripts/queue_luka_fullstat_hybrid_embedding.sh" "${COMMIT}" "${ROOT}" "${PREVIOUS}" 16 --execute "${PLAN_HASH}"
squeue --me -o '%.18i %.12P %.40j %.2t %.10M %R'
)
```

Expect `lukafsh_train_UE000`, `lukafsh_train_UE100`, `lukafsh_train_DE066`,
`lukafsh_train_DE033`, `lukafsh_train_DE000`. All five are queued immediately
with afterok dependencies. Once all exact IDs are recorded, you can disconnect:
Slurm owns the chain, not a login session. A technical failure blocks/cancels
dependent jobs; a finite low-AUC result does not. Do not submit later rungs
manually, resume old controllers, or cancel unrelated jobs by a name wildcard.

## Progress and results

Logs are `train_STAGE-JOBID.out`. The first job exports M0HLT embeddings and
OFFLINE probabilities before it begins training, so its first epoch takes
longer to appear. Each log prints phase heartbeats, per-epoch AUC/best AUC,
CE, T-squared KD and per-head losses. Every128 training batches it prints
batch progress and estimated remaining TRAIN time, excluding validation.

Selected outgoing targets are exported AFTER training and BEFORE task completion.
The first four jobs export temperature-two probabilities; UE100/DE066/DE033
also export selected native embeddings. An epoch20 log is not a completed task.

```bash
(
set -euo pipefail
read -r -p 'Execution commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
export PROJECT_DIR="/oscar/home/rlyang/hlt_classification/source/HLT_Classification_fullstat_hybrid_embedding_${COMMIT:0:8}"
ROOT="/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_hybrid_embedding_${COMMIT:0:8}_r1"
export JC2_SITE=oscar_l40s
unset PYTHONHOME PYTHONPATH LD_PRELOAD
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_hybrid_embedding.py" results --spec "${ROOT}/campaign_spec.json"
)
```

This authenticates saved results and prints selected AUC/accuracy/recovery,
R50/R30 and JSON with per-class recoveries/pending stages. No inference or
test access. It includes historical M0HLT/OFFLINE, DIRECT_D000/COARSE_D000,
U100-w5 and four-teacher references. The five new stages use their own recipes
and namespaces; same names in older campaigns are not interchangeable.

Only final DE000 is HLT-deployable. Intermediate richer-input results are oracle
diagnostics. All comparisons are single-seed development validation; the sealed
test is untouched. Local installed-Weaver GPU miniature coverage is not a new
Oscar/Tigris resource certification: the requested no-preflight route checks
and reuses genuine original Oscar evidence, then checks the combined path inside
each scientific job.

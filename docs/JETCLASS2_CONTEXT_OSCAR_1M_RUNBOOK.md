# Queue the unchanged CONTEXT_V1 1M/250k experiment on Oscar

This is a **new campaign**, using the already copied dataset: no data transfer,
new smearing, dataset regeneration, or cancellation of the old 100k run.
There are three admission jobs followed by 16 science jobs (nine fits, five
reducers, aggregate and completion). DIRECT and COARSE only; final test sealed.
Do not use the CONTEXT_V2 helper for this request.

## 1. Publish and pin source

First review/commit/push the implementation from Windows. Do not `git add .`:
this workspace contains unrelated ongoing work. Full-campaign files are
`context_full.py`, `cache_full.py`, the full CLI/helper/test, this runbook,
the 1M plan/full contract, and their explicit version dispatches in shared
release/data/cache/gate/production/submission modules. Inspect shared-file diffs
before committing, especially alongside concurrent CONTEXT_V2 changes.
The clean remote worktree must include every file named by its source lock.

On Oscar (`rlyang@login...`), use the actual Git repository under **source**;
`/oscar/home/rlyang/hlt_classification` is the output base, not the Git repo.

```bash
read -r -p 'New pushed 40-character commit: ' COMMIT
MAIN=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
BASE=/oscar/home/rlyang/hlt_classification
SHORT="${COMMIT:0:8}"
PROJECT="${BASE}/source/HLT_Classification_context1m_${SHORT}"
ROOT="${BASE}/checkpoints/jc2_context_1m250k_direct_coarse_${SHORT}_r1"
CONTAINER=/oscar/home/rlyang/datasets/literature_context_v1_2250k_4f473c3e_r1

(
  set -euo pipefail
  [[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
  git -C "${MAIN}" fetch origin main
  git -C "${MAIN}" cat-file -e "${COMMIT}^{commit}"
  git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
  if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
  else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
  fi
  bash "${PROJECT}/scripts/queue_jetclass2_context_full_ladder.sh" \
    "${COMMIT}" "${CONTAINER}" "${ROOT}" gate
)
```

Stop on an error; preserve existing files. Check `checkquota` before submitting.
The helper activates the registered Oscar environment. Dry review creates the
spec and exact plan, but submits **nothing**. Gate GPU request: one L40S, six
CPUs, 180000 MiB, 12 hours. Foundation uses the same CPU/RAM request without GPU.

## 2. Submit the reviewed gate (survives SSH disconnect)

Run only after the dry plan succeeded and you reviewed its three jobs. Read the
actual saved plan hash rather than pasting a placeholder:

```bash
(
  set -euo pipefail
  PYTHON=/oscar/scratch/rlyang/hlt_classification/environments/envs/atlas_kd_oscar/bin/python
  PLAN_HASH="$("${PYTHON}" -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
    "${ROOT}/gate/command_plan.json")"
  LOG="$(mktemp "${ROOT}/gate-submit.XXXXXX.log")"
  nohup bash -c '
    result=0
    bash "$@" || result=$?
    printf "\nQUEUE_HELPER_EXIT=%s\n" "$result"
    exit "$result"
  ' _ "${PROJECT}/scripts/queue_jetclass2_context_full_ladder.sh" \
    "${COMMIT}" "${CONTAINER}" "${ROOT}" gate --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
  printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
)
```

Inspect the log for the submission ledger and `QUEUE_HELPER_EXIT=0`. A helper
PID is not a Slurm job ID. Do not resubmit simply because the queue is empty.
Use the recorded ledger IDs with `sacct` and inspect errors. The old 100k gate
cannot admit this population; native full-size preflight must pass first.

## 3. Review and submit all science jobs after the gate passes

```bash
bash "${PROJECT}/scripts/queue_jetclass2_context_full_ladder.sh" \
  "${COMMIT}" "${CONTAINER}" "${ROOT}" science
```

This verifies gate completion and prints the 16-job DAG, with walltimes derived
from the full-size preflight. After reviewing that plan:

```bash
(
  set -euo pipefail
  PYTHON=/oscar/scratch/rlyang/hlt_classification/environments/envs/atlas_kd_oscar/bin/python
  PLAN_HASH="$("${PYTHON}" -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
    "${ROOT}/science/command_plan.json")"
  LOG="$(mktemp "${ROOT}/science-submit.XXXXXX.log")"
  nohup bash -c '
    result=0
    bash "$@" || result=$?
    printf "\nQUEUE_HELPER_EXIT=%s\n" "$result"
    exit "$result"
  ' _ "${PROJECT}/scripts/queue_jetclass2_context_full_ladder.sh" \
    "${COMMIT}" "${CONTAINER}" "${ROOT}" science --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
  printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
)
```

There is no automatic follow-up or cancellation. A failed/ambiguous submitter
must be inspected using its retained journal/claim, not retried or deleted blindly.

## Results (including partially completed campaigns)

```bash
(
  export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
  source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
  python -u -s "${PROJECT_DIR}/scripts/jetclass2_context_full_ladder.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
)
```

M0HLT=0%, pure OFFLINE=100% recovery. Intermediate U/D views with offline
information are oracles, not HLT-only deployment results. Final comparison is
DIRECT D000 versus COARSE D000 on the same 250k validation jets. No final-test
evaluation is included. On reconnect, restore the exact COMMIT/PROJECT/ROOT
variables; do not create another campaign merely to recover shell variables.

# Queue the restored Luka regular-teacher campaign

This is **100k TRAIN / 50k validation**, natural class ratios, original split,
DIRECT and COARSE only. All nine models have one encoder. The restored source
is `/home/ryreu/atlas/datasets/cms_fullsim_630k_trial_luka_v1` (890 ROOT files).
No deleted fusion/preflight artifact is needed. No new test evaluation is
implemented: the previous 149553-jet holdout has already been consumed.

Read [the active plan](plans/LUKA_FULLSIM_REGULAR_PLAN.md) and
[cache/campaign contract](contracts/LUKA_FULLSIM_REGULAR.md). Local tests do
not replace the real SPORC acceptance gate below. Nothing has been submitted
by implementing these files.

**Recovering the failed a54c60e5 preflight (21880176)?** Use the
[preflight-only recovery section](#preflight-only-recovery-after-21880176), not
the fresh three-job gate. Keep the original worktree and completed caches.

## 1. Commit and push on Windows

Review changes; do not stage the thousands of unrelated fixtures or raw data.
Use PowerShell in the repository:

```powershell
git diff -- src/hlt_classification/luka_fullsim/particle_audit.py
git add -- src/hlt_classification/luka_fullsim/particle_audit.py `
  src/hlt_classification/luka_fullsim/regular_cache.py `
  src/hlt_classification/luka_fullsim/regular_campaign.py `
  src/hlt_classification/luka_fullsim/regular_preflight.py `
  src/hlt_classification/luka_fullsim/regular_worker.py `
  src/hlt_classification/luka_fullsim/regular_submission.py `
  src/hlt_classification/luka_fullsim/regular_recovery.py `
  scripts/luka_fullsim_regular.py scripts/queue_luka_fullsim_regular.sh `
  scripts/queue_luka_fullsim_regular_recovery.sh `
  sbatch/run_luka_fullsim_regular.sh tests/test_luka_fullsim_regular.py `
  tests/test_luka_fullsim_regular_recovery.py `
  docs/plans/LUKA_FULLSIM_REGULAR_PLAN.md docs/contracts/LUKA_FULLSIM_REGULAR.md `
  docs/LUKA_FULLSIM_REGULAR_RUNBOOK.md docs/LEGACY_SOURCE_MAP.md
git add -p -- docs/HANDOFF.md
git diff --cached --stat
git diff --cached
# Proceed only when the staged diff contains the intended implementation.
git commit -m "Add cached Luka regular direct and coarse campaign"
if ($LASTEXITCODE -ne 0) { throw "Commit failed; stop" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed; stop" }
git rev-parse HEAD
```

## 2. Restore source and dry-review the three gate jobs on SPORC

Paste on **sporcsubmit**, not Tigris/Oscar. The clone is necessary only because
the old repository was deleted; an existing non-repository directory is never
overwritten. The detached worktree isolates this run from later development.

```bash
read -r -p 'Paste the pushed 40-character commit: ' COMMIT
export COMMIT
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve everything and paste the error." >&2' ERR
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_regular_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_regular_${COMMIT:0:8}_r1"
if [[ ! -e "${MAIN}" ]]; then
    git clone https://github.com/ryanlyang/HLT_CLASS.git "${MAIN}"
fi
test "$(git -C "${MAIN}" rev-parse --show-toplevel)" = "${MAIN}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
test -f /home/ryreu/atlas/datasets/cms_fullsim_630k_trial_luka_v1/source.sha256
df -h /home/ryreu/atlas/datasets
quota -s 2>/dev/null || true
bash "${PROJECT}/scripts/queue_luka_fullsim_regular.sh" "${COMMIT}" "${ROOT}" gate
printf '\nCampaign root: %s\n' "${ROOT}"
BASH
```

Review the full plan, particularly paths, account `reu-aisocial`, partition
`debug`, dependencies and resource shapes. `sbatch --test-only` checks each
shape without submitting. Initial requests are conservative bounds, not ETAs:
foundation 30min CPU, prepare 8 CPUs/64000 MiB/8h, preflight 6 CPUs/90000 MiB/A100/8h.
Preparation currently requires ~48 GiB **available quota** as a worst-case
support-512 bound including reserve, even though actual ragged caches should
be much smaller. `df` can describe shared free space rather than your quota.
Confirm quota yourself before proceeding; don't delete data automatically.

## 3. Submit exactly the reviewed gate

This reads the hash from the saved plan, displays it, and requires explicit
confirmation. It does not submit science or silently retry an interrupted
submission. Keep the same `COMMIT` from step 2; if reconnecting, set it again.

```bash
bash <<'BASH'
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_regular_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_regular_${COMMIT:0:8}_r1"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
PLAN_HASH="$(PYTHONNOUSERSITE=1 "${PYTHON}" -s -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
    "${ROOT}/submission/command_plan.json")"
printf 'Reviewed gate hash: %s\nRoot: %s\n' "${PLAN_HASH}" "${ROOT}"
read -r -p 'Submit these three gate jobs? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/gate-submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_regular.sh" \
    "${COMMIT}" "${ROOT}" gate --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nWatch: tail -f "%s"\n' "$!" "${LOG}"
BASH
```

Wait for `QUEUE_HELPER_EXIT=0` and three IDs. `Ctrl-C` stops tail only.
`squeue --me -o "%.18i %.12P %.42j %.2t %.10M %R"` shows `lukar_...` jobs.
`foundation` authenticates the restored snapshot; `prepare` audits and stores
all ordinary views with eight file workers; `preflight` tests the actual cached
input/model path, installed-Weaver parity, batch-256 longest-support stress and
seven full-population technical passes. It does not optimize the recipe.

Preparation prints `LUKA durable files=...`; preflight prints phase boundaries
and 15-second heartbeats. Successful gate publication is
`ROOT/preflight/preflight.json`; inspect its resource flags and projected
train/reducer minutes. Any failure leaves logs and partial outputs intact.
Do not rerun into a failed root or delete a live submission claim.

## 4. After all three gate jobs complete, dry-review science

```bash
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_regular_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_regular_${COMMIT:0:8}_r1"
bash "${PROJECT}/scripts/queue_luka_fullsim_regular.sh" "${COMMIT}" "${ROOT}" science
```

This authenticates the exact gate ledger/IDs, successful accounting, cached
population and preflight evidence. Only then does it register 16 science jobs.
No job is submitted by the dry review. All fits are cold-start, including the
controls. The preflight's technical checkpoints are never imported. Actual
measured requests must fit debug's 24h limit; no silent increase or partition change.

## 5. Submit all 16 science jobs

```bash
bash <<'BASH'
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_regular_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_regular_${COMMIT:0:8}_r1"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
PLAN_HASH="$(PYTHONNOUSERSITE=1 "${PYTHON}" -s -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
    "${ROOT}/science/submission/command_plan.json")"
printf 'Reviewed science hash: %s\n' "${PLAN_HASH}"
read -r -p 'Submit all 16 regular DIRECT/COARSE jobs? Type YES: ' CONFIRM </dev/tty
[[ "${CONFIRM}" == YES ]]
LOG="$(mktemp "${ROOT}/science-submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_regular.sh" \
    "${COMMIT}" "${ROOT}" science --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nWatch: tail -f "%s"\n' "$!" "${LOG}"
BASH
```

After `QUEUE_HELPER_EXIT=0`, the entire registered dependency graph is queued;
SSH need not remain open. Do not also manually submit individual science jobs.

## Results, including partial progress

```bash
bash <<'BASH'
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_regular_${COMMIT:0:8}"
ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_regular_${COMMIT:0:8}_r1"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -u -s "${PROJECT}/scripts/luka_fullsim_regular.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
BASH
```

GPU workers load persisted inputs, not ROOT particles. Reducers load TRAIN
only. Reports include validation accuracy/AUC/rejection50 and recovery relative
to M0HLT=0%, pure OFFLINE=100%, with unfinished rows retained. Direct versus
coarse is a development comparison with unequal training compute, not evidence
from a new blind holdout.

## Preflight-only recovery after 21880176

The original gate is
`/home/ryreu/atlas/HLT_Classification/checkpoints/luka_regular_a54c60e5_r1/gate_spec.json`.
Foundation 21880174 and preparation 21880175 completed; preflight 21880176
failed because validation support exceeded the TRAIN maximum. That comparison
is replaced by separate TRAIN-backward and validation-inference stress tests.
No particles, splits, labels or recipe change. No validation fitting occurs.

First commit/push the explicit files in section 1 (including the new recovery
module, helper and tests). This must produce a **new** commit, not a54c60e5.
Do not edit the old worktree, metadata, source hashes or failed preflight folder.
No preprocessing is rerun and no second copy of the caches is created.

On SPORC, paste the following. Leave `MODE=gate` for the initial recovery.
It dry-reviews **one** A100 preflight job (6 CPUs, 90000 MiB, 8h), then asks
before submitting. A helper PID is not proof of submission: wait for the ledger
and `QUEUE_HELPER_EXIT=0` in the printed log.

```bash
bash <<'BASH'
set -euo pipefail
trap 'echo "Stopped at line ${LINENO}; preserve everything and paste the error." >&2' ERR
MODE=gate
read -r -p 'Paste the new pushed 40-character recovery commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_regular_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_regular_recovery_${COMMIT:0:8}_r1"
ORIGINAL="${MAIN}/checkpoints/luka_regular_a54c60e5_r1/gate_spec.json"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
test -f "${ORIGINAL}"
test -f "$(dirname "${ORIGINAL}")/prepared/cache_manifest.json"
test -x "${PYTHON}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
HELPER="${PROJECT}/scripts/queue_luka_fullsim_regular_recovery.sh"
bash "${HELPER}" "${COMMIT}" "${ORIGINAL}" "${ROOT}" "${MODE}"
PLAN="${ROOT}/submission/command_plan.json"
if [[ "${MODE}" == science ]]; then PLAN="${ROOT}/science/submission/command_plan.json"; fi
PLAN_HASH="$(PYTHONNOUSERSITE=1 "${PYTHON}" -s -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' "${PLAN}")"
printf '\nMode: %s\nRoot: %s\nReviewed plan: %s\n' "${MODE}" "${ROOT}" "${PLAN_HASH}"
read -r -p 'Submit exactly this reviewed plan? Type YES: ' CONFIRM </dev/tty
if [[ "${CONFIRM}" != YES ]]; then echo 'Stopped without submission.'; exit 0; fi
LOG="$(mktemp "${ROOT}/${MODE}-recovery-submit.XXXXXX.log")"
nohup bash "${HELPER}" "${COMMIT}" "${ORIGINAL}" "${ROOT}" "${MODE}" \
    --execute "${PLAN_HASH}" >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nWatch with:\ntail -f "%s"\n' "$!" "${LOG}"
BASH
```

After the replacement preflight is COMPLETED/0:0, inspect its
`ROOT/preflight/preflight.json` (REGULAR_PREFLIGHT/v2): separate role maxima,
inference-only validation stress, technical checks and resource bounds must pass.
Then repeat the same block with **only `MODE=science`** and the same new commit.
That separately dry-reviews/submits the unchanged 16-job regular DIRECT/COARSE
graph using the old caches. The failed old job is not a dependency or acceptance
parent; the new gate binds its replacement ID alongside the two successful old
CPU IDs. No automatic follow-up, cancellation or retry occurs.

For results use the earlier results command, but set
`ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/luka_regular_recovery_${COMMIT:0:8}_r1"`.

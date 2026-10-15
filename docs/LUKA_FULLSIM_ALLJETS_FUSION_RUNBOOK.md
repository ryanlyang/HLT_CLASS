# Luka ALLJETS fusion, both final routes (SPORC debug)

**Current blocker (2026-10-15):** gate21896409 failed in D000/D000 CUDA stress.
Do not submit science or rerun the unchanged full gate. Use the
[targeted diagnostic](#cuda-diagnostic-after-failed-gate21896409) below first.

Authority: [plan](plans/LUKA_FULLSIM_ALLJETS_FUSION_PLAN.md) and
[contracts](contracts/LUKA_FULLSIM_ALLJETS_FUSION.md).

This is an additive continuation of the completed regular ALLJETS campaign:
299,234 TRAIN / 74,895 validation, with the original 50,000 validation rows as
a secondary report using the SAME full-validation-selected checkpoint.
Historical test rows are already in TRAIN. There is **no independent test**.

Keep the original dataset, foundation, cache, parent science results, all parent
worktrees and their journals. The new source imports the nine old comparison
rows and U000 TRAIN bank only after authenticating them with the original pinned
validator. It does not recreate views, refit controls, or import model weights.
Do not change old source to accommodate this continuation.

## Publish source

Review and commit the `alljets_fusion_*.py` modules, the CLI, queue helper,
worker, focused test, this runbook and its plan/contract. Include the applicable
HANDOFF and LEGACY_SOURCE_MAP hunks without staging unrelated work. Push the exact
commit to `origin/main`; copy its full 40-character hash. Nothing here commits,
pushes, cancels or submits automatically.

## Prepare one new GPU gate

On `sporcsubmit`, use a fresh pinned worktree and output directory. The original
regular campaign MUST remain complete and its source clean. The common parent
path below corresponds to the original all-eligible run; check it before use.

```bash
COMMIT=YOUR_NEW_40_CHARACTER_COMMIT
MAIN=/home/ryreu/atlas/HLT_Classification
PARENT="${MAIN}/checkpoints/luka_alljets_9e591c7a_r1/science/campaign_spec.json"
SHORT="${COMMIT:0:8}"
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_alljets_fusion_${SHORT}"
ROOT="${MAIN}/checkpoints/luka_alljets_fusion_${SHORT}_r1"

(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -f "${PARENT}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_fullsim_alljets_fusion.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" gate
)
```

If the parent path differs, locate its existing `science/campaign_spec.json`;
do not create a substitute or point at the old 100k/50k run. Authentication can
take several minutes. The dry plan is `${ROOT}/submission/command_plan.json`.
Review its single job: 6 CPUs, 256000 MiB host RAM, A100, debug, 8 hours, account
`reu-aisocial`, QoS `qos_tier3`. Copy its `content_hash`, not the gate/spec hash:

```bash
PLAN_HASH=YOUR_REVIEWED_64_CHARACTER_PLAN_HASH
LOG="$(mktemp "${ROOT}/gate-submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_alljets_fusion.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" gate --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
```

Inspect that log and the Slurm ledger. This queues ONLY the new fusion resource
gate, not science. No foundations or preparation jobs are needed. No automatic
followup is armed.

## After the gate passes

The successful report is `${ROOT}/preflight/preflight.json`. It must pass
installed-Weaver/native-mask and FP32/BF16 offload parity, longest TRAIN backward
and independent longest-validation inference stress, full-population technical
passes, checkpoint/bank readback, and memory/runtime envelopes. Bad classification
metrics alone never fail admission. The old single-encoder preflight cannot
substitute for this one.

If the unchanged conservative 100-pass projection exceeds debug's 24-hour
ceiling, the gate preserves `measurement.json` and stops. Review resources or a
different partition explicitly; do NOT lower the budget silently.

Create and review the science plan (the helper verifies exact completed gate
accounting and immutable payloads):

```bash
bash "${PROJECT}/scripts/queue_luka_fullsim_alljets_fusion.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" science
```

Review `${ROOT}/science/submission/command_plan.json`: eight fresh fits, six
TRAIN-only reducers, aggregate and complete, **16 jobs total**. Both routes are
included; existing regular jobs and models are unchanged. Each training/reducer
job requests the measured conservative walltime and 256000 MiB RAM. The graph:

```text
imported U000 TRAIN bank
  -> FUSION_U050 -> FUSION_U100 -> FUSION_D066 -> FUSION_D033 -> FUSION_D000
                                                                  |-> FINAL_DIRECT_D000
                                                                  `-> FUSION_D000_D000
                                                                         `-> FINAL_BRIDGE_D000
```

Copy this NEW science plan hash, then submit exactly that plan:

```bash
PLAN_HASH=YOUR_REVIEWED_SCIENCE_PLAN_HASH
LOG="$(mktemp "${ROOT}/science-submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fullsim_alljets_fusion.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" science --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Submission PID: %s\nLog: %s\n' "$!" "${LOG}"
```

Exclusive claims prevent duplicate submissions. If submission is interrupted,
preserve the claim/journal and reconcile exact IDs; do not delete and retry.

## Results

In a shell with the same `COMMIT`, `PROJECT`, `ROOT` definitions:

```bash
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT}/src"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -u -s "${PROJECT}/scripts/luka_fullsim_alljets_fusion.py" results \
    --spec "${ROOT}/science/campaign_spec.json"
```

All nine parent plus eight new rows are shown, including partial-run pending
rows. Primary comparison: M0HLT, regular DIRECT_D000, regular COARSE_D000,
FINAL_DIRECT_D000 and FINAL_BRIDGE_D000, all single-HLT. FUSION_D000 still includes
offline D033 information; FUSION_D000_D000 uses TWO HLT encoders. Their metrics
are diagnostics, not equal-capacity single-HLT deployment comparisons.

Full 74,895 validation is primary; fixed original50k is secondary, not a test or
a second checkpoint-selection opportunity. All recoveries use the same imported
M0HLT/OFFLINE controls within their respective evaluation populations.

## CUDA diagnostic after failed gate21896409

This queues ONE diagnostic, not another full preflight. Reuse the original
TRAIN caches; no data upload, ROOT processing, validation evaluation or full
training epoch. Four fresh processes isolate original asynchronous execution,
blocking CUDA localization, synchronous-restoration D000/D000 and synchronous-
restoration U050/U000. The last two are candidate tests, not an accepted repair.
Synchronous restore passed local small-model parity; the A100 failure is not
yet reproduced locally. The convolution-tiling idea was rejected by parity.

Publish only these new/changed implementation files (plus reviewed documentation
hunks; do not stage unrelated dirty files):

```powershell
git add -- src/hlt_classification/luka_fullsim/alljets_fusion_source.py src/hlt_classification/luka_fullsim/alljets_fusion_diagnostic.py scripts/luka_fusion_cuda_diagnostic.py scripts/queue_luka_fusion_cuda_diagnostic.sh sbatch/run_luka_fusion_cuda_diagnostic.sh tests/test_luka_fusion_cuda_diagnostic.py docs/plans/LUKA_FULLSIM_ALLJETS_FUSION_PLAN.md docs/contracts/LUKA_FULLSIM_ALLJETS_FUSION.md docs/LUKA_FULLSIM_ALLJETS_FUSION_RUNBOOK.md docs/LEGACY_SOURCE_MAP.md
git diff --cached --stat
git commit -m "Add isolated Luka fusion CUDA failure diagnostic"
if ($LASTEXITCODE -ne 0) { throw "Check commit output before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

Stage the applicable HANDOFF hunk separately with `git add -p docs/HANDOFF.md`;
that file contains unrelated pre-existing edits. Do not use `git add .`.

On SPORC, paste the full NEW commit into COMMIT (not e9b0b560):

```bash
COMMIT=YOUR_NEW_40_CHARACTER_COMMIT
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_cuda_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_fusion_cuda_${COMMIT:0:8}_r1"
ORIGINAL="${MAIN}/checkpoints/luka_alljets_fusion_e9b0b560_r1/gate_spec.json"
(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
test -f "${ORIGINAL}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_fusion_cuda_diagnostic.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}"
)
```

Review the printed plan: one debug A100 job, 6 CPUs, 250GiB, **1h limit**, no
dependencies or followup. This is a limit, not a promised runtime. Then:

```bash
PLAN_HASH=YOUR_REVIEWED_64_CHARACTER_PLAN_HASH
(
set -euo pipefail
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fusion_cuda_diagnostic.sh" \
    "${COMMIT}" "${ROOT}" "${ORIGINAL}" --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
)
```

Read `${ROOT}/diagnostic_report.json` after completion. Each case has an `.out`
log plus `cases/CASE/{runtime,witness,result|failure}.json` and
`cases/CASE/operations.jsonl`. These bind the exact operation shapes and strides;
the blocking case should localize an asynchronous CUDA failure. Results may be
printed/authenticated with the same isolated environment and:

```bash
python -u -s "${PROJECT}/scripts/luka_fusion_cuda_diagnostic.py" results \
    --spec "${ROOT}/diagnostic_spec.json"
```

The overall Slurm job deliberately exits FAILED if any case fails, even when
the other cases and summary completed. Inspect the per-case report rather than
assuming a failed reference prevented collection. Even all four passing does
not prove the old transient fault is fixed. No scientific preflight receipt is
written; the original failed gate remains failed. Do not weaken parity,
stress support, full256 batch or100-pass runtime admission to proceed.

The earlier U050 technical pass took about20.7min. Its unchanged conservative
100-pass request would be about60h, beyond debug24h. That independent runtime
issue needs fresh uninstrumented measurements and an explicit execution plan;
instrumented diagnostic timings do not establish a speedup.

## Synchronous production repair after diagnostic21896886

Use this section now, NOT the diagnostic commands above. The original two cases
failed while both synchronous candidates passed. This queues ONE full replacement
preflight, NOT science, using the same existing caches. Preserve both original
worktrees, all failed outputs and the successful regular parent. No uploads or
preprocessing rerun required. Request remains debug/A100/6CPU/250GiB/8h.

Publish the following scoped changes from Windows; review HANDOFF hunks separately
because that file contains unrelated work. Do not `git add .`:

```powershell
$files = @(
    "src/hlt_classification/luka_fullsim/alljets_fusion_sync.py"
    "src/hlt_classification/luka_fullsim/alljets_fusion_recovery.py"
    "src/hlt_classification/luka_fullsim/alljets_fusion_campaign.py"
    "src/hlt_classification/luka_fullsim/alljets_fusion_source.py"
    "src/hlt_classification/luka_fullsim/alljets_fusion_preflight.py"
    "src/hlt_classification/luka_fullsim/alljets_fusion_worker.py"
    "src/hlt_classification/luka_fullsim/alljets_fusion_submission.py"
    "scripts/luka_fullsim_alljets_fusion.py"
    "scripts/queue_luka_fusion_sync_recovery.sh"
    "tests/test_luka_fusion_sync_recovery.py"
    "tests/test_luka_fullsim_alljets_fusion.py"
    "tests/test_luka_fusion_cuda_diagnostic.py"
    "docs/plans/LUKA_FULLSIM_ALLJETS_FUSION_PLAN.md"
    "docs/contracts/LUKA_FULLSIM_ALLJETS_FUSION.md"
    "docs/LUKA_FULLSIM_ALLJETS_FUSION_RUNBOOK.md"
    "docs/LEGACY_SOURCE_MAP.md"
)
git add -- $files
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --stat
git commit -m "Promote synchronous Luka fusion restore with v2 preflight recovery"
if ($LASTEXITCODE -ne 0) { throw "Check commit output before continuing" }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
git rev-parse HEAD
```

Then on SPORC (prompts for the NEW commit, not658cafa5/e9b0b560):

```bash
bash <<'BASH'
set -euo pipefail
trap 'printf "Stopped at line %s. Preserve files and paste the error.\n" "${LINENO}" >&2' ERR
read -r -p 'Paste the NEW 40-character repair commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_luka_sync_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/luka_fusion_sync_${COMMIT:0:8}_r1"
DIAGNOSTIC="${MAIN}/checkpoints/luka_fusion_cuda_658cafa5_r1/diagnostic_spec.json"
PYTHON=/home/ryreu/miniconda3/envs/atlas_kd_sporc/bin/python
test -f "${DIAGNOSTIC}"
test -x "${PYTHON}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_fusion_sync_recovery.sh" \
    "${COMMIT}" "${ROOT}" "${DIAGNOSTIC}"
PLAN_HASH="$("${PYTHON}" -B -s -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["content_hash"])' \
    "${ROOT}/submission/command_plan.json")"
[[ "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
printf '\nReviewed plan: %s\n' "${PLAN_HASH}"
read -r -p 'Submit this ONE full repaired preflight? Type YES: ' CONFIRM </dev/tty
if [[ "${CONFIRM}" != YES ]]; then
    echo 'Stopped. No job submitted.'
    exit 0
fi
LOG="$(mktemp "${ROOT}/submit.XXXXXX.log")"
nohup bash "${PROJECT}/scripts/queue_luka_fusion_sync_recovery.sh" \
    "${COMMIT}" "${ROOT}" "${DIAGNOSTIC}" --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Submission helper PID: %s\nLog: %s\n' "$!" "${LOG}"
printf 'Watch with:\ntail -f "%s"\n' "${LOG}"
BASH
```

Wait for the job ID and `QUEUE_HELPER_EXIT=0`; helper PID alone is not submission.
Preflight progress is in `ROOT/preflight-JOBID.out` and `ROOT/preflight/phases/`.
Successful admission publishes `ROOT/preflight/preflight.json` with contract v2.
If technical checks finish but the conservative full-schedule request exceeds
debug24h, `measurement.json` is preserved and `preflight.json` is NOT published.
Paste that measured request/error; do not lower the schedule/margin or manually
create a receipt. An8h preflight limit does not imply8h scientific fits.

Only after admitted completion may the existing `create-science` and `submit`
commands operate on this NEW v2 root. Its16-job graph uses the synchronous model
for all paired fits/reducers and the unchanged factory for single-HLT endpoints.
Science still requires a separate exact dry review and explicit live permission.

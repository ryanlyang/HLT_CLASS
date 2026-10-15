# Luka ALLJETS fusion, both final routes (SPORC debug)

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

Review and commit the five `alljets_fusion_*.py` modules, the CLI, queue helper,
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

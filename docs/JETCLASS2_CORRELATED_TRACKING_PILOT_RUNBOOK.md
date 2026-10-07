# Correlated tracking: 20k Tigris diagnostic

This is a **new synthetic mechanism test**, not another run of CONTEXT_V1 or a
CMS-fitted proxy. It uses the original 20k OFFLINE pilot sample. Nothing is
submitted automatically; existing campaigns and all validation/test particles
remain untouched. See the [scientific plan](plans/JETCLASS2_CORRELATED_TRACKING_PILOT_PLAN.md)
and [artifact contract](contracts/JETCLASS2_CORRELATED_TRACKING.md).

## Prepare exact source

Commit and push only the new `correlated_tracking/` package, its two scripts,
Slurm worker, two test files, this runbook, new plan/contract, and the scoped
HANDOFF/LEGACY_SOURCE_MAP updates. Preserve unrelated local work. Use the real
40-character pushed commit, not a placeholder, for the following Tigris commands.

```bash
read -r -p 'Pushed 40-character commit: ' COMMIT
(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_corr_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/jc2_correlated_tracking_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/jc2_literature_pilot_08cbf063_r1/study_spec.json"
test -f "${PARENT}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [ -e "${PROJECT}" ]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_jetclass2_correlated_tracking.sh" \
    "${COMMIT}" "${PARENT}" "${ROOT}"
)
```

Review the single `jc2corr_pilot` plan: Tigris / reu-aisocial, CPU-only,
16 CPUs, 64 GiB, 2h. `automatic_followup` must be false. Original pilot banks
must still exist; the Oscar production provenance copy alone is not enough.
Preparation authenticates stored files and may take time. Do not launch a
duplicate helper while one is still running.

## Execute the reviewed plan

In the same shell, with COMMIT still set, paste the actual plan content hash.
The background log survives SSH disconnects.

```bash
read -r -p 'Reviewed 64-character PLAN content_hash: ' PLAN_HASH
(
set -euo pipefail
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ && "${PLAN_HASH}" =~ ^[0-9a-f]{64}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="/home/ryreu/atlas/HLT_Classification_corr_${COMMIT:0:8}"
ROOT="${MAIN}/checkpoints/jc2_correlated_tracking_${COMMIT:0:8}_r1"
PARENT="${MAIN}/checkpoints/jc2_literature_pilot_08cbf063_r1/study_spec.json"
test -f "${ROOT}/command_plan.json"
LOG="$(mktemp "${ROOT}/queue-submit.XXXXXX.log")"
nohup bash -c '
    result=0
    bash "$@" || result=$?
    printf "\nQUEUE_HELPER_EXIT=%s\n" "${result}"
    exit "${result}"
' _ "${PROJECT}/scripts/queue_jetclass2_correlated_tracking.sh" \
    "${COMMIT}" "${PARENT}" "${ROOT}" --execute "${PLAN_HASH}" \
    >"${LOG}" 2>&1 </dev/null &
printf 'Helper PID: %s\nLog: %s\n' "$!" "${LOG}"
)
```

Read the ledger for the actual job ID; a helper PID is not a Slurm job ID.
An empty `squeue` does not imply failure: inspect `sacct -j JOB_ID -X -P` and
`ROOT/slurm-JOB_ID.out`. A nonzero helper exit or retained submission lock
requires inspection before retry; never remove locks blindly.

## Saved diagnostics

The worker prints the complete chart into its Slurm log. For verified results,
activate `/home/ryreu/miniforge3-aarch64/envs/atlas_kd_tigris`, then:

```bash
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -u -s "${PROJECT}/scripts/jetclass2_correlated_tracking.py" results \
    --spec "${ROOT}/study_spec.json"
```

Reset PROJECT and ROOT using the pinned commit after reconnecting (the example
blocks use subshells). Artifacts are `report.json`, `statistics.csv`,
`overlays.pdf`, `mechanism.pdf`, `files/`, `blocks/`, with `receipt.json` last.
Report particle counts, SDs, tails and eligible counts, not just means. Correlated
and independent marginals match in expectation, not exactly in one realization.
Pair summaries are jet-weighted and not confidence intervals.

This job cannot establish a 1.5-point classifier gap or ladder superiority.
Next: inspect the genuine 20k results, then implement the CE-only strength
screen and matched direct/coarse controls on SPORC. No full dataset or SPORC
training queue is created by this helper.

# FullStat CONCAT + FUSION + U100 embedding study

Read the [science plan](plans/LUKA_FULLSTAT_MULTIVIEW_PLAN.md) and
[contracts](contracts/LUKA_FULLSTAT_MULTIVIEW.md) for the registered experiment.
Three fits: fresh CONCAT teacher, fresh cross-attention FUSION teacher, then
fresh HLT-only student predicting U100/CONCAT/FUSION representations.
Student objective is CE + (10/3) times each of the three SmoothL1 losses.
All fits use 3M TRAIN / 500k validation, 20 epochs, batch128. Auxiliary heads
are removed at inference. No separate preflight; technical checks run inside
each job. Existing campaigns and artifacts are untouched.

## Launch after committing and pushing

Commit the new namespace, entrypoints, tests and documentation together and push
to origin/main. Use its full 40-character commit below; never use a dirty source
worktree. Keep unrelated runtime changes out of this commit: the source guard
allows only the new multiview namespace and four entrypoints relative to74cba41f.
The exact completed corrected readout parent is required (db9afe8b,
job8118608), which authenticates the historical U100-w5 and original U100 bank.
Use a fresh root and at least 22 GiB of available personal result quota; `df`
alone does not measure your quota. `checkquota` gives the personal allowance.

On Oscar, replace COMMIT, REPO if necessary, and quota with available space:

```bash
bash <<'BASH'
set -euo pipefail
COMMIT=PASTE_FULL_PUSHED_COMMIT
REPO=/oscar/home/rlyang/hlt_classification/source/HLT_Classification
PROJECT=/oscar/home/rlyang/hlt_classification/source/HLT_Classification_multiview_${COMMIT:0:8}
ROOT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_multiview_${COMMIT:0:8}_r1
PARENT=/oscar/home/rlyang/hlt_classification/checkpoints/luka_fullstat_readout_db9afe8b_r1/campaign_spec.json
QUOTA=22
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
git -C "${REPO}" fetch origin main
git -C "${REPO}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ ! -e "${PROJECT}" ]]; then
    git -C "${REPO}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/launch_luka_fullstat_multiview.sh" \
    "${COMMIT}" "${ROOT}" "${PARENT}" "${QUOTA}" \
    --authorize AUTHORIZE_FULLSTAT_MULTIVIEW_AUTOSUBMIT_V1
BASH
```

After **DETACHED** you may disconnect. The foreground tail only displays the
log; Ctrl-C stops that display. The detached process authenticates parents,
writes the dry plan, then submits the authorized three fits. Look for
`LAUNCH_COMPLETE` and `LAUNCHER_EXIT=0` in `${ROOT}.launch/launcher.log`. DETACHED confirms
process launch, not successful job submission. There are no follow-up prompts.

Requests per job: Oscar L40S, 6 CPUs, 192000 MiB, 24h. CONCAT runs first,
FUSION after it ends; the student requires both teachers to complete, including
their TRAIN target exports. A poor finite teacher result does not block training.
Technical failure does. Do not delete claims, rerun a failed launcher or manually
submit science without first inspecting failure.json and submission journals.

## Inspect

```bash
tail -n 60 -f "${ROOT}.launch/launcher.log"
squeue --me -o '%.18i %.12P %.45j %.2t %.10M %R'
grep -h 'FULLSTAT MULTIVIEW' "${ROOT}"/*.out | tail -n 40
```

Saved result comparison (authenticates payloads, so may take several minutes):

```bash
export PROJECT_DIR="${PROJECT}" JC2_SITE=oscar_l40s
source "${PROJECT_DIR}/sbatch/luka_splithead_finalize_common.sh"
python -u -B -s "${PROJECT_DIR}/scripts/luka_fullstat_multiview.py" \
    results --spec "${ROOT}/campaign_spec.json"
```

The JSON includes selected epochs, AUC/accuracy/recovery and R50/R30/recovery,
labeling privileged teachers. Incomplete tasks are explicit. No test inference.
Every epoch saves full model/optimizer/RNG state; publication failure preserves
these states but needs an explicit recovery, never a blind resubmission.

Manual dry-review alternative: `queue_luka_fullstat_multiview.sh COMMIT ROOT
PARENT QUOTA`. Its `--execute PLAN_HASH` mode requires the exact reviewed hash.
Never mix this with a launched detached controller for the same root.

Local fixture/installed-Weaver verification is distinct from real Oscar
acceptance. No new full-size resources or completed science results are claimed
until the jobs have run. Startup refuses oversized memory/time estimates and
never silently crops particles, lowers batch size or changes the schedule.

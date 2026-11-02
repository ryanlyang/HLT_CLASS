# Queue and inspect the four U100 correction runs

This campaign adds four retained particle/pair correction methods to the same
CE1 + U100 embedding5 objective. It reuses the completed U100-w5 control from
job21993056 and its existing target bank. There is no control retraining,
teacher retraining, D066 arm or standalone preflight.

The [scientific plan](plans/LUKA_U100_PARTICLE_PAIR_CORRECTION_PLAN.md) and
[v1 contract](contracts/LUKA_U100_PAIR_CORRECTION.md) define the experiment.
Local implementation is ready for a clean pushed source and SPORC shared setup;
this document is not evidence that a remote job has run or passed.

## What gets queued

1. `lukapc_setup`: authenticate the completed control and U100 payload, build
   compact matched U100 particle targets, and measure all four real workers.
2. `lukapc_p4_pair_u100` (C): particle prediction with particle and derived-pair loss.
3. `lukapc_pair_u100` (A): direct pair prediction.
4. `lukapc_dual_u100` (D): independent particle/direct-pair predictors, after C.
5. `lukapc_p4_u100` (B): particle prediction with particle loss, after A.
6. `lukapc_report`: saved validation comparison after all four terminate.

There are two fit lanes. Ordering uses `afterany`; a failed fit does not block
its independent successor. Setup failure stops automatic science submission.
The dispatcher survives logout and submits science only after exact setup
completion and resource acceptance within the reviewed envelope.

## Commit the additive implementation locally

The workspace has unrelated edits. Review these new paths and stage only this
campaign. Stage shared README/HANDOFF/source-map edits by hunk if desired.

```powershell
git add -- src/hlt_classification/luka_u100_pair_correction `
  scripts/luka_u100_pair_correction.py scripts/queue_luka_u100_pair_correction.sh `
  sbatch/run_luka_u100_pair_correction.sh tests/test_luka_u100_pair_correction.py `
  docs/contracts/LUKA_U100_PAIR_CORRECTION.md docs/LUKA_U100_PAIR_CORRECTION_RUNBOOK.md `
  docs/plans/LUKA_U100_PARTICLE_PAIR_CORRECTION_PLAN.md
git diff --cached --check
git diff --cached --stat
```

After reviewing the staged change and committing any intended shared-document
hunks, commit and push the branch that will reach `origin/main`. The source guard
requires descendant source of431341db5eced8eaa3e52b37ba047f8cd4beed50 and permits
runtime changes only inside the new campaign. Unrelated runtime changes need
their own reviewed baseline; do not weaken the guard or discard them.

```powershell
git commit -m "Add four retained U100 particle and pair correction fits"
git push origin main
git rev-parse HEAD
```

## Queue on SPORC

Paste the full pushed40-character commit when prompted:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'New pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_u100pairs_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_pair_correction.sh"
BASH
```

The wrapper discovers the contrastive campaign containing the exact job21993056
log, then authenticates the saved spec, original-source receipts and target bytes.
If discovery is ambiguous, pass the actual spec explicitly:

```bash
bash "${PROJECT}/scripts/queue_luka_u100_pair_correction.sh" \
  --reference /absolute/path/to/completed/contrastive/campaign_spec.json
```

Enter personally available storage (at least8GiB), review the printed commands,
paste their `PLAN_HASH`, and type `YES`. Those two confirmations authorize the
concrete dry-reviewed automatic graph, as required by this repository's
research-compute rules. The fixed campaign allowance stays8GiB even if more
space is entered; an additional1GiB free headroom is checked.

The wrapper prints a dispatcher PID, campaign root and `tail -f` command.
You can log off once it has returned and the detached log reports the setup ID.
The log later says all four students and the report are queued. Until then the
detached dispatcher must remain running on the submit host. Do not rerun the
wrapper or delete claims if submission appears slow: inspect the log/journal.

The root is:

```text
/home/ryreu/atlas/HLT_Classification/checkpoints/
  luka_u100_pair_correction_<commit8>_debug_r1/
```

SPORC/debug requests are one A100, six CPUs and120000MiB. Setup has a12h
envelope. Each fit gets a measured request bounded by24h. The report has2h.
Worst-support stress is used for memory feasibility, not extrapolated as the
cost of every training batch. No support truncation or batch reduction is allowed.

## Results and progress

Use the exact campaign root printed by submission:

```bash
ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/luka_u100_pair_correction_COMMIT8_debug_r1
SPEC="${ROOT}/campaign_spec.json"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PROJECT="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PYTHONPATH="${PROJECT}/src"
python -u -B -s "${PROJECT}/scripts/luka_u100_pair_correction.py" results --spec "${SPEC}"
```

This command authenticates saved results and prints completed rows and pending
arms. It performs no new inference or submission. It compares full-validation
AUC, accuracy, recovery and delta against `U100_W5_MATCHED`; original50k is a
separate report-only table using the same selected checkpoint. Detailed saved
RESULTS contain per-class AUC and QCD rejection at30%/50% signal efficiency.
Zero QCD passing is censored, not an infinite measured rejection.

During fitting, `ARM/fit/epochs/NN.json` contains each completed epoch and
`NN-probe.json` contains TRAIN diagnostics. Slurm output includes
`PAIR_CORRECTION ARM epoch=... batch=... progress=...`, epoch AUC, running best
AUC and recovery. Progress is the latest logged completed training batch, not
an instantaneous reading. Logs are provisional until the task receipt exists.

If a job fails, preserve `claims`, `submission_*`, setup diagnostics and fit
outputs. Cancel only exact IDs from this campaign's submission journals if
needed; this implementation does not cancel any existing job or blindly retry.
An incomplete arm remains in the report's pending list with its accounting
status available through Slurm; poor finite performance still completes.

## Code map and reuse

| File | Purpose |
| --- | --- |
| [contracts.py](../src/hlt_classification/luka_u100_pair_correction/contracts.py) | Four arms, loss budgets, seeds, schedules and artifact identities |
| [source.py](../src/hlt_classification/luka_u100_pair_correction/source.py) | Clean pushed source and original-source control/bank authentication |
| [targets.py](../src/hlt_classification/luka_u100_pair_correction/targets.py) | HLT/U100 view replay, exact target reuse, matched ragged particles and TRAIN scales |
| [geometry.py](../src/hlt_classification/luka_u100_pair_correction/geometry.py) | Physical decoder, wrapped coordinates and differentiable native pair features |
| [model.py](../src/hlt_classification/luka_u100_pair_correction/model.py) | Block1 predictor tap, bounded bias in blocks2–7, retained inference model |
| [losses.py](../src/hlt_classification/luka_u100_pair_correction/losses.py) | Explicit particle/pair gradient paths and per-jet reductions |
| [training.py](../src/hlt_classification/luka_u100_pair_correction/training.py) | Full30-epoch fits, probes, selected deployment and reload checks |
| [resources.py](../src/hlt_classification/luka_u100_pair_correction/resources.py) | Shared real miniatures, stress/timing and admission |
| [submission.py](../src/hlt_classification/luka_u100_pair_correction/submission.py) | Dry graph, exact journals, detached dispatch and two lanes |
| [worker.py](../src/hlt_classification/luka_u100_pair_correction/worker.py) | Setup/fit/report ownership and authenticated comparisons |

`new_model(imported, arm, statistics)` creates a fresh correction student.
`load_deployment(state)` checks that all retained parameters are present,
removes the embedding predictor and sets alpha1. Use the `statistics` stored in
`fit/deployment.json`, then load `fit/selected.pt`. Deployment forward accepts
only `features`, `vectors`, `mask`. Do not use the older bare native-model loader:
it would omit the retained correction. Changing taps, geometry, loss weights,
reference population or schedule requires a new registered scientific recipe.

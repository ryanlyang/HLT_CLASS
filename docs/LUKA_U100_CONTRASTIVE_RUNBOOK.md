# U100-w5 plus paired OFFLINE identification: debug runbook

This implements the [twenty-run plan](plans/LUKA_U100_OFFLINE_CONTRASTIVE_PLAN.md)
under the [new v1 contract](contracts/LUKA_U100_CONTRASTIVE.md).
It uses the original small-data 299,234 TRAIN / 74,895 validation population,
thirty epochs and physical batch128. This is separate from the Oscar high-data
campaign. No recovery improvement is assumed.

Every student keeps **CE1 + U100 embedding5**. Eighteen fits add a separate
OFFLINE identification head, with the registered mixed/random/cross/hard/kinmix
samplers and weights; two are fresh U100-only controls. All students start
from scratch. Both auxiliary heads are removed for native HLT inference.
The first two submitted fits are **mix_l0p3_s0** and **u100_w5_s0**.

## What the queue helper does

1. Requires clean, committed source already reachable from fetched `origin/main`.
2. Authenticates original fixed30 M0HLT/OFFLINE, regular COARSE_U100, and completed
   direct-U100-w5 results in their original source. It permits a retired historical
   TRAIN payload only through the existing result-only import contract.
3. Registers one new output root and the complete twenty-fit envelope for review.
4. Launches a detached submit-host dispatcher. It queues **one shared setup job**
   on debug. Setup reuses the original U100 bank if still valid; otherwise it
   rebuilds that compact bank once. It exports OFFLINE keys once, checks teacher
   readback, capacities, native baseline parity, full-candidate miniatures and
   resource usage. There is no separate preflight or new teacher training job.
5. After successful setup, the dispatcher automatically queues **twenty students
   and one report** using measured walltimes within the reviewed envelope.
   No second confirmation or manual wait for setup is needed.

Total: **22 Slurm jobs**, including setup and report; exactly twenty scientific
student fits. The dispatcher is a detached submit-host process, not another
GPU job. It survives logout with `nohup`, but a submit-host reboot can stop it;
do not blindly restart a partial submission. Inspect its saved log and journal.

Two alternating `afterany` lanes enforce at most two simultaneous fits for this
study. They preserve rank order within each lane; cluster availability determines
actual start times across lanes and other users. A failed student releases its
lane for the next independent fit. A failed setup prevents science submission.
The report waits for all attempts and marks missing receipts incomplete.

Resource requests: six CPUs, 120,000 MiB RAM, one A100, debug,
`reu-aisocial`, `qos_tier3`, `atlas_kd_sporc`. Setup has twelve hours; fits have
measured requests capped at twenty-four hours. The four GiB campaign budget
does not expand to the entire personally available quota. Preserve one GiB
of filesystem headroom. No files are deleted automatically.

## Commit and push locally

Review existing unrelated changes before committing. The new runtime is entirely
under the paths below; historical runtime must remain unchanged from the reviewed
base `8ad0b381547134569818b4258f3cd31b509e9efb`.

```powershell
git add -- src/hlt_classification/luka_u100_contrastive `
  scripts/luka_u100_contrastive.py scripts/queue_luka_u100_contrastive.sh `
  sbatch/run_luka_u100_contrastive.sh tests/test_luka_u100_contrastive.py `
  docs/plans/LUKA_U100_OFFLINE_CONTRASTIVE_PLAN.md `
  docs/contracts/LUKA_U100_CONTRASTIVE.md docs/LUKA_U100_CONTRASTIVE_RUNBOOK.md

# Review/stage the new campaign entries, preserving unrelated documentation work.
git add -p -- docs/HANDOFF.md docs/LEGACY_SOURCE_MAP.md docs/plans/README.md
git diff --cached --check
if ($LASTEXITCODE -ne 0) { throw 'Resolve the staged whitespace errors first.' }
git diff --cached --stat
git commit -m "Add U100 paired OFFLINE identification debug campaign"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed.' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed.' }
git rev-parse HEAD
```

The source guard intentionally rejects unrelated runtime edits in this launch
commit relative to its reviewed base. Use a clean task-specific commit/worktree
if other runtime work has accumulated; do not reset or overwrite it.

## Queue on SPORC

Run from the SPORC submit host, entering the pushed full commit:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Pushed 40-character commit: ' COMMIT </dev/tty
[[ "${COMMIT}" =~ ^[0-9a-f]{40}$ ]]
MAIN=/home/ryreu/atlas/HLT_Classification
PROJECT="${MAIN}_u100contrast_${COMMIT:0:8}"
git -C "${MAIN}" fetch origin main
git -C "${MAIN}" merge-base --is-ancestor "${COMMIT}" origin/main
if [[ -e "${PROJECT}" ]]; then
    test "$(git -C "${PROJECT}" rev-parse HEAD)" = "${COMMIT}"
    test -z "$(git -C "${PROJECT}" status --porcelain --untracked-files=all)"
else
    git -C "${MAIN}" worktree add --detach "${PROJECT}" "${COMMIT}"
fi
bash "${PROJECT}/scripts/queue_luka_u100_contrastive.sh"
BASH
```

The helper asks for personally available GiB, prints the full command plan,
and requests its hash and `YES` for that concrete campaign. These are the
repository's exact-source/dry-review submission requirements, not an extra
scientific preflight. After it prints the dispatcher PID and log path, the
dispatcher is detached and can finish queuing after logout. Before leaving,
check the printed log for `Setup submitted as ...` to confirm launch succeeded.

Expected job prefix: `lukaoc_`. Initially only `lukaoc_setup` appears in Slurm.
That is expected; the complete scientific DAG is submitted after real measured
admission. Watch the printed dispatcher log and `setup-<jobid>.out`.

```bash
squeue --me -o "%.18i %.12P %.36j %.2t %.10M %.12l %R"
```

No old job is cancelled by this helper. Any later cancellation must use exact
IDs from this campaign's ledgers.

## Results and live progress

Use the root printed by the queue helper:

```bash
bash <<'BASH'
set -euo pipefail
read -r -p 'Printed campaign root: ' ROOT </dev/tty
SPEC="${ROOT}/campaign_spec.json"
test -s "${SPEC}"
source /home/ryreu/miniconda3/etc/profile.d/conda.sh
conda activate /home/ryreu/miniconda3/envs/atlas_kd_sporc
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
PROJECT="$(python -B -s -c 'import json,sys; print(json.load(open(sys.argv[1]))["project_dir"])' "${SPEC}")"
export PYTHONPATH="${PROJECT}/src"
python -u -B -s "${PROJECT}/scripts/luka_u100_contrastive.py" results --spec "${SPEC}"
BASH
```

Complete rows are authenticated from native selected predictions and reports.
The primary delta is against the fresh U100-only control of the same replicate;
if that control is incomplete, the delta stays unresolved. Historical recovery
uses the original M0HLT/OFFLINE gap. Original50k is overlapping and report-only.
The final `report/results.json` retains full metric dictionaries, including
accuracy, class AUC, rejection and zero-passing censoring.

Live logs contain epoch AUC, recovery, best epoch, raw/weighted losses, matching
weight, progress and memory peaks. Nonzero fits also publish immutable
`<arm>/fit/epochs/NN.json` and `NN-probe.json`. Probes use a separate deterministic
TRAIN candidate stream and do not update the optimizer. Control logs come from
the original U100 loop, with observation-only progress/recovery annotations;
their complete history is published in the final training report. Live values
are provisional until the task receipt authenticates the completed fit.

## Code map

| File | Responsibility |
| --- | --- |
| [contracts.py](../src/hlt_classification/luka_u100_contrastive/contracts.py) | Exact twenty arms, weights, seeds, sampler definitions, ramp and v1 schemas |
| [source.py](../src/hlt_classification/luka_u100_contrastive/source.py) | Additive source boundary and original-source historical imports |
| [model.py](../src/hlt_classification/luka_u100_contrastive/model.py) | Independent RNG-isolated matching head; native deployment stripping |
| [losses.py](../src/hlt_classification/luka_u100_contrastive/losses.py) | FP32 cosine-softmax identification alongside unchanged U100 regression |
| [sampling.py](../src/hlt_classification/luka_u100_contrastive/sampling.py) | Class quotas, deterministic G/S/H/X/K candidates, exclusions, bounded GPU mining |
| [targets.py](../src/hlt_classification/luka_u100_contrastive/targets.py) | Frozen TRAIN OFFLINE keys, moments, exact joins and HLT kinematic summaries |
| [training.py](../src/hlt_classification/luka_u100_contrastive/training.py) | Original zero kernel, new loss loop, isolated diagnostics, native checkpoint validation |
| [resources.py](../src/hlt_classification/luka_u100_contrastive/resources.py) | Full-candidate real miniatures, baseline tensor parity, sampling-inclusive timing and admission |
| [worker.py](../src/hlt_classification/luka_u100_contrastive/worker.py) | Shared setup, exact task ownership, production fits and results |
| [campaign.py](../src/hlt_classification/luka_u100_contrastive/campaign.py) | Immutable scientific/resource registration and storage checks |
| [submission.py](../src/hlt_classification/luka_u100_contrastive/submission.py) | Full review, detached automatic dispatch, exact journals and two lanes |
| [CLI](../scripts/luka_u100_contrastive.py) / [queue helper](../scripts/queue_luka_u100_contrastive.sh) / [worker shell](../sbatch/run_luka_u100_contrastive.sh) | Thin entrypoints and explicit SPORC environment |
| [tests](../tests/test_luka_u100_contrastive.py) | Samplers, gradients, true Weaver CPU/CUDA fits, native parity and submission failures |

## Failure handling and portability

Do not rerun the queue helper against an existing root. Exclusive claims and
journals intentionally prevent blind duplicate submissions. A partial dispatcher
or task needs inspection of exact accepted IDs and preserved outputs; this
version does not automatically cancel, requeue, overwrite or restart partial fits.

Setup publishes `resource_diagnostic.json` before a resource admission rejection.
Inspect measured representative times separately from longest-support stress.
Do not silently lower batch size, support, negative count, precision or epochs.
If a bound needs revision, record and test a new explicit execution policy.

To port the idea, retain the separate U100 and identification heads and losses,
but authenticate that environment's teachers, IDs and TRAIN moments and measure
its own workload. Do not reuse this screen's small-data recovery denominator or
assume its best weight transfers. Keep a matched U100-w5 control and an HLT-only
deployment check.

# Tigris Research-Compute Runbook

## Site defaults

```text
SSH host       ryreu@tigris.rc.rit.edu
project        /home/ryreu/atlas/HLT_Classification
JetClass data  /home/ryreu/atlas/PracticeTagging/data
outputs        /home/ryreu/atlas/HLT_Classification/checkpoints
logs           /home/ryreu/atlas/HLT_Classification/logs
conda base     /home/ryreu/miniforge3-aarch64
conda env      atlas_kd_tigris
Slurm account  reu-aisocial
partition      tigris
GPU request    gpu:gh200:1
```

These paths are also serialized in `configs/tigris.yaml`. Resources and
walltimes must be selected from measurements; they are not fixed by this
document.

## Required activation

Run this before any Python import, including submit-side utilities:

```bash
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd /home/ryreu/atlas/HLT_Classification
```

Preflight:

```bash
python -s -c '
import sys
assert sys.version_info[:2] == (3, 10)
import awkward_cpp, awkward, uproot, torch
print(sys.executable)
print(awkward_cpp.__file__)
'
```

`PYTHONNOUSERSITE=1` prevents incompatible packages in `~/.local` from
shadowing the conda environment. Prepending `${CONDA_PREFIX}/lib` prevents the
system `libstdc++` from causing the previously observed `GLIBCXX` failure.

If a campaign compiles PyTorch extensions:

```bash
python -c 'from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability()'
```

Do not assume `pytest` is installed on Tigris. Production acceptance must have
standalone runtime/parity commands as well as local unit tests.

## Source workflow

Production uses exact pushed source:

1. commit and push the intended code;
2. on Tigris, use `git pull --ff-only`;
3. verify `git status --short` is empty;
4. record `git rev-parse HEAD`;
5. create the campaign from that exact source;
6. never edit or pull the worker source while that campaign is active.

The production implementation should ultimately use a clean detached worktree
per source commit.

## Data and storage preflight

JetClass is read-only:

```bash
test -d /home/ryreu/atlas/PracticeTagging/data
find /home/ryreu/atlas/PracticeTagging/data -type f -name '*.root' -print -quit
df -B1 /home/ryreu
```

Do not write caches beside the source ROOT files. Before a full campaign,
measure representative cache/checkpoint sizes and project peak concurrent
storage, not only retained final size.

## Slurm worker rules

Workers must source the common helper by the absolute project path:

```bash
PROJECT_DIR="${PROJECT_DIR:-/home/ryreu/atlas/HLT_Classification}"
source "${PROJECT_DIR}/sbatch/common.sh"
```

Never locate it from `$0` or `BASH_SOURCE`; Slurm may execute a copied script
under `/var/spool/slurmd/`.

Submitters must:

- use `sbatch --parsable` and validate numeric job IDs;
- use `afterok` only for genuine artifact/runtime dependencies;
- use bounded arrays;
- export `PROJECT_DIR`, `CAMPAIGN_ROOT`, `CAMPAIGN_ID`, and task identity;
- write logs and job ledgers inside the campaign root;
- authenticate reusable completions;
- reconstruct failed dependency graphs through an explicit resume plan.

Scientific underperformance exits zero and writes metrics. Runtime or lineage
failure exits nonzero.

## Campaign modes

```bash
bash sbatch/submit_baseline.sh --campaign-spec SPEC --dry-run
bash sbatch/submit_baseline.sh --campaign-spec SPEC --smoke-simulate
bash sbatch/submit_baseline.sh --campaign-spec SPEC --smoke-submit
bash sbatch/submit_baseline.sh \
  --campaign-spec PRODUCTION_SPEC \
  --storage-measurement STORAGE_JSON \
  --resource-evidence SMOKE_RESOURCES_JSON \
  --full-production-submit
```

The first three modes require a smoke specification. The full-production mode
requires a production specification created with `--authorize-production`
and a campaign-bound storage measurement. `--dry-run` and
`--smoke-simulate` do not create campaign directories or ledgers.

## Monitoring

```bash
squeue --me -o "%.18i %.9P %.35j %.2t %.10M %.6D %R"

sacct -j JOB_ID -P \
  -o JobID,JobName,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,MaxVMSize,AveCPU,TotalCPU,NodeList
```

Inspect campaign-local `.out` and `.err` logs. Cancel only exact job IDs known
to belong to the campaign.

Generate an authenticated monitor report and explicit recovery plan:

```bash
python -s scripts/monitor_campaign.py \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --submission-ledger "${CAMPAIGN_ROOT}/ledgers/submission.json" \
  --output "${CAMPAIGN_ROOT}/ledgers/monitor.json"
python -s scripts/resume_campaign.py \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --submission-ledger "${CAMPAIGN_ROOT}/ledgers/submission.json" \
  --monitor-report "${CAMPAIGN_ROOT}/ledgers/monitor.json" \
  --output "${CAMPAIGN_ROOT}/ledgers/resume_plan.json"
python -s scripts/capture_slurm_resources.py \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --submission-ledger "${CAMPAIGN_ROOT}/ledgers/submission.json" \
  --monitor-report "${CAMPAIGN_ROOT}/ledgers/monitor.json" \
  --output "${CAMPAIGN_ROOT}/ledgers/slurm_resources.json"
```

Review the plan before adding `--submit`. If it lists stale pending jobs,
`--cancel-stale-exact` cancels only the serialized exact IDs.

The resource capture fails unless every smoke task is completed and
artifact-reusable and `sacct` supplies positive elapsed, MaxRSS, and CPU
measurements. A full-production storage preflight and submission both require
this exact evidence artifact.

The training worker also publishes
`reports/training_runtime_environment.json` before it trains. This immutable
report binds Python/package versions, CUDA runtime, and visible GPU identity
to the campaign and source snapshot.

## Recovery principles

- Validate already completed artifacts before reuse.
- Resume shard-by-shard; do not restart days of valid preprocessing.
- Submit failed nodes and descendants into a new recovery ledger.
- If submission itself stops midway, assemble its per-job journal with
  `scripts/assemble_submission_journal.py`; never rerun the original
  submitter while an incomplete journal exists.
- Do not use broad `scontrol release` to bypass failed prerequisites.
- `DependencyNeverSatisfied` requires a repaired dependency graph, not merely
  releasing downstream jobs.

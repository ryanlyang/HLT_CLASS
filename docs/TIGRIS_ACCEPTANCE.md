# Tigris Acceptance Procedure

This is the remaining Transfer-Block-5/8 acceptance procedure. It submits only
the real miniature, never the full production campaign.

## 1. Transfer the exact repository

From Windows PowerShell, transfer the generated bundle (its SHA-256 is
reported in the handoff that created it):

```powershell
scp artifacts/repository_transfer/HLT_Classification.bundle `
  ryreu@tigris.rc.rit.edu:/home/ryreu/atlas/
```

On Tigris, clone only if the destination is absent:

```bash
test ! -e /home/ryreu/atlas/HLT_Classification
git clone /home/ryreu/atlas/HLT_Classification.bundle \
  /home/ryreu/atlas/HLT_Classification
cd /home/ryreu/atlas/HLT_Classification
git status --short
git rev-parse HEAD
```

If the destination already exists, stop and inspect it. Do not overwrite,
reset, or delete an unknown worktree.

## 2. Activate and validate the runtime

```bash
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd /home/ryreu/atlas/HLT_Classification

bash -n sbatch/*.sh
python -s -c '
import sys
assert sys.version_info[:2] == (3, 10)
import awkward_cpp, awkward, uproot, torch
print(sys.executable)
print(awkward_cpp.__file__)
print(torch.__version__, torch.cuda.is_available())
'
python -s scripts/validate_weaver_parity.py --device cpu
```

The last command must exit zero and print `"passed": true`. Do not mark
Block 5 accepted otherwise.

## 3. Create and inspect the smoke campaign

The campaign specification is written below `checkpoints/`, which is ignored
by Git and therefore does not dirty the source snapshot.

```bash
mkdir -p checkpoints/specs
python -s scripts/create_campaign.py \
  --mode smoke \
  --output checkpoints/specs/hlt_part_smoke.json

bash sbatch/submit_baseline.sh \
  --campaign-spec checkpoints/specs/hlt_part_smoke.json \
  --dry-run

bash sbatch/submit_baseline.sh \
  --campaign-spec checkpoints/specs/hlt_part_smoke.json \
  --smoke-simulate \
  --simulate-failure-task hlt_cache
```

Both commands are nonmutating with respect to campaign artifacts and Slurm.
Inspect that the graph contains preflight, split construction, two-role
offline and HLT cache arrays, Weaver parity, an update-one interruption,
resumed training, and model-validation evaluation.

## 4. Submit only the miniature

```bash
bash sbatch/submit_baseline.sh \
  --campaign-spec checkpoints/specs/hlt_part_smoke.json \
  --smoke-submit
```

Record the printed campaign root and exact job IDs. Monitor with:

```bash
squeue --me -o "%.18i %.9P %.35j %.2t %.10M %.6D %R"
sacct -j COMMA_SEPARATED_EXACT_IDS -P \
  -o JobID,JobName,State,ExitCode,Elapsed,Timelimit,AllocCPUS,ReqMem,MaxRSS,MaxVMSize,AveCPU,TotalCPU,NodeList
```

After all jobs are terminal:

```bash
CAMPAIGN_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/CAMPAIGN_ID
python -s scripts/monitor_campaign.py \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --submission-ledger "${CAMPAIGN_ROOT}/ledgers/submission.json" \
  --output "${CAMPAIGN_ROOT}/ledgers/monitor.json"
python -s scripts/resume_campaign.py \
  --campaign-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --submission-ledger "${CAMPAIGN_ROOT}/ledgers/submission.json" \
  --monitor-report "${CAMPAIGN_ROOT}/ledgers/monitor.json" \
  --output "${CAMPAIGN_ROOT}/ledgers/resume_plan.json"
```

An entirely successful graph must report every task reusable and no rerun
tasks. A runtime failure produces a new exact failed-node-and-descendant plan;
scientific underperformance still exits successfully and remains in the
metrics report.

## 5. Return acceptance evidence

Record in `docs/HANDOFF.md`:

- exact Git commit and clean status;
- Python, Weaver, PyTorch, CUDA, and GPU identity;
- campaign root and every exact job ID;
- complete `sacct` resource rows;
- real-Weaver parity report;
- cache audit, training resume, prediction, and metric artifact hashes;
- final miniature metrics;
- dry-run graph result and any recovery ledger.

Do not use these commands to submit a full production campaign.

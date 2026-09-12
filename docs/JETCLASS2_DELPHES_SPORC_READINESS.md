# Delphes SPORC readiness handoff

This procedure queues **no scientific fits** and changes no existing jobs.
The selected profile is TRAIN_500K, with fixed 1M validation / 1M sealed test.
Raw ROOT files already exist on shared RC storage. Only compact split metadata
and pushed source are needed. The active scientific authority is the
[migration plan](plans/JETCLASS2_DELPHES_DATASET_MIGRATION_IMPLEMENTATION_PLAN.md).

## 1. Windows: scoped commit and push

The workspace also contains unrelated Scouting/salience changes. Do not use
`git add .` or `git add -A`. The staging helper requires an initially empty
index and excludes those files, the salience section of HANDOFF, and the
mixed README/plan-index edits. It leaves the working copies untouched.

```powershell
cd C:\Users\22rya\ComputerScience\CERN\HLT_Classification
& .\scripts\stage_jetclass2_delphes_sporc.ps1
if ($LASTEXITCODE -ne 0) { throw 'Inspect staging before continuing' }
git commit -m "Add JetClass2 Delphes migration and isolated SPORC readiness"
if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
git push origin HEAD:main
if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
git rev-parse HEAD
```

The staging helper itself never commits, pushes or changes remote jobs.
Inspect its printed staged-file summary before the commit. Leave the local
ignored inventory and split artifacts where they are; no ROOT file belongs
in Git. Production uses a separate clean worktree, not this dirty checkout.

## 2. Windows: transfer only the fixed split bundle

```powershell
$Jc2Bundle = 'C:\Users\22rya\ComputerScience\CERN\HLT_Classification\artifacts\jetclass2_delphes_scaling_splits_20260911_v1'
scp -r $Jc2Bundle 'ryreu@sporcsubmit.rc.rit.edu:/home/ryreu/atlas/datasets/jetclass2_10M_20260910/'
if ($LASTEXITCODE -ne 0) { throw 'Split metadata transfer incomplete; keep partial files' }
```

This is about 14.42 MiB, not another 23.68-GiB raw-data upload. The next step
validates the registry and all four exported profiles against the frozen
inventory before a foundation is created. For the first transfer, ensure the
destination does not already contain a differently sourced bundle of this name.
Do not overwrite a bundle referenced by existing jobs.

## 3. SPORC: clean source, authenticate metadata, create exact dry run

Run this inside a subshell so a failed check does not end the SSH login.
Replace `PASTE_THE_COMMIT_FROM_STEP_1` with the exact 40-character pushed hash.
Initial profile resources are 8 CPUs, 8 workers, 80 GiB, one A100, four hours.
Assignment elements are CPU-only, at most 16 concurrent, two hours each.
These are starting envelopes to measure, not claims that real acceptance passed.

```bash
(
set -euo pipefail
export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
export JC2_COMMIT=PASTE_THE_COMMIT_FROM_STEP_1
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_jc2_sporc_${JC2_COMMIT:0:8}"
export JC2_DATA_PARENT=/home/ryreu/atlas/datasets/jetclass2_10M_20260910
export JC2_BUNDLE="${JC2_DATA_PARENT}/jetclass2_delphes_scaling_splits_20260911_v1"
export JC2_READY="${MAIN_REPO}/checkpoints/jc2_sporc_ready_500k_${JC2_COMMIT:0:8}_r1"

git -C "${MAIN_REPO}" fetch origin main
test "$(git -C "${MAIN_REPO}" rev-parse origin/main)" = "${JC2_COMMIT}"
test ! -e "${JC2_READY}"
if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${JC2_COMMIT}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach "${PROJECT_DIR}" "${JC2_COMMIT}"
fi

export JC2_SITE=sporc_a100
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT_DIR}/scripts/create_jetclass2_delphes_split_registry.py" inspect \
  --inventory "${JC2_DATA_PARENT}/manifests/inventory.json" \
  --registry "${JC2_BUNDLE}/registry.json"

python -s "${PROJECT_DIR}/scripts/prepare_jetclass2_delphes_sporc.py" create \
  --inventory "${JC2_DATA_PARENT}/manifests/inventory.json" \
  --split-profile "${JC2_BUNDLE}/profiles/TRAIN_500K.json" \
  --data-root "${JC2_DATA_PARENT}/jetclass2" \
  --output-root "${JC2_READY}" --source-commit "${JC2_COMMIT}"

python -m json.tool "${JC2_READY}/command_plan.json"
echo "READINESS ROOT: ${JC2_READY}"
echo 'Dry run only. No jobs submitted.'
)
```

Check for four commands: `sample`, `assign`, `lock`, `profile`. Only profile
requests `gpu:a100:1`; every command uses `reu-aisocial/tier3/qos_tier3`.
No train_* science tasks, final-test tasks, external job dependencies, `scancel`
or `scontrol hold/update` operations occur. The role counts must be
500000/1000000/1000000. If anything differs, stop and inspect the artifact.

## 4. SPORC: separately authorize just this gate

Set the two paths printed/used above (subshell variables intentionally do not
persist into the login session). Submission rechecks raw-file checksums; that
is read-only byte verification, not final-test decoding/model access.

```bash
(
set -euo pipefail
export PROJECT_DIR=/home/ryreu/atlas/HLT_Classification_jc2_sporc_COMMIT8
export JC2_READY=/home/ryreu/atlas/HLT_Classification/checkpoints/jc2_sporc_ready_500k_COMMIT8_r1
export JC2_SITE=sporc_a100
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"
python -s "${PROJECT_DIR}/scripts/prepare_jetclass2_delphes_sporc.py" submit \
  --spec "${JC2_READY}/readiness_spec.json" --execute \
  --authorization-phrase 'AUTHORIZE JETCLASS2 DELPHES SPORC READINESS ONLY'
squeue --me -o '%.18i %.44j %.2t %.10M %R' | grep -E 'JOBID|jc2gate_' || true
)
```

The exact dependency chain is sample -> complete assignment array -> foundation
lock -> real A100 profile. [Slurm afterok array semantics](https://slurm.schedmd.com/job_array.html)
require every array element to succeed before the lock starts. Specifying
resources does not mean they were measured: only the completed real profile
provides that evidence. If a job fails/timeouts, keep its logs/outputs and inspect
that exact ID; do not blindly rerun preparation into an existing immutable root.

## What to return after the gate

Share the exact job IDs/states and the contents of:

- `foundation/sample_audit.json` and `foundation/foundation_lock.json`;
- `evidence/miniature/acceptance.json`;
- `evidence/resource_measurements.json` and `evidence/runtime_profile.json`.

The profile runs real installed-Weaver parity, CE/oracle/KD miniature checks,
one full-profile pass, reducer inference, capacity/batch memory checks and
U000/U050/D050 preprocessing measurements. Scientific results never control
acceptance. The production campaign still needs its separate creation, full
59-task dry run and explicit authorization after we review this evidence.
Nothing in this procedure launches it automatically.

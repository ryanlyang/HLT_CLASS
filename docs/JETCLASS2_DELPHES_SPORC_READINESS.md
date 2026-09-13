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

## Completed preparation: profile only on debug

Use this alternative only after the source readiness foundation lock exists.
It re-authenticates every existing assignment; it does not repeat sample,
matching or lock jobs. The new profile is measured on debug but explicitly
targets tier3 for subsequent scientific jobs on the same GPU/environment and
CPU/RAM settings. Both original site definitions and readiness artifacts remain
unchanged. The source checkout must include the debug-profile implementation.

On Windows, the existing scoped staging helper also includes the new Delphes
profile files and preserves the unrelated salience/HANDOFF edits:

```powershell
& .\scripts\stage_jetclass2_delphes_sporc.ps1
git diff --cached --check
git diff --cached --stat
git commit -m "Allow debug-only JetClass2 profiling with completed foundation reuse"
git push origin HEAD:main
git rev-parse HEAD
```

Run the following on SPORC after pushing. The original readiness path is the
user's existing TRAIN_500K execution. A fresh debug attempt requests one A100,
8 CPUs/workers, 72 GiB and four hours. `--profile-minutes 120` is an optional,
more aggressive unmeasured cap, not required by the debug partition.

```bash
(
set -euo pipefail
MAIN_REPO=/home/ryreu/atlas/HLT_Classification
git -C "${MAIN_REPO}" fetch origin main
JC2_FIX="$(git -C "${MAIN_REPO}" rev-parse origin/main)"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_jc2_debug_${JC2_FIX:0:8}"
READY_ROOT="${MAIN_REPO}/checkpoints/jc2_sporc_ready_500k_275b984d_r1"
DEBUG_ROOT="${MAIN_REPO}/checkpoints/jc2_debug_profile_${JC2_FIX:0:8}_r1"

git -C "${MAIN_REPO}" cat-file -e "${JC2_FIX}:scripts/prepare_jetclass2_delphes_debug_profile.py"
if [ -e "${PROJECT_DIR}" ]; then
  test "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" = "${JC2_FIX}"
  test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
else
  git -C "${MAIN_REPO}" worktree add --detach "${PROJECT_DIR}" "${JC2_FIX}"
fi
export JC2_SITE=sporc_a100_debug
source "${PROJECT_DIR}/sbatch/jetclass2_delphes_common.sh"

python -s "${PROJECT_DIR}/scripts/prepare_jetclass2_delphes_debug_profile.py" create \
  --readiness-spec "${READY_ROOT}/readiness_spec.json" \
  --output-root "${DEBUG_ROOT}" --source-commit "${JC2_FIX}"

# Creation has validated reuse and printed the single-job dry command.
# Resolve only the original profile ID from its authenticated live ledger.
OLD_PROFILE="$(python -s - "${READY_ROOT}" <<'PY'
import sys
from pathlib import Path
from hlt_classification.data.cache_contracts import load_json
from hlt_classification.scouting.hcwdl_recovery import validate_submission_ledger
root = Path(sys.argv[1])
spec = load_json(root / "readiness_spec.json")
ledger = load_json(root / "submission_ledger.json")
validate_submission_ledger(ledger)
assert ledger["campaign_spec_sha256"] == spec["content_hash"] and not ledger["dry_run"]
print(ledger["jobs"]["profile"])
PY
)"
OLD_STATE="$(squeue --me -h -o '%i %T' | awk -v job="${OLD_PROFILE}" '$1 == job {print $2}')"
if [ "${OLD_STATE}" = "PENDING" ]; then
  scancel --ctld --state=PENDING --name=jc2gate_profile "${OLD_PROFILE}"
else
  echo "Old profile ${OLD_PROFILE} is no longer pending. Inspect it before replacing it."
  exit 1
fi
OLD_STATE="$(squeue --me -h -o '%i %T' | awk -v job="${OLD_PROFILE}" '$1 == job {print $2}')"
if [ -n "${OLD_STATE}" ]; then
  echo "Old profile is still ${OLD_STATE}; stop and inspect before submitting a replacement."
  exit 1
fi

python -s "${PROJECT_DIR}/scripts/prepare_jetclass2_delphes_debug_profile.py" submit \
  --spec "${DEBUG_ROOT}/profile_attempt_spec.json" --execute \
  --authorization-phrase 'AUTHORIZE JETCLASS2 DELPHES DEBUG PROFILE ONLY'
echo "Debug evidence: ${DEBUG_ROOT}/evidence"
squeue --me -o '%.18i %.14P %.32j %.2t %.10M %R'
)
```

The optional replacement block above cancels only the old pending profile,
not its completed predecessors or any other project. There is no cancellation
in either Python submitter. The [Slurm state filter](https://slurm.schedmd.com/scancel.html)
also protects a job that starts between the check and cancellation request.
If the old job is already running/completed, inspect
it; there may be no reason to replace it. If an error occurs after creation,
keep the new root and resume with **submit only**, not create again. Never
re-submit an ambiguous sbatch acknowledgement without exact-job reconciliation.

After success return `profile_result.json`, `evidence/resource_measurements.json`
and `evidence/runtime_profile.json` from the debug root. Science creation uses
the unchanged `${READY_ROOT}/foundation`, the new profile and the SAME new
commit/worktree; all 59 scientific commands must still say `--partition=tier3`.
No source hot-patching or full-readiness resubmission is needed. Timing remains
an estimate; actual A100 acceptance is established only when this job passes.

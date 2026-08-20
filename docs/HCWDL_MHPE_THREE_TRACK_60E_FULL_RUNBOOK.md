# HCWDL-MHPE three-track 60-pass full-data runbook

This runbook prepares the exact all-mapped 32-fit campaign. It does not submit
the science DAG unless the final phrase-bound command is run. It never accesses
final test. Replace only the explicitly marked paths.

## 1. Clean source and immutable parents

On Tigris:

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

export MAIN_REPO=/home/ryreu/atlas/HLT_Classification
cd "${MAIN_REPO}"
git fetch origin main
export TRI_COMMIT="$(git rev-parse origin/main)"
export TRI_SHORT="${TRI_COMMIT:0:8}"
export PROJECT_DIR="/home/ryreu/atlas/HLT_Classification_mhpe_tri60_${TRI_SHORT}"
export CAMPAIGN_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_full_${TRI_SHORT}_r1"
export EVIDENCE_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_evidence_${TRI_SHORT}_r1"

test ! -e "${PROJECT_DIR}"
test ! -e "${CAMPAIGN_ROOT}"
mkdir -p "${EVIDENCE_ROOT}"
git -C "${MAIN_REPO}" worktree add --detach "${PROJECT_DIR}" "${TRI_COMMIT}"
test -z "$(git -C "${PROJECT_DIR}" status --porcelain)"
export PYTHONPATH="${PROJECT_DIR}/src"
```

Bind the completed all-mapped unified-balanced foundation and the exact
authorized v5 representation recipe:

```bash
export FOUNDATION_LOCK=/absolute/all-mapped/foundation/locks/foundation.json
export REPRESENTATION_RECIPE=/absolute/authenticated/representation_recipe_v5.json
test -f "${FOUNDATION_LOCK}"
test -f "${REPRESENTATION_RECIPE}"
```

Do not select either file by newest timestamp. Validate and record the exact
paths from their completed campaign lineage first.

## 2. Source tests and installed-Weaver parity

The source-evidence command runs the focused integration list, complete suite,
CLI-help checks, compilation, and diff check in the clean detached worktree:

```bash
python -s "${PROJECT_DIR}/scripts/validate_hcwdl_mhpe_tri60_source.py" \
  --source-commit "${TRI_COMMIT}" \
  --output "${EVIDENCE_ROOT}/source_tests.json"

python -s "${PROJECT_DIR}/scripts/validate_hcwdl_homotopy_weaver.py" \
  --source-commit "${TRI_COMMIT}" \
  --device cuda \
  --output "${EVIDENCE_ROOT}/installed_weaver_parity.json"
```

## 3. Bounded production-worker acceptance

This is not a 300k smoke or a scientific ladder. It uses bounded real rows to
exercise the production view, one-forward RSET target, no-resume training, and
RAM projection. All temporary tensors/checkpoints live under `SLURM_TMPDIR`
and are deleted before the small profile is published.

First inspect the command. Submit it only with the user's explicit approval:

```bash
sbatch --parsable \
  --account=reu-aisocial \
  --partition=tigris \
  --cpus-per-task=16 \
  --mem=384G \
  --time=02:00:00 \
  --gres=gpu:gh200:1 \
  --job-name=hcwtri60_acceptance \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",HCWDL_TRI60_FOUNDATION_LOCK="${FOUNDATION_LOCK}",HCWDL_TRI60_SOURCE_COMMIT="${TRI_COMMIT}",HCWDL_TRI60_PROFILE_OUTPUT="${EVIDENCE_ROOT}/runtime_profile.json" \
  "${PROJECT_DIR}/sbatch/run_hcwdl_mhpe_tri60_acceptance.sh"
```

Wait for that one job to complete, then require:

```bash
python -s - "${EVIDENCE_ROOT}/runtime_profile.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p["passed"] is True
assert p["genuine_tigris_production_worker"] is True
assert p["ram_only_targets_proved"] is True
assert p["no_resume_proved"] is True
assert p["peak_request_fraction"] <= 0.75
assert p["temporary_artifact_bytes_after_cleanup"] == 0
assert p["final_test_accessed"] is False
print("TRI60 bounded production acceptance passed")
PY
```

## 4. Create the immutable campaign and dry-run ledger

```bash
python -s "${PROJECT_DIR}/scripts/create_hcwdl_mhpe_tri60_campaign.py" \
  --foundation-lock "${FOUNDATION_LOCK}" \
  --representation-recipe "${REPRESENTATION_RECIPE}" \
  --test-evidence "${EVIDENCE_ROOT}/source_tests.json" \
  --installed-weaver-parity "${EVIDENCE_ROOT}/installed_weaver_parity.json" \
  --runtime-profile "${EVIDENCE_ROOT}/runtime_profile.json" \
  --campaign-root "${CAMPAIGN_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${TRI_COMMIT}" \
  --authorize-live-submission \
  --authorization-phrase \
    "AUTHORIZE HCWDL MHPE THREE TRACK 60E FULL EXACT SPEC"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_mhpe_tri60_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/dry_run_submission_ledger.json"

python -s - "${CAMPAIGN_ROOT}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
spec = json.load(open(root / "campaign_spec.json"))
plan = json.load(open(root / "command_plan.json"))
dry = json.load(open(root / "dry_run_submission_ledger.json"))
assert len(spec["tasks"]) == len(plan["commands"]) == len(dry["jobs"]) == 50
assert sum(x["kind"] == "train" for x in spec["tasks"]) == 32
assert sum(x["kind"] == "reducer" for x in spec["tasks"]) == 13
assert spec["representation_targets_persisted"] is False
assert spec["rolling_resume"] is False
assert spec["ordinary_final_test_capability"] is False
assert dry["dry_run"] is True
print("TRI60 all-data dry run passed: 50 jobs, 32 fits, 13 publications")
PY
```

Review the complete immutable spec and commands. The preflight job also
requires at least 16 GiB free after the small projected durable campaign
footprint; representation targets and view caches never enter that footprint.

## 5. Live science submission

Only after the preceding checks and a separate explicit user authorization:

```bash
python -s "${PROJECT_DIR}/scripts/submit_hcwdl_mhpe_tri60_campaign.py" \
  --spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --output "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase \
    "SUBMIT HCWDL MHPE THREE TRACK 60E FULL EXACT LEDGER"
```

The submitter safely resumes a partially published submission journal and
never resubmits a journaled job.

## 6. Exact monitoring, cancellation, and recovery

```bash
python -s "${PROJECT_DIR}/scripts/monitor_hcwdl_mhpe_tri60.py" \
  --subject-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --ledger "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --attestation-root "${CAMPAIGN_ROOT}" \
  --output "${CAMPAIGN_ROOT}/monitor.json"
```

Cancellation requires that exact monitor and prints its exact IDs unless both
`--execute` and the phrase are supplied:

```bash
python -s "${PROJECT_DIR}/scripts/cancel_hcwdl_mhpe_tri60.py" \
  --ledger "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --monitor "${CAMPAIGN_ROOT}/monitor.json" \
  --output "${CAMPAIGN_ROOT}/cancellation_preview.json"
```

Create recovery only after failed/downstream jobs are terminal or have been
cancelled by their exact IDs and a new monitor has been published:

```bash
export RECOVERY_ROOT="${MAIN_REPO}/checkpoints/hcwdl_mhpe_tri60_recovery_${TRI_SHORT}_r1"

python -s "${PROJECT_DIR}/scripts/create_hcwdl_mhpe_tri60_recovery.py" \
  --subject-spec "${CAMPAIGN_ROOT}/campaign_spec.json" \
  --subject-ledger "${CAMPAIGN_ROOT}/submission_ledger.json" \
  --monitor-report "${CAMPAIGN_ROOT}/monitor.json" \
  --recovery-root "${RECOVERY_ROOT}" \
  --project-dir "${PROJECT_DIR}" \
  --source-commit "${TRI_COMMIT}"

python -s "${PROJECT_DIR}/scripts/submit_hcwdl_mhpe_tri60_recovery.py" \
  --spec "${RECOVERY_ROOT}/recovery_spec.json" \
  --output "${RECOVERY_ROOT}/dry_run_submission_ledger.json"
```

Live recovery is a separate phrase-bound action. Every failed fit is cleaned
only inside its registered campaign subdirectory and restarts at update zero;
completed siblings and compact probability banks remain immutable.

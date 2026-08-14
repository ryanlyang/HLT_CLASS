# HCWDL structural-feature homotopy runbook

This runbook is the operator procedure for the validation-only structural and
feature homotopy defined by the
[implementation plan](plans/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY_IMPLEMENTATION_PLAN.md)
and [reusable contract](contracts/HCWDL_STRUCTURAL_FEATURE_HOMOTOPY.md). It
does not authorize final-test access. It also does not reinterpret or mutate
the parent HCWDL campaign, dense ladders, assignments, or checkpoints.

The completed 80-fit v1 smoke is immutable operational evidence. The reduced
v2 graph may be authorized either by a completed measured v2 smoke or by the
explicit operational-evidence carry-forward described below after v2 parity,
coupling, endpoint, target, and graph locks complete. The waiver is a human
launch decision, not a fabricated v2 completion.

## 1. Activate Tigris and pin a clean source worktree

Run from the canonical checkout after the implementation commit has been
pushed. This creates a detached worker checkout so later pulls in the main
checkout cannot change a running campaign.

```bash
set -euo pipefail

source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd /home/ryreu/atlas/HLT_Classification
git fetch origin main

export UJ_COMMIT="$(git rev-parse origin/main)"
export UJ_SHORT="${UJ_COMMIT:0:8}"
export UJ_WORKTREE="/home/ryreu/atlas/HLT_Classification_hcwdl_uj_${UJ_SHORT}"

test "$(printf '%s' "${UJ_COMMIT}" | wc -c)" -eq 40
test ! -e "${UJ_WORKTREE}"
git worktree add --detach "${UJ_WORKTREE}" "${UJ_COMMIT}"
test -z "$(git -C "${UJ_WORKTREE}" status --porcelain)"
test "$(git -C "${UJ_WORKTREE}" rev-parse HEAD)" = "${UJ_COMMIT}"
```

Find the newest completed, executable, unweighted HCWDL v4 smoke that has the
required M0, D100, TOFF, assignments, and locks. The authentication function
below is the same one used by campaign creation; path existence is not treated
as evidence.

```bash
export PYTHONPATH="${UJ_WORKTREE}/src"
export PARENT_SMOKE_SPEC="$(python -s - <<'PY'
from pathlib import Path
from hlt_classification.scouting.hcwdl_homotopy_campaign import authenticate_parent

candidates = sorted(
    Path('/home/ryreu/atlas/HLT_Classification/checkpoints').glob(
        'hcwdl_smoke_*/campaign_spec.json'
    ),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
for candidate in candidates:
    try:
        evidence = authenticate_parent(candidate, dense_d0_report=None)
    except Exception:
        continue
    if evidence['mode'] == 'smoke':
        print(candidate.resolve())
        break
else:
    raise SystemExit(
        'No authenticated unweighted HCWDL v4 smoke parent exists; run the '
        'primary HCWDL smoke first.'
    )
PY
)"
test -f "${PARENT_SMOKE_SPEC}"
echo "Parent smoke: ${PARENT_SMOKE_SPEC}"
```

If this fails, stop. A weighted smoke, a pilot, or a merely present directory
is not an admissible substitute.

## 2. Validate both installed-Weaver model factories

Run the direct-versus-wrapper FP32 acceptance on a real Tigris GPU from the
same clean pinned checkout. This covers both the unified 21-channel student
and the two-stream 19/7-channel native TOFF teacher. The local development
environment is not a substitute for the installed `atlas_kd_tigris` Weaver.

```bash
export UJ_WEAVER_PARITY="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_uj_weaver_parity_${UJ_SHORT}.json"
test ! -e "${UJ_WEAVER_PARITY}"

srun --account=reu-aisocial --partition=tigris \
  --gres=gpu:gh200:1 --cpus-per-task=8 --mem=64G --time=00:20:00 \
  python -s "${UJ_WORKTREE}/scripts/validate_hcwdl_homotopy_weaver.py" \
    --source-commit "${UJ_COMMIT}" \
    --device cuda \
    --output "${UJ_WEAVER_PARITY}"

test -f "${UJ_WEAVER_PARITY}"
python -m json.tool "${UJ_WEAVER_PARITY}"
```

If either parity path fails, do not create or submit the smoke.

## 3. Optional bounded local behavior check

This exercises all 45 registered v2 fits with synthetic tensors and a two-update
schedule. It is useful after an environment change but does not replace the
real Tigris smoke.

```bash
export LOCAL_SMOKE_ROOT="${TMPDIR:-/tmp}/hcwdl_uj_local_${UJ_SHORT}_$$"
python -s "${UJ_WORKTREE}/scripts/run_hcwdl_homotopy_local_smoke.py" \
  --output-root "${LOCAL_SMOKE_ROOT}"
```

## 4. Build and inspect a non-live smoke specification

Use a disposable planning root. Campaign roots are immutable, so the later
authorized campaign receives a different root.

```bash
export UJ_PLAN_ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_uj_v2_smoke_${UJ_SHORT}_plan_r1"
test ! -e "${UJ_PLAN_ROOT}"

python -s "${UJ_WORKTREE}/scripts/create_hcwdl_homotopy_pilot.py" \
  --parent-campaign-spec "${PARENT_SMOKE_SPEC}" \
  --campaign-root "${UJ_PLAN_ROOT}" \
  --project-dir "${UJ_WORKTREE}" \
  --source-commit "${UJ_COMMIT}" \
  --weaver-parity "${UJ_WEAVER_PARITY}"

python -s "${UJ_WORKTREE}/scripts/submit_hcwdl_homotopy_pilot.py" \
  --campaign-spec "${UJ_PLAN_ROOT}/campaign_spec.json" \
  --output "${UJ_PLAN_ROOT}/dry_run_submission_ledger.json"

python -s - "${UJ_PLAN_ROOT}" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
spec = json.loads((root / 'campaign_spec.json').read_text())
ledger = json.loads((root / 'dry_run_submission_ledger.json').read_text())
assert spec['mode'] == 'smoke'
assert spec['role_counts'] == {'train': 4096, 'validation': 4096, 'final_test': 0}
assert spec['final_test_accessed'] is False
assert len(spec['tasks']) == 66
assert len(ledger['jobs']) == 66 and ledger['dry_run'] is True
assert sum(row['kind'] == 'train_node' for row in spec['tasks']) == 45
print('Dry-run smoke:', spec['content_hash'])
print('Tasks:', len(spec['tasks']), 'fits:', 45)
PY
```

Inspect `command_plan.json` and the dry-run ledger. The factorized and joint
roots may run concurrently, each track is sequential, all controls are
registered, and no final-test task may appear.

## 5. Authorized real Tigris smoke

Do not run this block until the user has explicitly authorized the real smoke.
It creates the exact live specification and submits all 66 tasks once. The
submitter records an immutable journal and a partial exact-ID ledger even if
submission is interrupted.

```bash
export UJ_SMOKE_ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_uj_v2_smoke_${UJ_SHORT}_r1"
test -f "${UJ_WEAVER_PARITY}"
test ! -e "${UJ_SMOKE_ROOT}"

python -s "${UJ_WORKTREE}/scripts/create_hcwdl_homotopy_pilot.py" \
  --parent-campaign-spec "${PARENT_SMOKE_SPEC}" \
  --campaign-root "${UJ_SMOKE_ROOT}" \
  --project-dir "${UJ_WORKTREE}" \
  --source-commit "${UJ_COMMIT}" \
  --weaver-parity "${UJ_WEAVER_PARITY}" \
  --authorize-live-submission \
  --authorization-phrase "AUTHORIZE HCWDL UJ VALIDATION CAMPAIGN EXACT SPEC"

python -s "${UJ_WORKTREE}/scripts/submit_hcwdl_homotopy_pilot.py" \
  --campaign-spec "${UJ_SMOKE_ROOT}/campaign_spec.json" \
  --output "${UJ_SMOKE_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "SUBMIT HCWDL UJ VALIDATION CAMPAIGN EXACT SPEC"
```

This is the exact remaining smoke submission command. It is documented here
but was not executed during local implementation.

## 6. Monitor and inspect the smoke

```bash
python -s "${UJ_WORKTREE}/scripts/monitor_hcwdl_homotopy.py" \
  --campaign-spec "${UJ_SMOKE_ROOT}/campaign_spec.json" \
  --submission-ledger "${UJ_SMOKE_ROOT}/submission_ledger.json" \
  --query-slurm \
  --output "${UJ_SMOKE_ROOT}/monitor/latest.json"

python -m json.tool "${UJ_SMOKE_ROOT}/monitor/latest.json"
```

Poor but finite metrics are science and do not block descendants. Failed jobs,
nonfinite required values, stale source, invalid inputs, corrupt artifacts, or
lineage mismatches do. Completion is
`reports/campaign_complete.json`; the full validation report is
`reports/validation_aggregate.json`.

Corrected workers emit explicit `student_view_cache`, `teacher_targets`, and
`optimizer_training` phase lines. Homotopy view construction defaults to the
complete `SLURM_CPUS_PER_TASK` allocation, keeps at most one chunk per worker
in flight, and consumes completed chunks in canonical input order. An
operator may set `HCWDL_UJ_VIEW_BUILD_WORKERS` to a smaller positive value for
diagnosis; a value larger than the allocated CPU count fails closed. Absence
of a first validation checkpoint is therefore distinguishable from training
by inspecting the corresponding Slurm output. The matcher is never callable
from a training worker.

Exact cancellation first prints, then optionally executes, only the IDs bound
to this campaign ledger:

```bash
python -s "${UJ_WORKTREE}/scripts/cancel_hcwdl_homotopy.py" \
  --submission-ledger "${UJ_SMOKE_ROOT}/submission_ledger.json"

# Only after inspecting the exact IDs:
python -s "${UJ_WORKTREE}/scripts/cancel_hcwdl_homotopy.py" \
  --submission-ledger "${UJ_SMOKE_ROOT}/submission_ledger.json" \
  --execute \
  --authorization-phrase "CANCEL HCWDL UJ EXACT IDS"
```

## 7. Capture measured resources

Only a fully completed, artifact-valid v2 live smoke is admissible. Create a
JSON object with exactly the six resource classes printed in the smoke spec.
The 300k training request is fixed at 8 CPUs, 96 GiB, six hours, and one GH200;
all rows must still retain at least 25% measured smoke headroom or profile
publication fails closed.

```bash
cat > "${UJ_SMOKE_ROOT}/reviewed_pilot_requests.json" <<'JSON'
{
  "requests": {
    "cpu_calibration": {"cpus": 8, "memory": "64G", "walltime": "02:00:00", "gpu": null},
    "cpu_coupling": {"cpus": 8, "memory": "96G", "walltime": "04:00:00", "gpu": null},
    "cpu_finalize": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": null},
    "gpu_targets": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
    "gpu_training": {"cpus": 8, "memory": "96G", "walltime": "06:00:00", "gpu": "gpu:gh200:1"},
    "cpu_report": {"cpus": 4, "memory": "32G", "walltime": "01:00:00", "gpu": null}
  }
}
JSON
```

```bash
python -s "${UJ_WORKTREE}/scripts/build_hcwdl_homotopy_resource_profile.py" \
  --smoke-campaign-spec "${UJ_SMOKE_ROOT}/campaign_spec.json" \
  --submission-ledger "${UJ_SMOKE_ROOT}/submission_ledger.json" \
  --monitor-report "${UJ_SMOKE_ROOT}/monitor/latest.json" \
  --resume-evidence "${UJ_SMOKE_ROOT}/runtime/resume_evidence.json" \
  --requests-json "${UJ_SMOKE_ROOT}/reviewed_pilot_requests.json" \
  --storage-budget-gib 20 \
  --query-slurm \
  --measurement-output "${UJ_SMOKE_ROOT}/runtime/smoke_resource_measurement.json" \
  --output "${UJ_SMOKE_ROOT}/runtime/pilot_resource_profile.json"
```

The profiler authenticates exact job/array coverage, Slurm RAM/wall/disk I/O,
worker-recorded GPU peaks, all durable artifact bytes, and campaign completion.

## 8. Create the 300k pilot dry run

When the human authorizes v1 operational-evidence carry-forward, first publish
the immutable waiver with
`build_hcwdl_homotopy_operational_waiver.py`, then pass it as
`--operational-evidence-waiver` instead of `--resource-profile`. Never pass
both. The partial v2 smoke remains partial in its own ledger and is not used
as a scientific result.

The parent must be the exact completed unweighted 300k HCWDL pilot. The D0c
report and both dense direct controls must share that parent lineage. Paths
below are explicit operator inputs; campaign creation reauthenticates them.

```bash
export PARENT_300K_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_pilot_b3154d67_unweighted_auto_r1
export DENSE10_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_densecold300k_4be8576d_fixed_r1
export DENSE5_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_dense5cold300k_4be8576d_r1
export UJ_PILOT_PLAN_ROOT="/home/ryreu/atlas/HLT_Classification/checkpoints/hcwdl_uj_v2_pilot_${UJ_SHORT}_plan_r1"

test ! -e "${UJ_PILOT_PLAN_ROOT}"
python -s "${UJ_WORKTREE}/scripts/create_hcwdl_homotopy_pilot.py" \
  --parent-campaign-spec "${PARENT_300K_ROOT}/campaign_spec.json" \
  --dense-d0-report "${PARENT_300K_ROOT}/training/D0c/training_report.json" \
  --contextual-report "${DENSE10_ROOT}/training/D100offkd/training_report.json" \
  --contextual-report "${DENSE5_ROOT}/training/D100offkd/training_report.json" \
  --resource-profile "${UJ_SMOKE_ROOT}/runtime/pilot_resource_profile.json" \
  --campaign-root "${UJ_PILOT_PLAN_ROOT}" \
  --project-dir "${UJ_WORKTREE}" \
  --source-commit "${UJ_COMMIT}" \
  --weaver-parity "${UJ_WEAVER_PARITY}"

python -s "${UJ_WORKTREE}/scripts/submit_hcwdl_homotopy_pilot.py" \
  --campaign-spec "${UJ_PILOT_PLAN_ROOT}/campaign_spec.json" \
  --output "${UJ_PILOT_PLAN_ROOT}/dry_run_submission_ledger.json"
```

Do not create or submit the authorized 300k root until the real smoke,
resource review, exact pushed source, and complete 300k dry run have been
reviewed and the user separately authorizes the pilot.

## 9. Failed-closure recovery

First publish an immutable monitor. A source-only recovery preserves the exact
graph, recipe, coupling, seeds, resources, completed artifacts, and compatible
rolling checkpoints while changing only execution source. A resource-only
recovery permits reviewed increases for the authenticated failed closure and
does not change scientific identity. Both can be repeated and submit only the
exact failed/downstream closure.

Render source recovery without scheduler mutation:

```bash
python -s "${UJ_WORKTREE}/scripts/resume_hcwdl_homotopy.py" \
  --campaign-spec "${UJ_SMOKE_ROOT}/campaign_spec.json" \
  --submission-ledger "${UJ_SMOKE_ROOT}/submission_ledger.json" \
  --monitor-report "${UJ_SMOKE_ROOT}/monitor/latest.json" \
  --recovery-root "${UJ_SMOKE_ROOT}/recovery/source_${UJ_SHORT}_r1" \
  --project-dir "${UJ_WORKTREE}" \
  --source-commit "${UJ_COMMIT}"
```

Live source recovery additionally requires
`AUTHORIZE HCWDL UJ FAILED CLOSURE RECOVERY` and
`SUBMIT HCWDL UJ FAILED CLOSURE RECOVERY`. Resource recovery supplies
`--replacement-resources`, uses its distinct resource-only phrases, and is
accepted only when at least one failed task's requested resource increases.

When the same failed closure requires both an authenticated execution-only
source correction and a monotonic resource increase, use the separately
versioned combined path. The replacement JSON must contain the complete
effective resource-class map reconstructed from the parent ledger; classes
not being increased remain byte-for-byte equal. For the 2026-08-14 linear
endpoint-preparation recovery, only `gpu_training.cpus` changes from 8 to 16;
memory remains 96G, walltime remains six hours, and the GPU remains one GH200.

```bash
python -s "${UJ_WORKTREE}/scripts/resume_hcwdl_homotopy.py" \
  --campaign-spec "${UJ_PILOT_ROOT}/campaign_spec.json" \
  --submission-ledger "${UJ_PARENT_RECOVERY_LEDGER}" \
  --monitor-report "${UJ_POST_CANCEL_MONITOR}" \
  --recovery-root "${UJ_EXECUTION_RESOURCE_ROOT}" \
  --project-dir "${UJ_WORKTREE}" \
  --source-commit "${UJ_COMMIT}" \
  --replacement-resources "${UJ_REPLACEMENT_RESOURCES}" \
  --execution-and-resource-recovery \
  --authorization-phrase \
    "AUTHORIZE HCWDL UJ EXECUTION AND RESOURCE RECOVERY" \
  --execute \
  --submission-phrase \
    "SUBMIT HCWDL UJ EXECUTION AND RESOURCE RECOVERY"
```

This path accepts an exact-ID `CANCELLED` closure after the operator has
cancelled the superseded ledger. It preserves completed reports and compatible
rolling checkpoints, and every generated training command must contain
`--cpus-per-task=16` before live submission.

## 10. Claims boundary

The residual O/R coupling is a deterministic homotopy device, not inferred
truth correspondence. P0/U000 is the TOFF-visible particle multiset under the
unified projection, not the canonical TOFF adapter. U100 is exact D100;
D0F/J100 and the registered M1 models are HLT-only. Only M1F and M1J are
designated screen outputs. This validation-only study cannot read final test,
and a successful path does not imply that missing offline particles are
reconstructed at inference.

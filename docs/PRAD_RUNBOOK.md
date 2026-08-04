# PRAD Research-Compute Runbook

This runbook executes the active PRAD plan without weakening the repository's
source, data-lineage, resource, or final-test locks. The full campaign must not
be submitted until the smoke graph is complete and production authorization is
explicit.

## Fixed Tigris site

```text
account       reu-aisocial
partition     tigris
project       /home/ryreu/atlas/HLT_Classification
environment   atlas_kd_tigris
JetClass      /home/ryreu/atlas/PracticeTagging/data (read-only)
```

`sbatch/run_prad_task.sh` sets `PYTHONNOUSERSITE=1`, activates the registered
environment through the absolute project helper, and prepends
`${CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`. Workers reject a dirty or different
source snapshot before doing work.

## 1. Local acceptance

From a clean checkout of the exact commit intended for Tigris:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='src'
python -m pytest -q -p no:cacheprovider
git diff --check
```

Commit and push before creating a campaign specification. Untracked source or
documentation intentionally makes source capture fail.

## 2. Create and inspect the real miniature

On Tigris, after checking out the exact pushed commit:

```bash
cd /home/ryreu/atlas/HLT_Classification
export PYTHONNOUSERSITE=1
SHORT_SHA="$(git rev-parse --short=8 HEAD)"
SMOKE_ROOT="/home/ryreu/atlas/HLT_Classification/artifacts/prad/smoke_min_${SHORT_SHA}"
test ! -e "${SMOKE_ROOT}" || { echo "ERROR: smoke root already exists" >&2; exit 1; }
python -s scripts/create_prad_campaign.py \
  --mode smoke \
  --campaign-root "${SMOKE_ROOT}" \
  --output "${SMOKE_ROOT}/candidate_spec.json"
python -s scripts/submit_prad_campaign.py \
  --campaign-spec "${SMOKE_ROOT}/candidate_spec.json" \
  --dry-run \
  --dry-run-report "${SMOKE_ROOT}/dry_run.json"
```

Inspect every rendered command, requested resource, array, and dependency.
The smoke DAG runs both canonical installed-Weaver parity and a CUDA PRAD
mechanics attestation before E0. The latter must authenticate zero-gate
baseline parity, HLT-only public inference, gradient flow, and frozen-teacher
behavior in `reports/prad_runtime_validation.json`. Its v3 contract also
reproduces the pair-supervised Stage-A and mixed-freeze Stage-B backward
surfaces. It proves that the relation-only loss excludes the frozen final-logit
path and that CUDA Stage B successfully uses its registered math-SDPA policy.
The 15-node minimum-storage DAG has no durable paired-view, target, or teacher
cache nodes. Each consuming worker rebuilds the required arrays under its
Slurm job-local temporary directory, reuses them within that task, and removes
them on exit. The finalist and execution locks bind deterministic ephemeral
input manifests, so recomputation does not weaken lineage or test sealing.
Then submit only the miniature:

```bash
bash sbatch/submit_prad.sh \
  --campaign-spec "${SMOKE_ROOT}/candidate_spec.json" \
  --smoke-submit
```

The submitter durably records each numeric `sbatch --parsable` ID before
submitting the next node. Re-running after a submit-side interruption reuses
the exact recorded prefix instead of duplicating jobs.

## 3. Monitor, recover, or cancel by exact IDs

```bash
python -s scripts/monitor_prad_campaign.py \
  --campaign-spec "${SMOKE_ROOT}/campaign_spec.json" \
  --submission-ledger "${SMOKE_ROOT}/ledgers/submission.json" \
  --output "${SMOKE_ROOT}/monitor.json"
```

If a node fails, first inspect its exact `.out` and `.err` files. A recovery
graph reuses only completed jobs whose expected task attestations authenticate:

```bash
python -s scripts/resume_prad_campaign.py \
  --campaign-spec "${SMOKE_ROOT}/campaign_spec.json" \
  --prior-ledger "${SMOKE_ROOT}/ledgers/submission.json" \
  --monitor-report "${SMOKE_ROOT}/monitor.json" \
  --attempt 1 \
  --dry-run
```

Replace `--dry-run` with `--execute` only after inspecting the exact reuse and
resubmission lists. Cancellation likewise accepts task names but resolves them
only through the campaign ledger:

```bash
python -s scripts/cancel_prad_campaign.py \
  --campaign-spec "${SMOKE_ROOT}/campaign_spec.json" \
  --submission-ledger "${SMOKE_ROOT}/ledgers/submission.json" \
  --task confirmation \
  --dry-run
```

No name-based or wildcard cancellation is used.

## 4. Bind measured resources

After a monitor report marks the complete miniature reusable, review `sacct`
usage and write production requests. Defaults are only a template; use
`--override TASK,CPUS,MEMORY,WALLTIME` for every measured adjustment.

```bash
python -s scripts/write_prad_resource_requests.py \
  --smoke-campaign-spec "${SMOKE_ROOT}/campaign_spec.json" \
  --review-note "Reviewed complete smoke sacct use and projected full split scaling" \
  --output "${SMOKE_ROOT}/production_requests.json"
python -s scripts/capture_prad_resources.py \
  --smoke-campaign-spec "${SMOKE_ROOT}/campaign_spec.json" \
  --submission-ledger "${SMOKE_ROOT}/ledgers/submission.json" \
  --dry-run-report "${SMOKE_ROOT}/dry_run.json" \
  --monitor-report "${SMOKE_ROOT}/monitor.json" \
  --production-requests "${SMOKE_ROOT}/production_requests.json" \
  --output "${SMOKE_ROOT}/resource_evidence.json"
python -s scripts/measure_prad_storage.py \
  --resource-evidence "${SMOKE_ROOT}/resource_evidence.json" \
  --path /home/ryreu/atlas/HLT_Classification/artifacts \
  --output "${SMOKE_ROOT}/storage_evidence.json"
```

The resource evidence queries only ledger job IDs, requires every state to be
`COMPLETED`, records elapsed time/MaxRSS/CPUs, and binds the reviewed request
table and campaign artifact size to the exact clean source.
The storage measurement requires the reviewed projection to be at least the
repository's conservative full-campaign estimate (currently about 29 GiB)
and, by default, leaves another 100 GiB free after the projected peak. That is
a fail-safe worst case in which every validation graph lies inside the
one-percent confirmation window. Actual checkpoint bytes and the number of
confirmed graphs must be taken from the new minimum-storage miniature rather
than from the older disk-backed smoke.

## 5. Authorized production submission

Only after explicit authorization:

```bash
PROD_ROOT=/home/ryreu/atlas/HLT_Classification/artifacts/prad/production_min_v2
python -s scripts/create_prad_campaign.py \
  --mode production \
  --campaign-root "${PROD_ROOT}" \
  --resource-evidence "${SMOKE_ROOT}/resource_evidence.json" \
  --storage-evidence "${SMOKE_ROOT}/storage_evidence.json" \
  --authorize-production \
  --output "${PROD_ROOT}/candidate_spec.json"
python -s scripts/submit_prad_campaign.py \
  --campaign-spec "${PROD_ROOT}/candidate_spec.json" \
  --dry-run \
  --dry-run-report "${PROD_ROOT}/dry_run.json"
```

Review that second dry run. Full submission is then:

```bash
bash sbatch/submit_prad.sh \
  --campaign-spec "${PROD_ROOT}/candidate_spec.json" \
  --production-dry-run-report "${PROD_ROOT}/dry_run.json" \
  --resource-evidence "${SMOKE_ROOT}/resource_evidence.json" \
  --storage-evidence "${SMOKE_ROOT}/storage_evidence.json" \
  --full-production-submit
```

The production split is exactly 500,000/150,000/500,000 with seed 1337. Test
model outputs remain impossible until the predeclared selection is complete,
the finalist/execution locks exist, and each checkpoint consumes its own
execution claim in the single locked final wave. If a worker is preempted after
claim creation, a retry may recover only that same claim for the exact campaign,
source, cache, and checkpoint; immutable shards reject any drift.

## Expected evidence

The campaign root contains split checksums, the real data audit, train-only
statistics, compact best/final model checkpoints, validation
predictions/metrics and benchmarks, selection and final-test locks, compact
label-free test predictions, five-seed comparisons, statistical evidence,
plots, and (production only) `reports/final_report.md`. Rolling optimizer/RNG
checkpoints exist only while a run is incomplete. Large paired views,
structural targets, and teacher outputs are job-local and absent after worker
cleanup. Weak scientific performance does not fail or skip a registered row.

# PMARD Research-Compute Runbook

PMARD production is deliberately gated. Synthetic tests alone never authorize
a campaign or final-test branch read.

## Environment

```bash
export PROJECT_DIR=/home/ryreu/atlas/HLT_Classification
export DATA_ROOT=/home/ryreu/cms/data/ScoutingAK8_native_compact/2024/train
export PYTHONNOUSERSITE=1
source /home/ryreu/miniforge3-aarch64/etc/profile.d/conda.sh
conda activate atlas_kd_tigris
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_DIR}"
```

Confirm the exact clean commit is pushed and checked out. Never run from an
untracked or modified scientific source tree.

Define campaign-local paths (the example identity must be replaced, never
reused across changed source or manifests):

```bash
export CAMPAIGN_ROOT=/home/ryreu/atlas/HLT_Classification/checkpoints/pmard_smoke_<identity>
export SOURCE_MANIFEST=${CAMPAIGN_ROOT}/data/source_manifest.json
export SPLIT_MANIFEST=${CAMPAIGN_ROOT}/data/splits/split_manifest.json
export SMOKE_SPEC=${CAMPAIGN_ROOT}/campaign_spec.json
```

Create source and split artifacts, then create the clean-source smoke spec:

```bash
python -s scripts/validate_scouting_data.py --data-root "${DATA_ROOT}" --output "${SOURCE_MANIFEST}"
python -s scripts/build_scouting_splits.py --source-manifest "${SOURCE_MANIFEST}" --output-dir "$(dirname "${SPLIT_MANIFEST}")"
python -s scripts/create_pmard_campaign.py --source-manifest "${SOURCE_MANIFEST}" --split-manifest "${SPLIT_MANIFEST}" --campaign-root "${CAMPAIGN_ROOT}" --mode smoke --output "${SMOKE_SPEC}"
python -s scripts/submit_pmard_campaign.py --campaign-spec "${SMOKE_SPEC}" --output "${CAMPAIGN_ROOT}/dry_run_ledger.json"
```

The smoke ledger must contain `miniature_summary` and must not contain
`final_test`, `execution_lock`, or `aggregate_report`. Live smoke submission is
the same last command with `--execute`; it requires explicit user authority.

## Required order

1. Run `validate_scouting_data.py` against the immutable 53-file root.
2. Build and authenticate the seed-12345 file split.
3. Run the train/validation feature, p4-closure, category, and lost-track audit.
4. Run installed-Weaver FP32 parity.
5. Complete the five-fold matcher miniature and matching-only selector.
6. Complete the streamed end-to-end miniature, including deliberate resume,
   RAM cleanup, and resource/storage capture.
7. Create and inspect a dry-run DAG. This must not call `sbatch`.
8. Freeze measured resource requests and obtain explicit production authority.
9. Submit with `sbatch --parsable`; retain the exact campaign-local numeric-ID
   ledger. Cancel only IDs from that ledger.
10. Create finalist and execution locks before the one-time final-test claim.

After the live smoke monitor marks every exact job reusable, capture exact-ID
usage (elapsed time, RSS, GPU memory, ROOT bytes/wait, and RAM-temp peak) in the
documented JSON map and run `capture_pmard_evidence.py`. Render the separate
full production preview with `dry_run_pmard_campaign.py`. Production spec
creation accepts those artifact files, not hand-entered hashes, and still
requires `--authorize-production`.

ROOT is the only durable dataset. Match indices, repaired tensors, and training
teacher targets remain in job-owned RAM and are deleted on exit. Completed jobs
retain compact model/report artifacts; interrupted jobs retain one rolling
resume checkpoint.

Poor performance is reported and does not cancel registered rows. Invalid
inputs, lineage drift, nonfinite required quantities, corrupt artifacts, or
forbidden final-role access fail closed.

# PMARD Research-Compute Runbook

PMARD production is deliberately gated. Synthetic tests alone never authorize
a campaign or final-test branch read.

This runbook follows campaign spec v9 and the fitted-strict selective endpoint.
Any older description of M0--M5 cross-fit training, 100% token coverage,
ephemeral full-role assignment tables, or `FULL_PARTICLE_ENDPOINT/v1` as the
primary path is superseded by the workflow below.

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

Before high-data production, create the registered pilot by changing the mode
to `pilot` and using a new campaign root. It selects exactly 300,000 train,
100,000 validation, and 100,000 final-test jets. Unlike smoke, pilot includes
the finalist/execution locks and sealed final evaluation, so it must be treated
as a real one-time final-role execution for that pilot identity.

## Required order

1. Run `validate_scouting_data.py` against the immutable 53-file root. This
   performs a bounded scalar-only streaming pass to authenticate the per-file
   baseline and 15-class counts; it does not create a replacement dataset.
2. Build and authenticate the seed-12345 version-2 split. Confirm 60/20/20
   targets, 32/11/10 complete-file membership, and no class-by-role fraction
   deviation above 0.02 before continuing.
3. Run the train/validation feature, p4-closure, category, lost-track, and
   full-endpoint common-field semantics audit.
   Require relative p4 closure tolerance `1e-4` and failing-object fraction at
   most `1e-4` independently for HLT, charged offline, and neutral offline.
4. Build the deterministic row-selection manifest. Smoke is 4,096/4,096,
   pilot is 300k/100k, and production is all mapped train/validation jets.
   Confirm exact totals and largest-remainder class counts.
5. Lock the bundled `fitted_strict` artifact at threshold
   `0.9828147479721088`. Build every train/validation source shard exactly once,
   finalize the compact assignment manifest, and confirm the authorization
   binds matcher, threshold, split, row selection, assignment manifest, all
   categories, and unmatched-as-exact-HLT semantics.
6. Run installed-Weaver FP32 parity and the streamed end-to-end miniature,
   including deliberate resume,
   RAM cleanup, and resource/storage capture.
7. Create and inspect a dry-run DAG. This must not call `sbatch`.
8. Freeze measured resource requests and obtain explicit production authority.
9. Submit with `sbatch --parsable`; retain the exact campaign-local numeric-ID
   ledger. Cancel only IDs from that ledger.
10. Create finalist and execution locks before selecting or reading pilot/full
    final-test rows. Only then build final-test assignment shards and make the
    one-time final-test claim.

After the live smoke monitor marks every exact job reusable, capture exact-ID
usage (elapsed time, RSS, GPU memory, ROOT bytes/wait, and RAM-temp peak) in the
documented JSON map and run `capture_pmard_evidence.py`. Render the separate
full production preview with `dry_run_pmard_campaign.py`. Production spec
creation accepts those artifact files, not hand-entered hashes, and still
requires `--authorize-production`.

ROOT remains the only particle dataset. The campaign additionally retains a
compact sparse assignment cache: source entry numbers plus accepted HLT/offline
indices and quantized confidence. It never stores particle fields or repaired
views. Repaired tensors and teacher targets remain job-owned RAM. Completed
jobs retain compact model/report artifacts; interrupted jobs retain one rolling
resume checkpoint.

Teacher/student jobs lazily reload the authenticated per-source assignment
shards and precompute FP32 teacher logits once per frozen teacher. Train chunks are
round-robined across eight files and mixed in a deterministic 32,768-row RAM
buffer. After logit targets are cached, `R0` and zero-coefficient
representation controls use HLT-only batches and release unused teacher and
matcher models; positive-coefficient representation KD alone retains aligned
privileged batches. CE/KL/representation losses remain FP32 under BF16 model autocast.
If alpha selection returns zero, the aligned privileged batch is an identity
alias of the HLT view, the representation teacher is the shared frozen T0,
and generational companions use HLT initialization plus anchor KD. No
registered representation or generation row is skipped, and no offline or
matcher input is opened for that identity endpoint.
The default fixed budget performs eight validation/checkpoint boundaries and
retains FP32 interval-mean losses every 50 updates in the completed report;
preemption also publishes the one
rolling exact-resume checkpoint. Checkpointable Slurm tasks request
`--signal=B:USR1@120`, and the batch worker `exec`s Python so that batch-shell
signal reaches the installed handler. This gives the engine a two-minute
pre-timeout warning. The
confidence-weighted ablation uses the stored quantized probabilities. Every
ordinary smoke teacher and student consumes the same selected 4,096 rows per
role. The single `ram_cache_miniature` task uses that selection plus
`--cache-teacher-targets` to exercise bounded assignment/logit caching and the
subsequent HLT-only join path. Verify these modes and peak RSS in the miniature
rather than assuming the RAM estimate.

The smoke DAG matches only its selected 4,096/4,096 rows. Pilot and production
assignment builders are CPU array jobs over immutable source files and are the
only campaign stage that runs the fitted matcher. Reruns validate and reuse
identical shards; they never rematch during an epoch. Measure assignment bytes,
walltime, ROOT I/O, and peak RAM in the pilot before authorizing production.

Poor performance is reported and does not cancel registered rows. Invalid
inputs, lineage drift, nonfinite required quantities, corrupt artifacts, or
forbidden final-role access fail closed.

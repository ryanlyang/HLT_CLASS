# Tigris Workers

This directory contains thin, absolute-path Tigris Slurm workers and the
baseline submitter. Every worker validates the immutable campaign
specification and active clean source before considering reusable artifacts.

Create a source-bound specification first, then inspect the complete DAG:

```bash
python -s scripts/create_campaign.py \
  --mode smoke \
  --output /tmp/hlt_smoke_campaign_spec.json
bash sbatch/submit_baseline.sh \
  --campaign-spec /tmp/hlt_smoke_campaign_spec.json \
  --dry-run
```

`--smoke-simulate` performs a no-Slurm failure/recovery exercise.
`--smoke-submit` submits only a miniature. Full production additionally
requires an explicitly authorized production specification and authenticated
storage measurement plus successful smoke resource evidence.

PRAD uses the same absolute-path environment bootstrap and the same Tigris
account/partition contract. Create and inspect it before any submission:

```bash
python -s scripts/create_prad_campaign.py --mode smoke \
  --campaign-root /home/ryreu/atlas/HLT_Classification/artifacts/prad/smoke \
  --output /tmp/prad_smoke.json
bash sbatch/submit_prad.sh --campaign-spec /tmp/prad_smoke.json --dry-run
```

`--full-production-submit` is unavailable unless the immutable production
spec records explicit authorization plus hashes for the prior dry run, real
miniature, and measured resource evidence.

The complete smoke, monitoring, exact-ID recovery/cancellation, measured
resource, and authorized-production sequence is documented in
[`docs/PRAD_RUNBOOK.md`](../docs/PRAD_RUNBOOK.md).

HCWDL uses `run_hcwdl_task.sh`. It activates the exact project environment,
sets `PYTHONNOUSERSITE=1`, prepends `${CONDA_PREFIX}/lib`, and ends with
`exec python -s`, allowing Slurm `B:USR1` to reach the checkpointing process.
All future commands are generated locally from an immutable HCWDL spec. The
`shell_endpoint_qualification_lock` job includes `--hold`; release is a later,
separately authorized operation after the lineage-bound endpoint diagnostic
acknowledgement is written. Pilot/production resources remain planning values
until a genuine Tigris miniature publishes measured evidence. The measured
prelaunch candidate and the executable spec must share the same independently
hashed `HCWDL_COMMAND_PLAN/v1`; explicit submission authorization binds that
hash and the exact resource requests, avoiding any circular dependency on the
enclosing campaign-spec hash. A first bounded smoke can therefore use
explicitly authorized conservative bootstrap requests without being mislabeled
as measured; pilot and production cannot.

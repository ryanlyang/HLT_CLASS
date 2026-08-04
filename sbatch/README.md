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

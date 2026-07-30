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
storage measurement.

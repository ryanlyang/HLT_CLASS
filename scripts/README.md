# Command-Line Entry Points

Scripts in this directory will remain thin wrappers around
`src/hlt_classification`.

Implemented in Transfer Block 2:

- `preflight_data.py`: strict recursive schema and per-class capacity audit;
- `build_splits.py`: deterministic balanced construction and atomic immutable
  publication of the authenticated split manifest.

The preflight's diagnostic skip mode always reports itself as
non-production-eligible. The split builder has no unreadable-file skip path.

Implemented in Transfer Block 4:

- `build_offline_cache.py`: build or resume one bounded immutable offline cache;
- `build_hlt_cache.py`: deterministically degrade an authenticated offline
  parent into bounded HLT-v3 shards;
- `audit_hlt_cache.py`: authenticate all HLT shards and optional exact profile,
  replica, and degradation lineage.

Both builders publish only complete deterministic shards, require an explicit
source-snapshot SHA-256, and provide `--max-new-shards` for focused
interruption/resume validation. An intentionally incomplete build exits with
status 3 and has no final manifest. The HLT CLI emits progress as JSON lines.

Implemented out of order for Transfer Block 5:

- `validate_weaver_parity.py`: standalone installed-Weaver FP32 comparison of
  logits, input gradients, parameter gradients, masks, and state dictionaries.

It requires Weaver and PyTorch. On Tigris, activate `atlas_kd_tigris`, disable
the user site, and run:

```bash
PYTHONNOUSERSITE=1 python -s scripts/validate_weaver_parity.py --device cpu
```

The optional `--bf16-smoke --device cuda` path checks only runtime finiteness;
it is never a substitute for the default FP32 equivalence attestation.

Implemented in Transfer Block 6:

- `train_part.py`: fixed-budget canonical ParT training with exact resume;
- `evaluate_part.py`: ordered label-free inference followed by an
  identity-bound label join and frozen metrics.

Training configuration is a versioned JSON object. Poor performance never
changes the exit status or remaining update budget; invalid or nonfinite
execution fails closed.

Implemented in Transfer Block 7:

- `create_campaign.py`: capture a clean Git source snapshot and create one
  immutable smoke or explicitly authorized production specification;
- `validate_campaign.py`: revalidate the campaign and active source;
- `measure_storage.py`: bind storage headroom and task requests to a campaign;
- `capture_slurm_resources.py`: derive immutable elapsed-time, MaxRSS, CPU,
  and campaign-byte evidence from exact successful smoke job IDs;
- `capture_runtime_environment.py`: publish source-bound Python, package,
  CUDA, and GPU identity from the real training worker;
- `submit_campaign.py`: dry-run, failure-simulate, smoke-submit, or gated
  full-production submission using exact numeric Slurm job IDs;
- `assemble_submission_journal.py`: recover the exact already-submitted IDs
  if `sbatch` fails partway through constructing an initial or resume graph;
- `monitor_campaign.py`: query exact campaign IDs and authenticate every
  attested file before declaring a task reusable;
- `resume_campaign.py`: create a new failed-node-and-descendant recovery graph;
- `run_campaign_task.py`: shared production task executor used by the thin
  Slurm workers.

Every submit, monitor, resume, and worker entry path validates the active
clean source before checking reusable artifacts.

Full production requires both the storage measurement and its successful
smoke resource-evidence parent. The measured request table is passed into the
actual `sbatch` commands rather than merely validated and ignored.
Every successful `sbatch --parsable` return is durably journaled before the
next job is submitted, so a submit-side outage cannot lose already-created
job IDs or cause an ambiguous duplicate submission.

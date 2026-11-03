# Frozen adapter implementation evidence and donor inventory

Implementation base: `54473f108c4b1b435f799ca2e41a9c69a4ebe332`.
All new runtime code is confined to `luka_frozen_adapter_hydra/`, its thin CLI,
queue helper and sbatch worker. Existing model/campaign runtime is unchanged.
Scientific authority: [active plan](plans/LUKA_FROZEN_ADAPTER_HYDRA_PLAN.md).
New contracts: `LUKA_FROZEN_ADAPTER_HYDRA_<KIND>/v1`.

## 2026-11-03: SPORC export contract repair

The user-reported export from `d95975fa256360814225900284f72a1be1d94afc`
failed because `inputs.export()` called `validate_content_hash()` without its
required `expected_contract` keyword. This happened before the export directory
was created or any payload copied; the export command submits no jobs.

The repair supplies the existing `LUKA_FULLSIM_ALLJETS_CACHE/v1` contract,
matching the prepared manifest producer. Hash, schema, source and payload checks
remain in force. No scientific recipe, reusable contract version, scheduling or
storage policy changed. Modified runtime donor: this campaign's `inputs.py` at
the commit above. No external donor or Fresh_check migration.

The new exporter roundtrip regression reproduced the exact TypeError before
the repair. It now exercises actual copying, manifest publication, destination
verification, source preservation and minimal transfer scope. Six additional
cases reject wrong contract/schema, bad hash, corrupt checkpoint/shard and a
missing shard before output creation. Only the external original-checkout
authentication is mocked in these exporter tests; the validator and exporter
are real. The earlier tests covered delta import but missed this export call.

**71 focused tests passed in 42.92 seconds** using the local environment below:

```powershell
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -m pytest -q -p no:cacheprovider tests/test_luka_frozen_adapter_hydra.py tests/test_luka_u100_pair_transfer.py --basetemp=tmp/frozen-export-fixed-d95975fa
```

These suites include the existing real-Weaver/CUDA synthetic workers; remote
SPORC export and Hydra acceptance after this repair remain unverified. Next:
push the repair and rerun the runbook export with the new commit, then copy and
queue using its new manifest hash. Preserve the old source checkout.

## Internal donors at the base commit

* `luka_gating/model.py`, `training.py`: frozen native backbone/eval convention,
  additive attention hook and Torch 2.5.1 bias-only backward constraint. The old
  gate's KD recipe, warm-up/unfreezing policy and shared-head scalar gate are
  **not** this campaign's scientific implementation.
* `luka_embedding_aux/model.py`: installed-Weaver classifier-input tap and
  independent jet-embedding predictor. `luka_u100_embedding/contracts.py`,
  `targets.py`, `training.py`: matched U100 SmoothL1 weight5, authenticated bank,
  counts and full30 minibatch schedule.
* `luka_u100_pair_correction/geometry.py`: four physical pair-feature formulas;
  `model.py`, `targets.py`: reference for contextual correction and slot joins.
* `jetclass2_delphes/model.py`, `runner.py`, `batch128_30e_recipe.py`,
  `reporting.py`, `execution.py`: native model, selection key, AdamW schedule,
  metrics and GPU identity. Installed Weaver0.5.3 supplies unchanged Attention
  and Block implementations; no vendored code is modified.
* `luka_u100_pair_transfer/bundle.py`: original-source replay, safe regular paths,
  exact original campaign and transfer manifest. `luka_u100_pair_hydra/inputs.py`,
  `portability.py`, `site.py`, `submission.py`: allowlisted Brown relocation,
  completed precision evidence, Hydra allocation and reviewed walltime envelope.
* `luka_u100_contrastive_hydra/campaign.py`, `submission.py`, `worker.py`,
  `transfer.py`: additive source pin, independent exact-DAG submission, immutable
  receipts and minimal transfer patterns. Reuse only its fixed eight-GiB storage
  helper; new scientific kernels do not import contrastive objectives/sampling.
  `luka_u100_contrastive/training.py` supplies resource-peak observation only.
* `luka_fullsim/foundation.py`, `regular_cache.py`, `regular_submission.py`,
  `preflight_v2.py`, `alljets_data.py`: source evidence, authenticated prepared
  shards, clean Slurm environment, backend checks and dataset scope flags.
* `data/cache_contracts.py`, `cms_proxy_ladder/contracts.py`, `submission.py`,
  `context_fusion/worker.py`: content hashes, immutable publication, file refs,
  walltime formatting and tensor-state serialization.
* `scouting/hcwdl_exact_dag_submission.py`, `hcwdl_recovery.py`: accepted-ID
  journals and ledger assembly.
* `scripts/queue_luka_u100_contrastive_hydra.sh`,
  `sbatch/run_luka_u100_contrastive_hydra.sh`, and
  `scripts/luka_u100_contrastive_hydra.py`: detached queue/worker/CLI structure.
* `tests/test_luka_attention.py`, `test_luka_embedding_aux.py`,
  `test_luka_gating.py`, `test_luka_u100_contrastive_hydra.py`: synthetic real-Weaver
  caches, native model fixtures, role/gradient and exact-submission test patterns.

No Fresh_check runtime import or external donor migration; no license changes.

## Local evidence

Before changes: **85 focused baseline tests passed** in 144.34 seconds (gating,
U100 embedding, Hydra portability). Final regression: **176 tests passed in
270.33 seconds**, covering the new adapter, gating, U100 embedding, pair Hydra,
pair Hydra portability and contrastive Hydra suites. The new package contributes
37 tests, including a real CUDA/BF16 synthetic shared setup with all fourteen
production miniatures, followed by full thirty-epoch oracle and combined-student
workers and exact parent/receipt validation. The standalone new suite also
passed in 58.33 seconds before the final scoped-SDPA regression run.

Final command (Windows PowerShell, with `PYTHONDONTWRITEBYTECODE=1`,
`PYTHONNOUSERSITE=1`, `PYTHONPATH=src;tmp/hydra-weaver-fixture-20261102`, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`):

```powershell
& C:/Users/22rya/miniconda3/envs/tagging-hlt/python.exe -m pytest -q -p no:cacheprovider tests/test_luka_frozen_adapter_hydra.py tests/test_luka_gating.py tests/test_luka_u100_embedding.py tests/test_luka_u100_pair_hydra.py tests/test_luka_u100_pair_hydra_portability.py tests/test_luka_u100_contrastive_hydra.py --basetemp=tmp/frozen-adapter-final-20261103
```

Tests use the real pinned Weaver0.5.3 sources from
`tmp/hydra-weaver-fixture-20261102`, Torch2.5.1, and local CUDA where available.
They cover frozen parameters **and buffers**, CUDA/BF16 mask-only gradients,
teacher-target detach, head/message alignment, native pair conditioning,
permutation/masking, compact-versus-dense edge-message equivalence, deployment
head removal/reload, both losses, full30 synthetic fits, transfer corruption,
source-scope rejection, and the exact dependency graph. Synthetic fixtures and
mocked scheduler/site admission are not remote Slurm evidence.

Git Bash syntax checks for both new shell scripts, all runbook Bash snippets,
the thin CLI help check, new Python AST/whitespace checks, documentation links
and tracked whitespace checks passed. Native zero-adapter identity is
mathematical; CPU fused-kernel differences are tested with FP32 tolerances
(2e-6 absolute, 1e-5 relative), not described as cross-backend bitwise equality.

## Remote status and exact next action

No remote job has been submitted by this implementation. No new L40 resource
acceptance or real JetClass miniature has run. The existing precision diagnostic
qualifies the reused numerical environment; it does not certify new adapter
memory/time or performance. The shared setup will run the new production
miniatures before scientific submission. This is locally queue-ready code,
conditional on that real-data acceptance, not completed production validation.

Next: commit the scoped files, export/copy the small M0HLT + validation-U100
delta, and invoke the new helper as documented in the
[runbook](LUKA_FROZEN_ADAPTER_HYDRA_RUNBOOK.md). Existing SPORC/Hydra campaigns,
datasets, source worktrees and saved controls remain unchanged.

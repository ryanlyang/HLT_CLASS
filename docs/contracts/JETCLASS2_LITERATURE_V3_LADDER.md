# Frozen literature V3 ladder contracts

Authority: ../plans/JETCLASS2_LITERATURE_V3_200K_DIRECT_COARSE_PLAN.md.

The shared JETCLASS2_CMS_PROXY_LADDER namespace names reusable classifier
tooling, not the generator. New versions explicitly bind the literature
dataset; older versions retain their original semantics.

- RELEASE_REQUEST/v2 and RELEASE/v2: 200k/50k label-blind ordinary roles,
  source STUDY byte reference, literal JC2_LITERATURE_PRODUCTION_STUDY/v1,
  controlled_literature_synthetic_proxy, NOISE_V3, complete role manifests.
- FOUNDATION/v2 and VIEWS/v2 label the new endpoint truthfully as frozen
  literature V3, with unchanged reusable matching/interpolation semantics.
  INPUTS/v1 remains the same physical schema. Release parent hashes necessarily
  differ from CMS studies; old FOUNDATION/v1/VIEWS/v1 retain CMS endpoint labels.
- GATE_SPEC/v7 and RUNTIME_PROFILE/v7: fresh literature endpoints, SPORC debug
  for measurement and science, 16 workers/160000 MiB/one A100, full 200k/50k
  caches, real one-pass fit/reducer, installed-Weaver FP32 parity and no site
  transfer. Measured train/reduce time requests must not exceed 1440 minutes.
- SCIENTIFIC_PLAN/v3 and CAMPAIGN_SPEC/v3: exactly DIRECT and COARSE,
  LITV3-prefixed branch nodes, nine fresh fits and five reducers. No imported
  weights, population-selection relabeling, multi-parent KD, or test metrics.

Every new artifact binds canonical content hash, exact parent hashes and
final_test_accessed=false. Byte references are checked before reuse. Source
locking includes the new wrapper/plan/contracts and producer-consumer code.
The consumer authenticates the original production study without requiring
the classifier to run inside the old generator worktree or on its ARM host.
It verifies saved physical banks rather than rerunning the generator on x86.
Unknown source contracts, swapped releases, corrupted banks, nonfinite inputs,
incorrect canonical joins, capacity overflow and forbidden roles fail closed.

Live submission first checks the SPORC cluster and resource shapes using
`sbatch --test-only`, with inherited SBATCH/SLURM variables removed. Actual
submissions use that same sanitized environment, a canonical reviewed dry
ledger and an exclusive durable claim. A completed ledger is idempotently
reusable. A claim without a completed ledger requires inspection of exact
journals/jobs before explicit repair; it is not automatically retried.

The source final-test materialization flag is not a classifier test-access
authorization. This campaign may not read those particle banks. Training
reports include per-class QCD rejection at 50% efficiency using the existing
metric definitions, as well as accuracy/AUC and pure-offline recovery.

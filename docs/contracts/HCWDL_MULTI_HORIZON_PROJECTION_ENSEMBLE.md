# HCWDL Multi-Horizon Projection-Ensemble Contracts

The scientific authority is the
[HCWDL-MHPE-FULL implementation plan](../plans/HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_IMPLEMENTATION_PLAN.md).
This document records the reusable v1 artifact identities implemented by that
plan:

- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_NODE_SPEC/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FOUNDATION_REUSE_LOCK/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_SHARD/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_MANIFEST/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TARGET_LOCK/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_COMMAND_PLAN/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_STAGE_REPORT/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RUNTIME/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_AGGREGATE/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINALIST_LOCK/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_EXECUTION_LOCK/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_COMPLETE/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FINAL_EVALUATION/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECOVERY_SPEC/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RESOURCE_RECOVERY_SPEC/v1`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v1`.

The graph owns exactly 16 fresh fits and four probability ensembles. Every
specialist has one teacher. Component probabilities are formed in lexical
component order by max-subtracted FP32 softmax, canonical FP64 accumulation
and division, and a single little-endian FP32 publication. Probability targets
are consumed directly by KL at their declared temperature; they are never
treated as logits or temperature-transformed a second time.

The foundation reuse lock authenticates an already completed all-mapped
FULL3 foundation and the exact U000/M0paired checkpoints and rows. It is an
additive authorization and does not rewrite the foundation. The campaign may
read only train and validation roles. Final-test access remains zero until a
separate finalist lock and explicit execution lock exist.

All reused model, endpoint, coupling, cache, optimizer, checkpoint, and resume
scientific-core files remain byte-exact. Corrected foundation execution
builders are source-pinned by the new campaign and their completed products
are authenticated by the imported foundation lock; they are not compared to
the foundation specification's pre-recovery producer bytes. The two generic
training entry points that must
accept the new probability-target type are recorded separately in the reuse
lock with both their foundation and current hashes. Their old logit-target
defaults remain unchanged and are guarded by a numerical regression. This is
an explicit additive adapter, not an unrecorded relaxation of foundation
lineage.

The selected U050 model publishes one immutable identity-ordered FP32 logit
bank for its four direct consumers. Every ensemble reducer prepares the
all-mapped train and validation view once, evaluates each component once per
role, and publishes authenticated T1 and T2 probability bundles. Repaired
particle views remain process-local and are never durable.

Any change to graph edges, coordinates, components, probability reduction,
target temperature, loss weights, training passes, seeds, checkpoint
selection, population, views, endpoints, or final-test registry requires a
new version. Recovery preserves all scientific hashes and the exact failed
and downstream closure; resource recovery may only increase operational
requests and may not change the GPU class.

## Additive C10P90 companion

The [C10P90 parallel plan](../plans/HCWDL_MHPE_C10P90_PARALLEL_PLAN.md)
registers a paired companion that changes only non-M1 specialist loss weights
from `0.25 CE + 0.75 KD` to `0.10 CE + 0.90 KD`; both use temperature 2. M1
remains `0.10 CE + 0.90 KD` at temperature 1. Its changed semantic artifacts
use:

- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_GRAPH/v2`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_NODE_SPEC/v2`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_RECIPE/v2`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_CAMPAIGN_SPEC/v2`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_TRAINING_REPORT/v2`;
- `HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_OPERATIONAL_EVIDENCE_WAIVER/v2`.

Unchanged foundation reuse, probability targets, stage reporting, aggregate,
finalist/final-evaluation, submission, monitoring, attestation, and recovery
formats retain v1 while transitively binding the v2 campaign, graph, recipe,
and training-report hashes. The two campaigns use identical seed aliases and
separate roots and ledgers, making their comparison paired without permitting
cross-campaign output reuse.

## Additive paired 300k/60-pass study

The [paired 300k/60-pass plan](../plans/HCWDL_MHPE_300K_60E_PAIRED_PLAN.md)
registers the complete graph at 300k train, 100k validation, 100k sealed test,
and 60 passes. `C25P75_300K60` uses graph/node/recipe/campaign/training-report/
waiver v3; `C10P90_300K60` uses the corresponding v4 identities. Both bind the
`HCWDL_MULTI_HORIZON_PROJECTION_ENSEMBLE_FOUNDATION_REUSE_LOCK/v2` population
and immutable-product authorization. All probability-ensemble semantics and
the 16-fit topology remain unchanged. Population, passes, seeds, resources,
and specialist loss are immutable scientific identity.

## Additive dense anchor-50 300k/60-pass profile

The [dense anchor-50 plan](../plans/HCWDL_MHPE_DENSE_ANCHOR50_300K60_PLAN.md)
registers 29 cold C10P90 fits over U033, U066, U100, D75, D50, D25, D0,
and M1. Six durable ensembles assign exactly one half of their probability
mass to the immediate-predecessor specialist and divide the other half
equally among skip specialists. It uses graph/node/recipe/campaign/training
v5, foundation-reuse v3, and target/stage/aggregate/finalist/completion/final
v2 identities. Old v1-v4 artifacts retain their original uniform semantics.
The uniform ensemble is validation-only diagnostics and may not supervise a
child. See the [dense runbook](../HCWDL_MHPE_DENSE_ANCHOR50_300K60_RUNBOOK.md).

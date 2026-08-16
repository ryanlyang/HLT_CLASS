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

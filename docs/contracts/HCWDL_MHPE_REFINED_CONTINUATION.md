# HCWDL-MHPE Refined Continuation Contracts

The scientific authority is the
[refined-continuation plan](../plans/HCWDL_MHPE_REFINED_CONTINUATION_300K60_PLAN.md).

The additive v1 family is:

- `HCWDL_MHPE_REFINED_CONTINUATION_GRAPH/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_NODE_SPEC/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_RECIPE/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_CAMPAIGN_SPEC/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_COMMAND_PLAN/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_TRAINING_REPORT/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_TARGET_SHARD/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_TARGET_MANIFEST/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_TARGET_LOCK/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_STAGE_REPORT/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_RUNTIME/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_AGGREGATE/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_CAMPAIGN_COMPLETE/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_RECOVERY_SPEC/v1`;
- `HCWDL_MHPE_REFINED_CONTINUATION_RECOVERY_COMMAND_PLAN/v1`.

The target family stores identity-ordered T1 probabilities only. An augmented
ensemble with `K` authenticated source specialists assigns `K/(K+1)` to their
existing uniform aggregate and `1/(K+1)` to the new specialist. This is exactly
`1/(K+1)` per underlying specialist. The implementation must reload the `K`
source components, reproduce the stored source aggregate byte-for-byte, and
run one canonical `K+1` reduction so intermediate FP32 rounding cannot alter
the result. Locks bind source bundles, new reports,
checkpoints and logits, population manifests, campaign, and producer source.
Final-test access is forbidden.

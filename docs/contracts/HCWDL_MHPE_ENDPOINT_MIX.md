# HCWDL-MHPE Endpoint Mixture Contract

Version 1 defines a validation-only downstream add-on over one completed dense
300k MHPE campaign. Its reusable identities are:

```text
HCWDL_MHPE_ENDPOINT_MIX_GRAPH/v1
HCWDL_MHPE_ENDPOINT_MIX_NODE_SPEC/v1
HCWDL_MHPE_ENDPOINT_MIX_RECIPE/v1
HCWDL_MHPE_ENDPOINT_MIX_TARGET_SHARD/v1
HCWDL_MHPE_ENDPOINT_MIX_TARGET_MANIFEST/v1
HCWDL_MHPE_ENDPOINT_MIX_TARGET_LOCK/v1
HCWDL_MHPE_ENDPOINT_MIX_CAMPAIGN_SPEC/v1
HCWDL_MHPE_ENDPOINT_MIX_COMMAND_PLAN/v1
HCWDL_MHPE_ENDPOINT_MIX_TARGET_BUILD_REPORT/v1
HCWDL_MHPE_ENDPOINT_MIX_TRAINING_REPORT/v1
HCWDL_MHPE_ENDPOINT_MIX_RUNTIME/v1
HCWDL_MHPE_ENDPOINT_MIX_AGGREGATE/v1
HCWDL_MHPE_ENDPOINT_MIX_CAMPAIGN_COMPLETE/v1
HCWDL_MHPE_ENDPOINT_MIX_RECOVERY_SPEC/v1
HCWDL_MHPE_ENDPOINT_MIX_RECOVERY_COMMAND_PLAN/v1
```

The graph contains exactly the four weights `1`, `9/10`, `3/4`, and `1/2`.
The source D0E probability and M0paired probability are identity joined.
Targets use max-subtracted FP32 softmax, exact-rational FP64 accumulation, and
little-endian FP32 publication. All students use exact HLT, paired seeds,
unweighted 10% CE plus 90% probability KD at temperature one, and 60 passes.

Every artifact binds its complete parent lineage and content hash. Source
campaign artifacts are read-only. Labels never enter target construction.
Final-test data are inaccessible to this contract family.

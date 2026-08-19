# HCWDL-MHPE D000 Teacher-Distance Schedule-Screen Contracts

Authority: [full-data implementation plan](../plans/HCWDL_MHPE_D000_TEACHER_DISTANCE_SCHEDULE_SCREEN_FULL_PLAN.md).

The additive validation-only family is:

- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_GRAPH/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_NODE_SPEC/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RECIPE/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_VALIDATION_PARTITION/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_SOURCE_READINESS/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_SOURCE_REUSE_LOCK/v2`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_OPERATIONAL_EVIDENCE_WAIVER/v2`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_CAMPAIGN_SPEC/v2`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_COMMAND_PLAN/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_HORIZON_CHECKPOINTS/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_TRAINING_REPORT/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RUNTIME/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_AGGREGATE/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_CAMPAIGN_COMPLETE/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RECOVERY_SPEC/v1`;
- `HCWDL_MHPE_D000_TEACHER_DISTANCE_SCREEN_RECOVERY_COMMAND_PLAN/v1`.

The readiness artifact and source-reuse lock require an authenticated all-
mapped full-data `C25P75` MHPE source, its exact U000 report/checkpoint/logits,
and completed U100E, D066E, and D033E temperature-2 probability bundles.
Whole-source-campaign completion is explicitly not required. The lock
authorizes only the declared 24 new consumers.

Every fit consumes exact HLT, runs 80 passes, and binds six possible peak LRs,
one of four teachers, C25P75/T2, and a shared pairing seed. Its horizon manifest
authenticates the best-up-to-20/40/60/80 checkpoints, selected updates and
checkpoint-validation metrics. The scoring report binds one held-out metric
row per horizon. Horizon scoring cannot control selection or completion.

Final-test access is false in every artifact. Any change to teachers, LR grid,
80-pass schedule, horizon definitions, checkpoint key, validation partition,
loss, temperature, seeds, input, or scoring semantics requires a new contract.

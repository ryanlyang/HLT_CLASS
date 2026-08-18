# HCWDL-MHPE D066 Schedule-Screen Contracts

Authority: [full-data implementation plan](../plans/HCWDL_MHPE_D066_SCHEDULE_SCREEN_FULL_PLAN.md).

The executable full-data C25P75 study uses the additive v3 family:

- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_GRAPH/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_NODE_SPEC/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECIPE/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_VALIDATION_PARTITION/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_SOURCE_REUSE_LOCK/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_SOURCE_READINESS/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_OPERATIONAL_EVIDENCE_WAIVER/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_SPEC/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_COMMAND_PLAN/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_TRAINING_REPORT/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RUNTIME/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_AGGREGATE/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_COMPLETE/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_SPEC/v3`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_COMMAND_PLAN/v3`.

The source-reuse lock binds the full-data C25P75 campaign, all-mapped
foundation and foundation lock, exact role counts, selection, validation
assignment manifest, source recipe, U000/U050 reports and train targets, and
U100E/T2 probability lock. Source completion is not an input; a separate
readiness artifact proves every consumed product is complete and immutable.

The validation-partition contract binds all full validation identities from
authenticated assignment shards and the label-free global SHA-rank half split.
The recipe binds the resulting dynamic checkpoint and scoring row counts.

Each report binds source, graph, recipe, partition, teacher target, selected
engine report/checkpoint, checkpoint-half metrics, and untouched scoring-half
metrics. The runtime report proves the full train/checkpoint caches were
released before the scoring cache was created. Final-test access is false in
every artifact.

The v1 C10P90/300k and v2 C25P75/300k schemas were never submitted and are
retired. They are not readable as v3 and may not authorize this campaign.

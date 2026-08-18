# HCWDL-MHPE D066 Schedule-Screen Contracts

The scientific authority is the
[300k D066 schedule-screen plan](../plans/HCWDL_MHPE_D066_SCHEDULE_SCREEN_300K_PLAN.md).

The additive v1 family is:

- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_GRAPH/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_NODE_SPEC/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECIPE/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_VALIDATION_PARTITION/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_SOURCE_REUSE_LOCK/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_OPERATIONAL_EVIDENCE_WAIVER/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_SPEC/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_COMMAND_PLAN/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_TRAINING_REPORT/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RUNTIME/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_AGGREGATE/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_COMPLETE/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_SPEC/v1`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_COMMAND_PLAN/v1`.

The graph contains exactly 60 independent D066 students, 20 schedules, and
three authenticated teacher horizons per schedule. All train/checkpoint/
scoring identities, teacher target hashes, graph, recipe, source commit,
resources, and semantic source files are parents of the campaign spec.

Any change to grid values, teacher set, source profile, validation partition,
loss, temperature, architecture, D066 view, seed aliases, optimizer, schedule
shape, selection/ranking policy, resources, or final-test registry requires a
new semantic version. Recovery may only execute the exact failed/downstream
closure and preserve already authenticated outputs.

# HCWDL-MHPE D066 Schedule-Screen Contracts

The scientific authority is the
[300k D066 schedule-screen plan](../plans/HCWDL_MHPE_D066_SCHEDULE_SCREEN_300K_PLAN.md).

The executable C25P75 study uses the additive v2 family:

- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_GRAPH/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_NODE_SPEC/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECIPE/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_VALIDATION_PARTITION/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_SOURCE_REUSE_LOCK/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_OPERATIONAL_EVIDENCE_WAIVER/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_SPEC/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_COMMAND_PLAN/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_TRAINING_REPORT/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RUNTIME/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_AGGREGATE/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_CAMPAIGN_COMPLETE/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_SPEC/v2`;
- `HCWDL_MHPE_D066_SCHEDULE_SCREEN_RECOVERY_COMMAND_PLAN/v2`.

The graph contains exactly 60 independent D066 students, 20 schedules, and
three authenticated teacher horizons per schedule. All train/checkpoint/
scoring identities, teacher target hashes, graph, recipe, source commit,
resources, and semantic source files are parents of the campaign spec.

Any change to grid values, teacher set, source profile, validation partition,
loss, temperature, architecture, D066 view, seed aliases, optimizer, schedule
shape, selection/ranking policy, resources, or final-test registry requires a
new semantic version. Recovery may only execute the exact failed/downstream
closure and preserve already authenticated outputs.

The pushed but unexecuted v1 definition authenticated C10P90. It produced no
campaign root or ledger and is retired. v2 authenticates completed
`C25P75_300K60` products and fixes every screen student to C25P75/T2.

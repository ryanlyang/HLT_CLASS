# HCWDL TRI60 D000 matched-seed diversity contracts

This additive `/v1` family implements
[`HCWDL_TRI60_D000_SD5_ABLATION_PLAN.md`](../plans/HCWDL_TRI60_D000_SD5_ABLATION_PLAN.md).
It neither changes nor broadens the source TRI60 or CE5 contracts.

The graph contains exactly five fresh exact-HLT D000 students. Their frozen
probability teachers are `U000`, `LOGIT_U050E`, `LOGIT_U100E`,
`LOGIT_D066E`, and `LOGIT_D033E`; their seed aliases are exactly those of
`CE5_S01` through `CE5_S05` in the same order. Each node fixes C25P75/T2,
60 passes, batch size 256, fresh initialization, and no representation loss.
The graph fixes a uniform five-member temperature-one probability ensemble.

The campaign specification binds the source TRI60 and CE5 campaign hashes,
foundation, recipe, endpoint resource evidence, source teacher probability
locks, source U000 and LOGIT D000E reports, CE5E report, graph, source commit,
full role counts, task DAG, and resource registry. Source artifacts are
read-only parents and never scheduler dependencies.

Training reports and selected/final checkpoints bind the campaign, recipe,
graph, exact node payload, frozen source probability lock, RNG domains, and
source commit. They are published only after all 60 passes complete; no resume
checkpoint is durable.

The ensemble report binds the five selected reports and checkpoints, fixed
component order and weights, exact validation identities, component and
leave-one-out metrics, diversity diagnostics, and the authenticated source
comparators. It explicitly declares that no probability bank, logits, or
particle views are durable. The aggregate and completion artifacts preserve
the same lineage and make scientific quality non-controlling.

No artifact grants ordinary final-test capability. Invalid lineage, missing
roles, corrupt identities, nonfinite required quantities, stale source, or an
unauthorized source tree fail closed. Poor finite validation performance does
not fail or prune the campaign.

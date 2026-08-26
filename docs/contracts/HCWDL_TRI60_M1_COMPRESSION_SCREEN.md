# HCWDL TRI60 M1 compression-screen contract

The `HCWDL_TRI60_M1_COMPRESSION_SCREEN_* /v1` family binds an additive
validation-only study to one immutable TRI60 source subset.

The source lock hashes the source campaign, foundation, recipe, source graph,
`LOGIT_D000E` probability artifacts and stage report, selected `M1_LOGIT`, and
selected `LOGIT_D000_from_D033E`. Path existence alone is insufficient;
content hashes, report node specifications, checkpoint bytes, population
roles, and no-final-test assertions are revalidated before reuse.

The graph contract fixes 20 conditions, 19 fresh fits, initialization source,
CE/KD mixture or exact ramp, temperature, peak learning rate, shared stochastic
domains, 60 passes, and batch 256. A semantic change requires a new graph and
contract version.

Warm and polish initialization inherit model parameters only. All optimizer,
learning-rate schedule, RNG, and sampler state is fresh and explicitly bound
to the screen condition. Temperature two means
`softmax(log(p_LOGIT_D000E)/2)` and is never presented as reconstructed
component-logit KD.

The campaign and command-plan contracts prohibit source scheduler
dependencies and source mutation, set nice 5000, prohibit final-test access,
and register exact task dependencies. Training reports bind both selected and
terminal checkpoint hashes plus initialization lineage and loss schedule.
Poor metrics remain valid scientific outputs.

Monitor and recovery contracts operate on exact campaign job IDs and task
attestations. Recovery restarts an incomplete fit from update zero and may
clean only that task's registered study-root output scope.

# HCWDL TRI60 CE5 seed-ensemble reviewer contracts

This additive `/v1` family implements
[`HCWDL_TRI60_CE5_SEED_ENSEMBLE_REVIEWER_PLAN.md`](../plans/HCWDL_TRI60_CE5_SEED_ENSEMBLE_REVIEWER_PLAN.md).
It does not modify or broaden any TRI60 campaign contract.

The graph contains exactly five fresh CE-only teacher nodes, one C10P90/T1
ensemble-distilled student, one seed-matched CE-only student, and the fixed
uniform `CE5E` probability distribution. All nodes consume exact HLT inputs
through the ordinary unified model. Teachers have distinct frozen seed
aliases. The two students have the same frozen seed alias and differ only in
their declared loss/teacher fields.

The probability family contains complete-role shards, manifests, and a lock.
It binds the exact five-member lexical component order, selected report and
checkpoint lineage, temperature one, canonical identities, class order, and
FP64 uniform accumulation followed by one FP32 cast. Only identities and 15
class probabilities are durable.

Training reports and selected/final checkpoints use an additive authority over
the established TRI60 no-resume trainer. Each report binds the campaign,
source campaign, foundation, recipe, graph, node, and source commit. The
`CE5_KD` report additionally binds the `CE5E` probability lock. Successful and
poor finite results both complete normally. Final test is unavailable.

The campaign family contains specification, graph, node, command-plan,
aggregate, completion, monitor, and source-pinned restart-from-zero recovery
contracts. Submission is journaled by exact task ID. Recovery may clean only
the exact incomplete outputs of failed/downstream tasks inside this campaign
root; completed attestations remain immutable. Neither execution nor recovery
can reference, cancel, hold, or mutate source-campaign jobs or artifacts.

# TRI60 D000-from-D033E 180-pass contracts

Authority: [implementation plan](../plans/HCWDL_TRI60_D000_D033E_LONG180_PLAN.md).

The `/v1` family authenticates the graph, additive node, source lock,
campaign spec, command plan, training/checkpoint products, and compact
comparison report.

The source lock requires the original full-data TRI60 campaign, its completed
60-pass `LOGIT_D000_from_D033E` report selected at pass 60, the selected
checkpoint bytes, the complete temperature-2 `LOGIT_D033E` train and
validation probability manifests and lock, teacher stage, foundation, recipe,
graph, role counts, and no-final-test assertions.

The new node is exact HLT, fresh, C25P75/T2, batch 256, and uses the original
node's seed alias. Its only changed training fields are 180 passes, total
updates, and the resulting cosine horizon. The report contains 180 validation
rows and globally selects by the existing macro-AUC-first rule. The comparison
retains pass 60/120/180 metrics without asserting 60-pass schedule
equivalence.

The command plan contains exactly one source-independent, nice-10000 GH200
job with 72 CPUs, 320 GiB RAM, and a 72-hour walltime. Source outputs are
read-only, teacher probabilities are not copied, particle views remain in
RAM, rolling resume is forbidden, final test is inaccessible, and scientific
performance cannot control completion.

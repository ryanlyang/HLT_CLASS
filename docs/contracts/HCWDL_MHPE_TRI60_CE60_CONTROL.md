# HCWDL-MHPE TRI60 CE60 control contracts

This additive `/v1` family implements the separately registered control in
[`HCWDL_MHPE_TRI60_CE60_CONTROL_PLAN.md`](../plans/HCWDL_MHPE_TRI60_CE60_CONTROL_PLAN.md).
It does not version, broaden, or relabel the immutable three-track graph.

The family contains graph, node, specification, command-plan, training-report,
selected-checkpoint, and final-checkpoint contracts. Every artifact is
content-hashed and binds the exact source campaign, foundation, recipe,
all-mapped population, replicate seed, code commit, and control graph.

The graph contains exactly one `M0CE60` node. Its input is exact HLT, its
architecture is the ordinary unified 21-channel Particle Transformer, its
loss is 100% unweighted CE, and its budget is 60 passes with validation every
pass. Its stochastic seed alias equals the immutable source graph's `M2`
alias. KD, offline inputs, representation targets, rolling resume, final-test
access, and performance-based early termination are forbidden.

The command plan contains exactly one dependency-free GH200 job named
`hcwce60_train_M0CE60`. The source campaign is authenticated read-only and
has no scheduler or mutation edge to the control. Live execution requires the
canonical dry-run ledger and exact creation/submission phrases.

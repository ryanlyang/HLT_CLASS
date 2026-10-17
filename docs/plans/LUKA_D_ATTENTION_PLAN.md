# Luka aligned-attention D ladder implementation plan

Authorized 2026-10-17: implement the complete two-arm experiment in
[the scientific proposal](../LUKA_D_LADDER_ATTENTION_DISTILLATION.md), without a
one-step scientific screen. This plan and the proposal are the scientific
authority; the new namespace is `luka_attention`, not a modification of fixed30.

Freeze last particle block (index7), head-mean pre-dropout probabilities, FP32
attention math/targets, epsilon1e-8 and forward KL averaged per query then per
jet. Auxiliary coefficient0.1, constant; control0. Both retain C25/P75 T2,
exact30 epochs, batch128, original coordinate seeds, masks and optimizer groups.
Six fresh fits descend from the same authenticated regular COARSE_U100:
LOGIT_D066/D033/D000 and ATTN_D066/D033/D000. No fusion, warm start, offline
deployment inputs, class rebalancing, truncation or performance-driven gate.

Reuse trial ALLJETS299234/74895 inputs and original50k secondary reporting.
The historical holdout was consumed and repurposed for TRAIN; no independent
test exists. The separate FullStat experiment remains untouched.

Implementation sequence: model/loss parity; immutable ragged TRAIN banks;
matched training/selected export; original-source anchor authentication;
production-worker technical miniature and full-support resource measurement;
reviewed dry/live Slurm DAG and runbook. Existing scientific source must remain
unchanged. Local checks are not remote resource acceptance. Queue readiness
means the user can queue a genuine SPORC debug gate; full science additionally
requires successful gate accounting, measured resources and separate exact-plan
authorization. No jobs are submitted by implementation.

Native forwards remain untouched: a hook reads the final attention module's
actual inputs/masks and reconstructs its packed Q/K projections for the FP32
auxiliary surface. This adds a small final-block projection, not a second model
forward. Support explicitly checked Torch MHA and installed Weaver Attention;
unknown structures fail. Test the probability surface against the module's
native dropout-disabled operation, and zero-weight logits/gradients/updates
against the uninstrumented model. Production environment bytes are recorded.

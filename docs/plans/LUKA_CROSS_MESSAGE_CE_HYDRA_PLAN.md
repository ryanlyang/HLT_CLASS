# Cross-message architecture controls without privileged supervision

The user authorized two additional controls alongside the running Hydra
cross-message campaign. This additive study tests the width32 and width16
HLT message architectures with ground-truth cross entropy alone. Existing
CE+U100-w5 controls remain the matched controls for the message-distillation
question; these new runs test what the architecture achieves without richer
training supervision.

## Registered science

Tasks are `m32_ce_only_s0` and `m16_ce_only_s0`. Both start from scratch,
using the same native, predictor, expansion and sampler seeds as their
`m32_ce_s0` and `m16_ce_s0` parent controls. They use the same authenticated
TRAIN299234 and validation74895 D000 jets, batch128, thirty epochs, AdamW,
learning-rate schedule, BF16 forward and full-validation AUC selection.
There is no early stopping or quality gate. Original50k remains overlapping,
report-only; there is no independent test.

The inherited message branch reads block4 contextual tokens (128), own HLT
features (17) and masked mean HLT context (128). LayerNorm and an MLP
273->128->width produce a per-particle message. A zero-initialized width->128
projection adds it before block5. The branch and backbone learn through CE.
Both the predictor and expansion stay in deployed HLT-only inference.

The U100 embedding head is removed before optimizer construction. No embedding
bank, U100 particles, oracle messages, teacher weights, pretrained student
weights or logit KD may enter training or its miniature. The sole loss is
`cross_entropy(logits, ground_truth)`. Historical predictions and provenance
metadata may be read for reporting/qualification, never as optimization targets.

The existing model/training implementation at
`199928da991f8a4194dcf42df009130cfcf84118` is unchanged. A separate package
subclasses its student architecture and reuses its forward, CE step, optimizer
and schedule. Scientific reports/recipe remain version1. Execution SPEC and
RESOURCE_CHECK are version2 to distinguish warmed timing from the original
cold-start estimate.

## Execution

Use the completed parent A6000 setup and the same pinned environment. Authenticate
the parent in its own original source process; do not import a different
worktree's source guard into the new process. Bind the parent spec, receipt,
acceptance, GPU/software identity and measured matching control resource requests.
Each new independent job requests one RTX A6000, six CPUs and32000MiB, using
the matching parent's measured minutes plus ten minutes for the integrated
CE-only miniature (within48h). No teacher or student scheduler dependencies,
no setup/target/reference rerun, and no login-host watcher are needed.

Before full training inside each allocation, load only HLT TRAIN/validation.
First run a disposable one-epoch CE-only production miniature, including
publication/reload, to warm the kernels and allocator. Then reseed and run a
fresh one-epoch production miniature on the identical worst-multiplicity full
and partial batches. Project only this second fit's training/validation timing.
Count all elapsed loading, warmup and measurement overhead once, plus900s;
retain the2x thirty-epoch projection and85% GPU/RAM limits. Memory peaks cover
both fits without a reset between them. Both reports and the explicit timing
policy are authenticated by RESOURCE_CHECK/v2. No estimate is clamped or
ignored when it exceeds the actual allocation. Weak finite validation never
fails. Reseed again for full training: neither technical fit contributes weights,
optimizer state or updates to the thirty-epoch scientific fit.

This corrects the execution failure in jobs145718/145719: v1 extrapolated the
first three cold-start batches to924/896 minutes, exceeding421/425-minute
allocations. Those projections do not establish actual full-fit runtimes. The
matching parent controls were measured after other kernels had already run.
New specs keep the same parent-derived requests; warmed measurements must still
fit them. Use a fresh source-bound root and retain the failed v1 diagnostics.

Use a new source-bound root with a two-job dry plan, exact submission journal,
exclusive claim and explicit CLI authorization. No writes, cancellations or
updates to parent campaign files or jobs are permitted. Completed receipts and
epoch snapshots support read-only results/progress versus M0HLT, U100-w5 and
the corresponding CE+embedding architecture control. Parent fits may still be
running; their provisional epochs are clearly distinguished from final results.
